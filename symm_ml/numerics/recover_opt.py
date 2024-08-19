import jax.numpy as jnp

import json
import time
import numpy as np
import cvxpy as cp
from tqdm import tqdm


from symm_ml.numerics.optimize import setup_symm
from symm_ml.numerics.lib_coefs import get_lib_fn

def filtration_domain(input_dim, max_sample_pts, sample_bounds=[-1.0, 1.0]):
    '''
    Setup random input values. Ensures that if n_sample_1 < n_sample_2 points are used
    during a binary search for a particular instantiation of a random seed, then the
    input domain X_1 is a subset of X_2 for |X_i| = n_sample_i.
    
    Arguments:
        input_dim: int
            dimension of the input space, X. dim(X) = m
        max_sample_points:
            maximum number of points willing to store in memory.
        sample_bounds:
            endpoints of the m-dimensional cube to sample from.
    Returns: 
        ndarray(max_sample_points, input_dim)
    '''
    return np.random.uniform(*sample_bounds, size=(max_sample_pts, input_dim))

def filtration_range(domain_data, poly_lib, true_coef):
    '''
    Dictionary and polynomial evaluated on a domain.
    
    Arugments:
        domain_data: ndarray(n_points, input_dim)
            input data to evaluate on.
        poly_lib: pysindy.PolynomialLibrary
            dictionary library to evaluate
        true_coef:
            coefficients for the polynomial
    Returns:
        Theta_train: ndarray(n_points, n_lib)
            Dictionary Library evaluations
        Y_train: ndarray (n_points, 1)
            polynomial evaluations
    '''
    X_train = domain_data
    Theta_train =  poly_lib.transform(X_train)
    Y_train = Theta_train @ true_coef
    return Theta_train, Y_train.reshape(-1,1)


def fit_data_symm(Theta_train, Y_train, 
                  X_train = None, 
                #   symm_model=None, 
                symm_loss_fn = None,
                solver_kwargs = {}, 
                return_solver=False):
    '''
    Train the symmetry model with CVXPY
    
    Arguments:
        Theta_train: ndarray(n_points, n_lib)
            dictionary evaluations on library
        Y_train: (n_points, 1)
            output evaluations on library
        X_train: (n_points, input_dim)
            input points in library (unused)
        symm_loss_fn: cvxpy loss function
            symmetry loss function for the problem
        solver_kwargs: dict
            dictionary keyword arugments to pass the CVXPY solver
        return_solver: bool (default False)
            whether to return the solver from the cvxpy optimization
            (mainly used for examining number of iterations during testing/debugging)
    
    Returns:
        W_pred: coefficients from optimization
        cp_prob (optionally): CVXPY solver object
            if `return_solver` is True
    '''

    n_fn_lib = Theta_train.shape[1] # symm_model.n_fn_lib
    n_dim_out = Y_train.shape[1] # symm_model.n_dim_out


    # define the cvxpy problem
    W_fn_flat = cp.Variable((n_fn_lib * n_dim_out,1))
    # objective = cp.Minimize(cvxpy_trainer.symm_loss(W_fn_flat))
    objective = cp.Minimize(symm_loss_fn(W_fn_flat))
    cp_prob = cp.Problem(objective,
                        [Theta_train @ cp.reshape(W_fn_flat, (n_fn_lib, n_dim_out), order='C') == Y_train])

    cp_prob.solve(**solver_kwargs)
    W_pred = W_fn_flat.value.reshape((n_fn_lib, n_dim_out))
    
    if return_solver:
        return W_pred, cp_prob
    else:
        return W_pred
    
    
def fit_data_sparse(Theta_train, Y_train, solver_kwargs={}, return_solver=False):
    '''
    Fit the sparse model (L1 regularization)
    
    Arguments:
        Theta_train: ndarray(n_points, n_lib)
            dictionary evaluations on library
        Y_train: (n_points, 1)
            output evaluations on library
        solver_kwargs: dict
            dictionary keyword arugments to pass the CVXPY solver
        return_solver: bool (default False)
            whether to return the solver from the cvxpy optimization
            (mainly used for examining number of iterations during testing/debugging)
    '''
    n_fn_lib = Theta_train.shape[1]
    n_dim_out = Y_train.shape[1]
    W_fn_flat = cp.Variable((n_fn_lib * n_dim_out,1))

    cp_prob = cp.Problem(cp.Minimize(cp.norm1(W_fn_flat)),
                    [Theta_train @ cp.reshape(W_fn_flat, (n_fn_lib, n_dim_out), order='C') == Y_train])
    cp_prob.solve(**solver_kwargs)
    W_pred = W_fn_flat.value.reshape((n_fn_lib, n_dim_out))
    
    if return_solver:
        return W_pred, cp_prob
    else:
        return W_pred

    
def get_symm_loss_fn(model):
    '''
    Initialize symmetry loss with cvxpy
    
    Arugments:
        model: symm_ml.symm_layers.BaseSymmLayer
            symmetry model.
    Returns:
        symm_loss: cvxpy expression
            Nuclear norm of the appropriately reshaped L_hat operator
    '''
    n_gen = len(model.lie_in_generators)
    n_fn_lib = model.n_fn_lib
    n_dim_out = model.n_dim_out
    n_pts = model.n_sample_pts
    
    # Transpose Operator
    symm_op_flat = jnp.einsum('qNnsf->sfqNn', model.symm_op)
    
    # Concatenate first two indices
    # sfqNn -> FqNn, 
    # size(F) = s*f
    symm_op_flat = jnp.concatenate(symm_op_flat, axis=0)
    
    # Transpose again
    symm_op_flat = jnp.einsum('FqNn->qNnF', symm_op_flat)
    
    # Concatenate first THREE indicies:
    # qNnF -> QF
    # size(Q) = q*N*n
    symm_op_flat = jnp.concatenate(
                            jnp.concatenate(symm_op_flat, axis=0),
                        axis=0)
    
    def symm_loss(W_flat):
        '''Assume W_flat is W_fn.flatten()'''
        
        # QF,F->Q (=qNn)
        l_hat_flat = symm_op_flat @ W_flat
        l_hat = cp.reshape(l_hat_flat, (n_gen, n_pts * n_dim_out), order='C').T
        return cp.norm(l_hat, p='nuc')
    
    return symm_loss


class BaseRecoverOpt:
    '''Abstract Class for performing a recovery optimization'''
    def __init__(self, input_dim, feature_dim, phi_deg, fn_cat='ortho', seed=0, opt_kwargs=None):
        '''
        Arguments:
            `input_dim`: int
                The input dimension, m
            `feature_dim`: int
                number of latent variables, r
            `phi_deg`: int
                degree of polynomial
            `fn_cat`: str
                function category: either "ortho", an orthogonal projection onto r variables
                    or "quad", where r-latent variables are obtained as y_i = ||x - c_i||^2 
                    for i = 1, 2, ... r and c_i are randomly chosen vectors
            seed: int
                random seed
            opt_kwargs: dict
                cvxpy optimization keyword arguments to be passed to the solver.
        '''
        self.m = input_dim
        self.r = feature_dim
        self.d = phi_deg
        self.fn_cat = fn_cat
        self.opt_kwargs = opt_kwargs or {}
        self.seed = seed
        
    def _init_problem(self):
        '''Initialize the optimization problem'''
        np.random.seed(self.seed)
        
        # get the coefficients for the polynomial
        self.coefs, self.poly_lib =  get_lib_fn(input_dim=self.m, 
                                                feature_dim=self.r, 
                                                poly_deg=self.d, 
                                                fn_cat=self.fn_cat)
        self.gt_coefs = np.array(self.coefs, dtype=np.float64).reshape((-1,1))
        self.max_sample_pts = len(self.gt_coefs) + 1

        # build the set of random data we'll be using
        self.X_train_filtration = filtration_domain(input_dim=self.m, 
                                                    max_sample_pts=self.max_sample_pts)
        self.Theta_train_filtration, self.Y_train_filtration = filtration_range(self.X_train_filtration, 
                                                                           poly_lib=self.poly_lib, 
                                                                           true_coef=self.coefs)
    def fit(self, n_pts):
        pass
    
    def save(self):
        pass
    
    def infinity_norm(self, coefs):
        '''Relative infinity norm ||W - W_true||_\infty / ||W_true||_\infty'''
        gt_max = np.abs(self.gt_coefs).max()
        return np.abs(coefs - self.gt_coefs).max() / gt_max
    
    def get_model_info(self):
        '''Data used for logging'''
        return {
            'm': self.m,
            'r': self.r,
            'd': self.d,
            'fn_cat': self.fn_cat,
            'seed': self.seed,
            'opt_kwargs': self.opt_kwargs, 
            'gt_coefs': self.gt_coefs,
            'lib_terms': len(self.gt_coefs)
        }
        
    def get_data(self):
        return {
            'X_train_filtration': self.X_train_filtration,
            'Y_train_filtration': self.Y_train_filtration,
            'Theta_train_filtration': self.Theta_train_filtration
        }
         

class SymmRecoverOpt(BaseRecoverOpt):
    '''Symmetry recovery class'''
    def __init__(self, input_dim, feature_dim, phi_deg, fn_cat='ortho', seed=0, opt_kwargs=None, 
                 lie_alg = 'se_n', sample_bounds= [-1,1]):
        '''
        Arguments:
            `input_dim`: int
                The input dimension, m
            `feature_dim`: int
                number of latent variables, r
            `phi_deg`: int
                degree of polynomial
            `fn_cat`: str
                function category: either "ortho", an orthogonal projection onto r variables
                    or "quad", where r-latent variables are obtained as y_i = ||x - c_i||^2 
                    for i = 1, 2, ... r and c_i are randomly chosen vectors
            seed: int
                random seed
            opt_kwargs: dict
                cvxpy optimization keyword arguments to be passed to the solver.
        '''
        
        super().__init__(input_dim, feature_dim, phi_deg, fn_cat=fn_cat, seed=seed, opt_kwargs=opt_kwargs)
        self.lie_alg = lie_alg
        self.disc_sample_bounds = sample_bounds
        
        self.opt_kwargs = self.opt_kwargs or {'solver':cp.SCS, 
                                              'max_iters': 2500}
        
    def _init_problem(self):
        super()._init_problem()
        self.symm_model = setup_symm(n_vars = self.m, 
                                    n_deg = self.poly_lib.degree, 
                                    sample_bounds = self.disc_sample_bounds, 
                                    lie=self.lie_alg, 
                                    sample_seed = self.seed)
        
        self.symm_loss_fn = get_symm_loss_fn(self.symm_model)
    
    def fit(self, n_pts):
        '''
        
        Fit the optimization problem with the first `n_pts` number of
        training points from the filtration. 
        
        Returns:
            res: ndarray
                fitted coefficients
            info: dict
                dictionary of metrics.
        '''
        Theta_train = self.Theta_train_filtration[:n_pts]
        Y_train = self.Y_train_filtration[:n_pts]
        X_train = self.X_train_filtration[:n_pts]
        
        t0 = time.time()
        res, cp_prob = fit_data_symm(Theta_train, Y_train, X_train, 
                            symm_loss_fn=self.symm_loss_fn,
                            solver_kwargs=self.opt_kwargs, 
                            return_solver=True)
        dt = time.time() - t0
        
        loss = self.infinity_norm(res)
        n_iter = cp_prob.solver_stats.num_iters
        info = {'loss': loss, 'n_iter': n_iter, 'dt': dt}
        return res, info

    def get_model_info(self):
        info =  super().get_model_info()
        extra_info = {'lie_alg': self.lie_alg}
        info.update(extra_info)
        return info
    
    def check_sigmas(self, coefs = None):
        '''Check the number of singular values below a cutoff'''
        if coefs is None:
            coefs = self.gt_coefs
        self.sigmas = jnp.linalg.svd(self.symm_model.L_hat(coefs.T)[:,:,0].T)[1]
        return np.sum(self.sigmas/self.sigmas.max() < 1e-5)


class L1RecoverOpt(BaseRecoverOpt):
    def __init__(self, input_dim, feature_dim, phi_deg, fn_cat='ortho', seed=0, opt_kwargs=None):
        super().__init__(input_dim, feature_dim, phi_deg, fn_cat=fn_cat, seed=seed, opt_kwargs=opt_kwargs)
        
        self.opt_kwargs = self.opt_kwargs or {'solver':cp.SCS, 
                                              'max_iters': 2500}
        
    def fit(self, n_pts):
        Theta_train = self.Theta_train_filtration[:n_pts]
        Y_train = self.Y_train_filtration[:n_pts]
        X_train = self.X_train_filtration[:n_pts]
        
        t0 = time.time()
        res, cp_prob = fit_data_sparse(Theta_train, Y_train, 
                                       solver_kwargs=self.opt_kwargs,
                                       return_solver=True)
        dt = time.time() - t0
        
        loss = self.infinity_norm(res)
        n_iter = cp_prob.solver_stats.num_iters
        info = {'loss': loss, 'n_iter': n_iter, 'dt': dt}
        return res, info


class BinaryRecovery:
    '''Perfom a binary search for the recovery dimension for a given recovery optimization problem'''
    def __init__(self, recovery_opt, n_pts_range=None, max_search = 15, thresh = 1e-2):
        '''
        Arguments:
            recovery_opt: BaseRecoveryOpt
                The recovery optimization problem. In practice, SymmRecoverOpt or L1RecoverOpt.
            n_pts_range: list
                range of points to consider when bounding the search [n_low, n_high].
                By default, we use a loose bound of [0, n_lib+1], though in theory one could
                strategically choose these bounds to speed up convergence.
            max_search: int
                the maximum number of times you would perform a search.
            thresh: float
                threshold used to declare success for recovery. 
        '''

        self.recovery_opt = recovery_opt
        self.max_search = max_search
        self.thresh = thresh
        
        if n_pts_range is None:
            self.low_guess = 1
            self.high_guess = len(self.recovery_opt.gt_coefs) + 1
        else: 
            self.low_guess, self.high_guess = n_pts_range    
        
        self.N_train_high = [self.high_guess]   # list of num trial points that are above the threshold
        self.N_train_low = [self.low_guess]     # list of num trial points that are below the threshold
        self.N_train_list = []                  # list of all attempted number of trial points
        self.info_dict = {}
        
    def fit(self, verbose = False):
        
        for n_search in tqdm(range(self.max_search), disable=not verbose):
            
            # define new ranges to look at
            N_hi =  self.N_train_high[-1]
            N_low = self.N_train_low[-1]
            
            # select new number of trial points
            N_train = (N_low + N_hi)//2
            
            # Check if finished
            if ((N_hi - N_low) <= 1  # No middle values
                or (N_train == N_hi) # Repeated value
                or (N_train == N_low)# Repeated value
                ):
                break
                
            # fit the coefficients for the given number of training points
            coef, info = self.recovery_opt.fit(N_train)
            
            self.info_dict[N_train] = {'coef': coef, **info}
                
            self.N_train_list.append(N_train)
            
            if info['loss'] < self.thresh:
                self.N_train_high.append(N_train)
            else:
                self.N_train_low.append(N_train)
                
        return np.min(self.N_train_high)

    def get_data(self):
        data = {'N_list': self.N_train_list,
                'N_high': self.N_train_high,
                'N_low': self.N_train_low,
                'info': self.info_dict,
                **self.recovery_opt.get_model_info()
                }
        return data
    
    def save(self, fpath):
        data = self.get_data()
        
        # Convert and write JSON object to file
        with open(fpath, "w") as outfile: 
            json.dump(data, outfile)
        
        return data