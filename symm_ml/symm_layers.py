import pysindy as ps
import numpy
import jax.numpy as jnp
from jax import random, jit, vmap

from .symbolic_utils import symb_jac_from_feat, get_jax_from_symb

def L_out_X(lie_gens_out, lie_out_jac_X, fn_lib_X):
    '''
    L_symm = L_out - L_in

    The component of the lienar symmetry operator corresponding
    to the lie generators acting in the image space. The operator
    is built at sample points X.

    For the einsum, we use the convention:
        N:      N_pts (number of points, X)
        n,s:    n_out (output dimension)
        q:      n_lie_gen (number of lie generators)
        l:      n_lie_out_lib (size of lie dictionary)
        f:      n_fn_lib (number of fn dictionary elements)

    lie_gens_out: ndarray shape (n_lie_gen, n_out, n_lie_out_lib)
        coefficient of lie generators on the output space (lie_out)

    lie_out_jac_X: ndarray shape (N_pts, n_lie_out_lib, n_out)
        lie_out library jacobian evaluated at points X

    fn_lib_X: ndarray shape(N_pts, n_fn_lib)
        function, F library evaluated at points X
    '''
    return jnp.einsum('qnl,Nls,Nf->qNnsf',
                      lie_gens_out, 
                      lie_out_jac_X,
                      fn_lib_X)


def L_in_X(lie_gens_in, fn_jac_X, lie_in_lib_X, n_dim_out):
    '''
    L_symm = L_out - L_in

    The component of the lienar symmetry operator corresponding
    to the lie generators acting in the domain space. The operator
    is built at sample points X.

    F : R^m -> R^n
    F(x) = W_fn @ fn_lib(X)

    For the einsum, we use the convention:
        N:      N_pts (number of points, X)
        m:      n_in (input dimension)
        n,s:    n_out (output dimension)
        q:      n_lie_gen (number of lie generators)
        l:      n_lie_in_lib (size of lie dictionary)
        f:      n_fn_lib (number of fn dictionary elements)
        
        Note, there's a need for a dummy variable because 
        `L_in` and `L_out` are different shaped tensors. 

    lie_gens_in: ndarray shape (n_lie_gen, n_in, n_lie_in_lib)
        coefficient of lie generators on the output space (lie_in)

    fn_jac_X: ndarray shape (N_pts, n_fn_lib, n_in)
        fn library jacobian evaluated at points X

    lie_in_lib_X: ndarray shape(N_pts, n_lie_lib)
        lie_in library evaluated at points X
    '''
    return jnp.einsum('Nfm,qml,Nl,ns->qNnsf',
                      fn_jac_X, 
                      lie_gens_in,  
                      lie_in_lib_X,
                      jnp.eye(n_dim_out)
                      )


def L_hat_op_X(lie_gens_in, lie_gens_out, 
          lie_in_lib_X, lie_out_jac_X, 
          fn_lib_X, fn_jac_X, n_dim_out):
    '''
    L_symm = L_out - L_in

    The symmetry operator built at sample points X.
    '''
    L_o = L_out_X(lie_gens_out, lie_out_jac_X, fn_lib_X)
    L_i = L_in_X(lie_gens_in, fn_jac_X, lie_in_lib_X, n_dim_out)
    return L_o - L_i


def L_hat_jax(W_fn, symm_op):
    '''
    The action of the symmetry operator on the set of coefficients
    '''
    return jnp.einsum('qNnsf,sf->qNn', symm_op, W_fn)

class BaseSymmLayer:
    def __init__(self, 
                 n_dim_in=None, 
                 n_dim_out=None, 
                 fn_library=None, 
                 lie_in_library=None, 
                 lie_out_library=None, 
                 lie_in_generators=None, 
                 lie_out_generators= None):
        '''
        dF/dx(x) @ phi_0(xi)(x) - d\phi_1(xi)/dx (x) @ F(x)
        W_fn @ dfn_lib(x) @ W_0 @ lie_lib(x) - W_1 @ dlie_lib(x) @ W_fn @ fn_lib(x)
        
        Parameters: 
            `n_dim_in`: int
                size of the input dimension
            `n_dim_out`: int
                size of the output dimension
            `fn_library`: pysindy.FeatureLibrary
                library of features for the function to fit
            `lie_in_library`: pysindy.FeatureLibrary
                PySINDy library of features for the Lie Algebra acting in the 
                domain of the function F
            `lie_out_library`: pysindy.FeatureLibrary
                PySINDy library of features for the Lie Algebra acting in the 
                output domain of the function F
            `lie_in_generators`: ndarray (n_lie, n_dim_in?, N_lie_dict)
                Basis of Lie Algebra generators
            `lie_out_generators`: ndarray (n_lie, n_dim_out?, N_lie_dict)
                Basis of Lie Algebra generators
        '''
        self.n_dim_in = n_dim_in
        self.n_dim_out= n_dim_out
        
        self.fn_library = fn_library
        self.lie_in_library = lie_in_library
        self.lie_out_library = lie_out_library
        
        self.lie_in_generators = lie_in_generators
        self.lie_out_generators = lie_out_generators
        
        assert lie_in_generators.shape[0] == lie_out_generators.shape[0]
        
        self.n_lie_gens = lie_in_generators.shape[0]
        
        self._data_init = False
        
        
        # initialize functions
        self._make_fn_symbols()
        self._make_lie_symbols()
        self._make_jax_fn()
        self._make_jax_lie()
        
        self.W_fn = jnp.zeros((self.n_dim_out, self.n_fn_lib))

    def set_W_fn_(self, W_fn):
        '''Set coefficients'''
        self.W_fn = W_fn
        
    def set_lie_in_(self, lie_in_generators):
        '''Set Lie generator coefficients for the input domain'''
        self.lie_in_generators = lie_in_generators
        
    def set_lie_out_(self, lie_out_generators):
        '''Set Lie generator coefficients for the co-domain'''
        self.lie_out_generators = lie_out_generators
    
    def _make_fn_symbols(self):
        # TO-DO: GET RID OF MANDATORY SYMBOLIC JACOBIAN—JUST USE AUTODIFF!
        '''Create SymPy symbols for the function library'''
        self.fn_library.fit(jnp.zeros(self.n_dim_in))
        
        self.fn_feature_names = self.fn_library.get_feature_names()
        self.n_fn_lib = len(self.fn_feature_names)
        self.symb_fn_lib, self.symb_fn_jac = symb_jac_from_feat(feature_names=self.fn_feature_names, 
                                                   n_state=self.n_dim_in)
        return self.symb_fn_lib, self.symb_fn_jac
    
    def _make_lie_symbols(self):
        '''Create SymPy symbols for the Lie library'''
        self.lie_in_library.fit(jnp.zeros(self.n_dim_in))
        self.lie_out_library.fit(jnp.zeros(self.n_dim_out))
        
        # get feature names
        self.lie_in_feature_names = self.lie_in_library.get_feature_names()
        self.lie_out_feature_names = self.lie_out_library.get_feature_names()
        
        # number of features (size of library)
        self.n_lie_in_lib = len(self.lie_in_feature_names)
        self.n_lie_out_lib = len(self.lie_out_feature_names)
        
        # get sympy symbols
        self.symb_lie_in_lib, self.symb_lie_in_jac = symb_jac_from_feat(
                                                    feature_names=self.lie_in_feature_names, 
                                                    n_state=self.n_dim_in)
        # get sympy symbols
        self.symb_lie_out_lib, self.symb_lie_out_jac = symb_jac_from_feat(
                                                    feature_names=self.lie_out_feature_names, 
                                                    n_state=self.n_dim_out)
        
        return self.symb_lie_in_lib, self.symb_lie_in_jac, self.symb_lie_out_lib, self.symb_lie_out_jac
    
    def _make_jax_fn(self): 
        '''
        Convert SymPy symbols for function library 
        to JIT-compiled and vectorized JAX code
        '''
        self.jax_fn_lib, self.jax_fn_jac = get_jax_from_symb(self.symb_fn_lib, 
                                                             self.symb_fn_jac, 
                                                             vectorize=True)
        return self.jax_fn_lib, self.jax_fn_jac
    
    def _make_jax_lie(self): 
        '''
        Convert SymPy symbols for Lie
        to JIT-compiled and vectorized JAX code
        '''
        
        self.jax_lie_in_lib, self.jax_lie_in_jac = get_jax_from_symb(self.symb_lie_in_lib, 
                                                                     self.symb_lie_in_jac, 
                                                                     vectorize=True)
        self.jax_lie_out_lib, self.jax_lie_out_jac = get_jax_from_symb(self.symb_lie_out_lib, 
                                                                       self.symb_lie_out_jac, 
                                                                       vectorize=True)
    
    def _initialize_data(self, sample_points_in, sample_points_out=None):
        '''Initialize data to be used for building the discrete linear operator'''
        # evaluate the lie library and jacobian at the sample points
        self.lie_in_lib_sample = self.jax_lie_in_lib(sample_points_in)
        self.lie_jac_sample = self.jax_lie_in_jac(sample_points_in)
        

        # evaluate the fn library and jacobian at the sample points
        self.fn_lib_sample = self.jax_fn_lib(sample_points_in)
        self.fn_jac_sample = self.jax_fn_jac(sample_points_in)
        
        
        # TO-DO: update this for general input/output symmetries. 
        # I think this should be taking all of F(x) as in input.
        # i.e. W_fn @ fn_lib(x). Not just fn_lib(x). 
        # Fine for right now because it's not used, but generally we 
        # need to push these samples forward if we're going to consider
        # compositions of symmetry layers. 
        sample_points_out = self.fn_lib_sample
        self.lie_out_lib_sample = self.jax_lie_out_lib(sample_points_out)
        self.lie_out_jac_sample = self.jax_lie_out_jac(sample_points_out)
        
        self.n_sample_pts = len(sample_points_in)
        self._data_init = True
    
    def _L_hat_wrapper(self):
        '''Wrapper for creating the L_hat operator from sampled data'''
        assert self._data_init, "Data not Initialized"
        
        @jit
        def L_hat_op_sample(gen_in, gen_out):
            return L_hat_op_X(gen_in, gen_out, 
                             self.lie_in_lib_sample, 
                             self.lie_out_jac_sample, 
                             self.fn_lib_sample, 
                             self.fn_jac_sample, 
                             self.n_dim_out)
            
        self.L_hat_op_sample = L_hat_op_sample

        self.symm_op = self.L_hat_op_sample(self.lie_in_generators, 
                                            self.lie_out_generators)
        @jit
        def L_hat(W_fn):
            '''
            W_fn: ndarray shape(n_out, n_fn_lib)
            
            Returns shape: 
                (n_lie_gen, n_sample_pts, n_out)
            '''
            return L_hat_jax(W_fn, self.symm_op)
        
        self.L_hat = L_hat
        
        @jit
        def project_generators(W_fn):
            '''Project the lie generators onto the SVD basis'''
            n_gen = self.n_lie_gens
            l_hat = self.L_hat(W_fn).reshape(n_gen, -1).T
            u,s,vh = jnp.linalg.svd(l_hat, full_matrices=False)
            
            new_basis = vh.T.conj()
            new_gens_in = jnp.einsum('qlm,qp->plm', self.lie_in_generators, new_basis)
            new_gens_out = jnp.einsum('qlm,qp->plm', self.lie_out_generators, new_basis)
            
            return new_gens_in, new_gens_out, s
        
        self.project_generators = project_generators
        

    def symbolic_L_hat(self, W_fn, lie_gen_idx=0):
        '''SymPy expression for L_hat
        
        Inputs: 
            `W_fn`: ndarray()
                weights for F: R^m -> R^n library
                F(x) = W_fn @ lib_fn(x)
            `lie_gen_idx`: int
                lie generator index
        Outputs:
            sympy expression of L_hat
        '''
        W = numpy.array(W_fn)
        gen_in = numpy.array(self.lie_in_generators[lie_gen_idx])
        gen_out = numpy.array(self.lie_out_generators[lie_gen_idx])
        
        return W @ self.symb_fn_jac @ gen_in @ self.symb_lie_in_lib - gen_out @ self.symb_lie_out_jac @ W @ self.symb_fn_lib
