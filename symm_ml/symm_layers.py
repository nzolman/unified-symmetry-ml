from scipy.special import comb

import pysindy as ps
import numpy
import jax.numpy as jnp
from jax import random, jit, vmap, jacfwd
import jax

from symm_ml.symbolic_utils import symb_jac_from_feat, get_jax_from_symb, get_jax_fnlib_from_symb

def n_poly_points(d,m):
    '''Calculate number of points needed to approximate a d-degree multivariate polynomial in m-variables.'''
    return int(sum([comb(k+m-1, m-1) for k in range(d+1)]))


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

class DummyLayer:
    def __init__(self, n_dim_in, **kwargs):
        self.n_dim_in = n_dim_in
        self.n_dim_out = 1
        self.n_fn_lib = 0

    def _initialize_data(self, 
                         sample_points_in=None, 
                         sample_points_out=None):
        pass
    def L_hat(self, W_fn):
        return jnp.array([])
    
    def _L_hat_wrapper(self):
        pass

class BaseSymmLayer:
    def __init__(self, 
                 n_dim_in=None, 
                 n_dim_out=None, 
                 fn_library=None, 
                 lie_in_library=None, 
                 lie_out_library=None, 
                 lie_in_generators=None, 
                 lie_out_generators= None,
                 jac_from_symb = False):
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
        self._make_fn_symbols(jac_from_symb=jac_from_symb)
        

        self._make_lie_symbols(jac_from_symb=jac_from_symb)
        self._make_jax_fn(jac_from_symb=jac_from_symb)
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
    
    def _make_fn_symbols(self, jac_from_symb=False):
        # TO-DO: GET RID OF MANDATORY SYMBOLIC JACOBIAN—JUST USE AUTODIFF!
        '''Create SymPy symbols for the function library'''
        self.fn_library.fit(jnp.zeros(self.n_dim_in))
        
        self.fn_feature_names = self.fn_library.get_feature_names()
        self.n_fn_lib = len(self.fn_feature_names)
        

        self.symb_fn_lib, self.symb_fn_jac = symb_jac_from_feat(
                                                feature_names=self.fn_feature_names, 
                                                n_state=self.n_dim_in,
                                                return_jac = jac_from_symb
                                                )
        
        return self.symb_fn_lib, self.symb_fn_jac
    
    def _make_lie_symbols(self, jac_from_symb = False):
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
                                                    n_state=self.n_dim_in,
                                                    return_jac=jac_from_symb)
        # get sympy symbols
        self.symb_lie_out_lib, self.symb_lie_out_jac = symb_jac_from_feat(
                                                    feature_names=self.lie_out_feature_names, 
                                                    n_state=self.n_dim_out,
                                                    return_jac=jac_from_symb)
        
        return self.symb_lie_in_lib, self.symb_lie_in_jac, self.symb_lie_out_lib, self.symb_lie_out_jac
    
    def _make_jax_fn(self, jac_from_symb=False): 
        '''
        Convert SymPy symbols for function library 
        to JIT-compiled and vectorized JAX code
        '''
        
        # jaxify lib + jacobian from sympy
        if jac_from_symb:
            self.jax_fn_lib, self.jax_fn_jac = get_jax_from_symb(self.symb_fn_lib, 
                                                                self.symb_fn_jac, 
                                                                vectorize=True)
        # jaxify lib from sympy + jacobian from autodiff
        else:
            self.jax_fn_lib = get_jax_fnlib_from_symb(self.symb_fn_lib,vectorize=False)
            self.jax_fn_jac = jit(jacfwd(self.jax_fn_lib))
            
            # vectorize
            self.jax_fn_lib = vmap(self.jax_fn_lib, in_axes=(0,))
            self.jax_fn_jac = vmap(self.jax_fn_jac, in_axes=(0,))
            
        
        return self.jax_fn_lib, self.jax_fn_jac
    
    def _make_jax_lie(self, from_symb=False): 
        '''
        Convert SymPy symbols for Lie
        to JIT-compiled and vectorized JAX code
        '''
        
        # jaxify lib + jacobian from sympy
        if from_symb:
            self.jax_lie_in_lib, self.jax_lie_in_jac = get_jax_from_symb(self.symb_lie_in_lib, 
                                                                        self.symb_lie_in_jac, 
                                                                        vectorize=True)
            self.jax_lie_out_lib, self.jax_lie_out_jac = get_jax_from_symb(self.symb_lie_out_lib, 
                                                                        self.symb_lie_out_jac, 
                                                                        vectorize=True)
        # jaxify lib from sympy + jacobian from autodiff
        else:
            self.jax_lie_in_lib = get_jax_fnlib_from_symb(self.symb_lie_in_lib, vectorize=False)
            self.jax_lie_out_lib = get_jax_fnlib_from_symb(self.symb_lie_out_lib, vectorize=False)
            
            self.jax_lie_in_jac = jit(jacfwd(self.jax_lie_in_lib))
            self.jax_lie_out_jac = jit(jacfwd(self.jax_lie_out_lib))
            
            self.jax_lie_in_lib = vmap(self.jax_lie_in_lib, in_axes=(0,))
            self.jax_lie_out_lib = vmap(self.jax_lie_out_lib, in_axes=(0,))
            
            self.jax_lie_in_jac = vmap(self.jax_lie_in_jac, in_axes=(0,))
            self.jax_lie_out_jac = vmap(self.jax_lie_out_jac, in_axes=(0,))
            
    
    def _initialize_data(self, sample_points_in, sample_points_out=None):
        '''Initialize data to be used for building the discrete linear operator'''
        # evaluate the lie library and jacobian at the sample points
        # self.sample_points_in = sample_points_in
        # self.sample_points_out = sample_points_out
        
        self.lie_in_lib_sample = self.jax_lie_in_lib(sample_points_in)
        self.lie_in_jac_sample = self.jax_lie_in_jac(sample_points_in)
        

        # evaluate the fn library and jacobian at the sample points
        self.fn_lib_sample = self.jax_fn_lib(sample_points_in)
        self.fn_jac_sample = self.jax_fn_jac(sample_points_in)
        
        
        # TO-DO: update this for general input/output symmetries. 
        # I think this should be taking all of F(x) as in input.
        # i.e. W_fn @ fn_lib(x). Not just fn_lib(x). 
        # Fine for right now because it's not used, but generally we 
        # need to push these samples forward if we're going to consider
        # compositions of symmetry layers.
        
        # TO-DO: There may be an error now if we don't explicitly 
        # pass sample_points_out
        if sample_points_out is None:
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



class BasicSymmBlock:
    ''' 
    Convenient wrapper for a symmetry block consisting of an equivariant linear layer: 
        z = Ax + b 
    and an invariant quadratic layer
        y = (1 x) Q (1 x)^T
    The output of the layer is of the form:
        x_out = sigma(y) * z
    Where sigma is a nonlinearity. A la Finzi, this is a gated nonliearity.
    '''
    def __init__(self, input_rep, output_rep, nonlinearity=jax.nn.elu, include_rep_bias = False, no_quad = False):
        '''
        input_rep: ndarray(n_gens, n_in, n_lie_in)
            input representation of the lie generators. For a linear rep,
            we expect n_lie_in = n_in. For an affine rep, we expect
            n_lie_in = n_in + 1
        output_rep: ndarray(n_gens, n_out, n_lie_out)
            output representation of the lie generators. For a linear rep,
            we expect n_lie_out = n_out. For an affine rep, we expect
            n_lie_out = n_out + 1
        nonlinearity: function
            scalar-valued (vectorized is fine) nonlinearity. NOTE: 
            for deep networks, it may be beneficial to use a bounded nonlinearity. 
        include_rep_bias: bool
            Whether to include the bias term for the lie library 
            i.e. whether the lie representations are affine (true)
            or linear (false). 
            NOTE: this is probably redundant since
            we are currently forcing the form to linear/affine, so we can get
            this value from the input/output reps
        '''
        
        self.no_quad = no_quad
        
        # number of generators
        self.n_gens =input_rep.shape[0]
        assert self.n_gens == output_rep.shape[0]
        
        self.input_size = input_rep.shape[1]
        self.output_size = output_rep.shape[1]
        
        self.nonlinearity = nonlinearity
        self.input_rep = input_rep
        self.output_rep = output_rep
        self.scalar_rep = jnp.zeros((self.n_gens,1,1)) # used for quadratic layer
        
        self.include_rep_bias = include_rep_bias

        self.lin_layer = BaseSymmLayer(
                                n_dim_in=self.input_size,
                                n_dim_out=self.output_size,
                                lie_in_library = ps.PolynomialLibrary(
                                                        degree=1, 
                                                        include_bias=include_rep_bias
                                                        ),
                                lie_out_library = ps.PolynomialLibrary(
                                                        degree=1, 
                                                        include_bias=include_rep_bias
                                                        ),
                                fn_library= ps.PolynomialLibrary(
                                                        degree=1, 
                                                        include_bias=True
                                                        ),
                                lie_in_generators = self.input_rep,
                                lie_out_generators= self.output_rep
                                )
        self.n_dim_in = self.lin_layer.n_dim_in
        if self.no_quad:
            self.quad_layer = DummyLayer(n_dim_in=self.n_dim_in)
        else: 
            self.quad_layer = BaseSymmLayer(
                                            n_dim_in=self.input_size,
                                            n_dim_out=1,
                                            lie_in_library=ps.PolynomialLibrary(
                                                                degree=1, 
                                                                include_bias=include_rep_bias
                                                                ),
                                            lie_out_library=ps.PolynomialLibrary(
                                                                degree=1, 
                                                                include_bias=include_rep_bias
                                                                ),
                                            fn_library=ps.PolynomialLibrary(
                                                                degree=2, 
                                                                include_bias=True
                                                                ),
                                            lie_in_generators=self.input_rep,
                                            lie_out_generators=self.scalar_rep
                                            )
        
        
        self.symm_layers = [self.lin_layer, self.quad_layer]
        self.layer_names = ['lin', 'quad']
        self.transform = self._get_fn()
        self.transform_v = self._get_fn_v()
        
    def init_layers(self, key, data_min=None, data_max=None):
        '''
        Initialize the layers with sample points from the domain
        Arguments:
            key: jax.random.PRNGKey
                random key
            data_min, data_max: array-like
                minimum and maximum values to sample from the domain.
        '''
        r_key = key.copy()
        for i, (layer, name) in enumerate(zip(self.symm_layers, self.layer_names)):
            if not isinstance(layer, DummyLayer):
                N_sample =  n_poly_points(layer.fn_library.degree, 
                                        self.n_dim_in) + 1
            else:
                N_sample = 1

            if data_min is None or data_max is None:
                data_min = -2
                data_max = 2
            
            # TO-DO figure out how to choose minval and maxval for layers
            sample_data_in = random.uniform(
                                        r_key, 
                                        shape=(N_sample, layer.n_dim_in), 
                                        minval= data_min, 
                                        maxval= data_max, 
                                        )
            r_key, _ = random.split(r_key)
            # print('in_split', r_key, name)
            
            # NOTE: for linear (or affine) Lie algebra generators, this is really just a formality for now. 
            # We only use the lie_gen_out library for it's jacobian, which does not depend on the data.
            # However, we do need the same number of data points in order to get this to all work out.
            # In the future, if we have nonlienar generators, we might need something like this.
            out_data_bounds = 1.0
            sample_data_out = random.uniform(
                                        r_key, 
                                        shape=(N_sample, layer.n_dim_out), 
                                        minval= -out_data_bounds,
                                        maxval= out_data_bounds,
                                        )
            
            r_key, _ = random.split(r_key)
            # print('out split', r_key, name)
    
            layer._initialize_data(sample_data_in, 
                                   sample_data_out)
            layer._L_hat_wrapper()
            
        self.lhat = self._get_lhat()
        return r_key
    
    def initialize_parameters(self, key, bounds=[-0.01,0.01]):
        '''
        Convenience function for initializing the parameters of the layer.
        
        Arguments: 
            key: jax.random.PRNGKey
                random key
        
        TO-DO: Open up bounds
        '''
        minval, maxval = bounds
        r_key  = key.copy()
        params = []
        for name, layer in zip(self.layer_names, self.symm_layers):
            l_params = random.uniform(r_key, minval=minval, maxval=maxval, 
                                    shape = (layer.n_dim_out, layer.n_fn_lib))
            params.append(l_params)
        # params = {}
        # for name, layer in zip(self.layer_names, self.symm_layers):
        #     l_params = random.uniform(r_key, minval=-0.01, maxval=0.01, 
        #                             shape = (layer.n_dim_out, layer.n_fn_lib))
        #     params[name] = l_params
            r_key, _ = random.split(r_key)
            # print('params', r_key,name)
        return r_key, params

    def _get_fn_v(self):
        if self.no_quad:
            @jit
            def fn(x, block_params):
                W_lin, W_quad = block_params
                
                # W_lin = block_params['lin']
                # W_quad = block_params['quad']
                
                # first layer
                x0 = x
                z1 = (W_lin @ self.lin_layer.jax_fn_lib(x0).T).T     # vector
                x1 =  z1
                
                return x1
        else:
            @jit
            def fn(x, block_params):
                W_lin, W_quad = block_params
                
                # W_lin = block_params['lin']
                # W_quad = block_params['quad']
                
                # first layer
                x0 = x
                z1 = (W_lin @ self.lin_layer.jax_fn_lib(x0).T).T     # vector
                y1 = (W_quad @ self.quad_layer.jax_fn_lib(x0).T).T   # scalar
                x1 =  z1 * self.nonlinearity(y1)
                
                return x1
        return fn


    def _get_fn(self):
        if self.no_quad:
            @jit
            def fn(x, block_params):
                W_lin, W_quad = block_params
                
                # W_lin = block_params['lin']
                # W_quad = block_params['quad']
                
                # first layer
                x0 = x.reshape(1,-1)
                z1 = (W_lin @ self.lin_layer.jax_fn_lib(x0).T).T     # vector
                x1 =  z1
                
                return x1[0]
        else:
            @jit
            def fn(x, block_params):
                W_lin, W_quad = block_params
                
                # W_lin = block_params['lin']
                # W_quad = block_params['quad']
                
                # first layer
                x0 = x.reshape(1,-1)
                z1 = (W_lin @ self.lin_layer.jax_fn_lib(x0).T).T     # vector
                y1 = (W_quad @ self.quad_layer.jax_fn_lib(x0).T).T   # scalar
                x1 =  z1 * self.nonlinearity(y1)
                
                return x1[0]
        return fn
    
    def _get_lhat(self):
        
        @jit
        def lhat(block_params):
            W_lin, W_quad = block_params
            l_hats = [] 
            n_lie_gens = self.lin_layer.n_lie_gens
            lhat_lin = self.lin_layer.L_hat(W_lin).reshape(n_lie_gens,-1)
            lhat_quad = self.quad_layer.L_hat(W_quad).reshape(n_lie_gens,-1)
            l_hats = [lhat_lin, lhat_quad]
                
            big_l_hat = jnp.concatenate(l_hats, axis=-1)
            return big_l_hat

        return lhat
    
class BlockLayer:
    def __init__(self, 
                 num_blocks, 
                 input_rep, output_rep, 
                 nonlinearity=jax.nn.elu, 
                 include_rep_bias = True,
                 out_layer = False):
        self.out_layer = out_layer
        self.num_blocks = num_blocks
        self.input_rep = input_rep
        self.output_rep = output_rep
        self.nonlinearity = nonlinearity
        
        self.blocks = [BasicSymmBlock(input_rep=self.input_rep, 
                                      output_rep=self.output_rep, 
                                      nonlinearity=self.nonlinearity, 
                                      include_rep_bias = include_rep_bias,
                                      no_quad=out_layer) 
                       for i in range(self.num_blocks)]
        
        self.input_size = self.blocks[0].input_size # sum([block.input_size for block in self.blocks])
        self.output_size = sum([block.output_size for block in self.blocks])
        # self.transform = self._get_fn()
        
    def init_layers(self, key, *bounds):
        r_key = key.copy()
        
        for ii, block in enumerate(self.blocks):
            r_key = block.init_layers(r_key, *bounds)
        
        self.lhat = self._get_lhat()
        return r_key
    
    def initialize_parameters(self, key, bounds=[-0.01, 0.01]):
        r_key = key.copy()
        layer_params = []
        for ii, block in enumerate(self.blocks):
            r_key, block_params = block.initialize_parameters(r_key, bounds)
            # print(r_key, block)
            layer_params.append(block_params)
        return r_key, layer_params
    
    def _get_lhat(self):
        @jit
        def lhat(layer_params):
            l_hats = [] 
            
            for block, block_params in zip(self.blocks, layer_params):
                lhat = block.lhat(block_params)
                l_hats.append(lhat)
            big_l_hat = jnp.concatenate(l_hats, axis=-1)
            return big_l_hat
        
        return lhat
    
    # def _get_fn(self):
    #     v_fn = vmap(self.blocks[0].transform, in_axes=(None,0), out_axes=(0))
        
    #     @jit
    #     def fn(x, layer_params):
    #         return jnp.concatenate(v_fn(x, layer_params))
    #     return fn

    
    def transform(self, x, layer_params):
        # NOTE: the way this is written, the transform is actually all the same, the need for independent blocks
        # comes from independent Lhats. Might just conconcatenate a vmap here. 
        # UPDATE: this requires reformatting the parameters. Might need to do something where we have a dict like:
        # {'lin':  [block_1_lin, block_2_lin, ...], 
        #  'quad': [block_1_quad, block_2_quad, ...]}
        return jnp.concatenate([block.transform(x, block_params) for block_params, block in zip(layer_params, self.blocks)])


    def transform_v(self, x, layer_params):
        # NOTE: the way this is written, the transform is actually all the same, the need for independent blocks
        # comes from independent Lhats. Might just conconcatenate a vmap here. 
        # UPDATE: this requires reformatting the parameters. Might need to do something where we have a dict like:
        # {'lin':  [block_1_lin, block_2_lin, ...], 
        #  'quad': [block_1_quad, block_2_quad, ...]}
        return jnp.concatenate([block.transform_v(x, block_params) for block_params, block in zip(layer_params, self.blocks)], axis=-1)

