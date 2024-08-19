from symm_ml import jnp_float

from sympy import sympify, Matrix
from jax import jit, vmap
import jax.numpy as jnp

from sympy2jax import SymbolicModule


def symb_jac_from_feat(feature_names, n_state):
    '''
    Take a list of string expressions, convert them to sympy symbols in 
        `n_state` variables and compute the Jacobian.
    
    Inputs:
        `feature_names`: list[str]
            List of string expressions to convert. e.g.
                ['x', 'x^2', 'sin(x)']
        `n_state`: int
            Number of variables to define
    Returns:
        `symb_lib`: sympy form of the feature library
        `symb_jac`: sympy form of the Jacobian Matrix
    '''
    feature_names = [feat.replace(' ', '*') for feat in feature_names]
    state_names = [f"x{i}" for i in range(n_state)]
    symb_state = [sympify(state) for state in state_names]
    symb_features = [sympify(feature) for feature in feature_names]
    symb_lib = Matrix([symb_features]).T
    symb_jac = symb_lib.jacobian(symb_state)
    return symb_lib, symb_jac

def get_jax_from_symb(symb_lib, symb_jac, vectorize=True):
    '''
    Returns JAX (jitted) versions of the symbolic form
    
    Inputs:
        `symb_lib`:
            sympy form of the library 
        `symb_jac`:
            sympy form of the jacobian matrix
        `vectorize`: bool
            whether to wrap in vmap
    Returns:
        `feat_lib_fn`: function
            jit-compiled pointiwse library function
        `feat_jac_fn`: function
            jit-compiled pointwise Jacobian function
    '''
    n_dict, n_state = symb_jac.shape
    jax_feat_lib = SymbolicModule([symb for symb in symb_lib])
    jax_feat_jac = SymbolicModule([symb for symb in symb_jac])
    
    @jit
    def feat_lib_fn(x):
        eval_dict = {f'x{i}': x[i] for i in range(x.shape[0])}
        return jnp.array(jax_feat_lib(**eval_dict), dtype=jnp_float)
    
    @jit
    def feat_jac_fn(x):
        eval_dict = {f'x{i}': x[i] for i in range(x.shape[0])}
        return jnp.array(jax_feat_jac(**eval_dict), dtype=jnp_float).reshape(n_dict, n_state)
    
    if vectorize:
        return vmap(feat_lib_fn, in_axes=(0)), vmap(feat_jac_fn, in_axes=(0))
    return feat_lib_fn, feat_jac_fn