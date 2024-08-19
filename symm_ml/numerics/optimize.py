import jax.numpy as jnp
from jax import random
import cvxpy as cp
import numpy as np
import pysindy as ps

from symm_ml.symm_layers import BaseSymmLayer

from symm_ml.numerics.utils import n_poly_points, se_n, t_n


def fit_data_sparse(n, Theta_train, Y_train, solver_kwargs={}):
    '''
    Fit the sparse model (L1 regularization)
    '''
    xi = cp.Variable((n,1)) 

    prob = cp.Problem(cp.Minimize(cp.norm1(xi)),
                    [Theta_train @ xi == Y_train])
    prob.solve(**solver_kwargs)
    return xi.value

def setup_symm(n_vars, n_deg, sample_seed=0, sample_bounds=[-1,1], lie='se_n'):
    '''Setup the symmetry model'''
    
    # create linear generators
    if lie == 'se_n':
        W_0 = se_n(n_vars, normalize=True)
    elif lie == 't_n':
        W_0 = t_n(n_vars, normalize=True)
    else:
        raise KeyError(f'invalid lie: {lie}')
    n_gen = len(W_0)
    W_1 = jnp.zeros((n_gen,1,1))

    # create libraries
    lie_in_lib = ps.PolynomialLibrary(degree=1, include_bias=True)
    lie_out_lib = ps.PolynomialLibrary(degree=1, include_bias=False)
    fn_lib = ps.PolynomialLibrary(degree=n_deg, include_bias=True)

    # initialize model
    symm_model = BaseSymmLayer(n_dim_in=n_vars,
                                n_dim_out=1,
                                lie_in_library = lie_in_lib,
                                lie_out_library = lie_out_lib,
                                fn_library=fn_lib, 
                                lie_in_generators = W_0,
                                lie_out_generators= W_1)
        
    
    # sample points from the domain to define the L_hat operator
    if lie == 't_n':
        N_sample = n_poly_points(symm_model.fn_library.degree-1, 
                                symm_model.n_dim_in) + 1 
    else:
        N_sample = n_poly_points(symm_model.fn_library.degree, 
                                symm_model.n_dim_in) + 1 

    r_key = random.PRNGKey(sample_seed)
    sample_data = random.uniform(r_key, shape=(N_sample, symm_model.n_dim_in), 
                                minval=sample_bounds[0], 
                                maxval=sample_bounds[1]
                                )

    symm_model._initialize_data(sample_data)
    symm_model._L_hat_wrapper()
    
    return symm_model