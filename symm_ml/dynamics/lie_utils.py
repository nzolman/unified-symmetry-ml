from jax import numpy as jnp
from jax import scipy as jscp

from symm_ml.numerics.utils import se_n, so_n

def duplicate(matrix_list, d=3):
    '''Create block-diagonal duplicates'''
    return jnp.array([jscp.linalg.block_diag(*[M for i in range(d)]) for M in matrix_list])

def normalize_gens(W):
    n_gens = W.shape[0]
    W_0 = W.copy()
    for i in range(n_gens):
        norm = jnp.linalg.norm(W_0[i], ord = 'fro')
        W_0 = W_0.at[i].divide(norm)
    return W_0
    

def so3_state_system(n_particles):
    return duplicate(so_n(3), d=2*n_particles)

def so3_system_copy(n_particles, n_copies): 
    so3_base = so3_state_system(n_particles)
    so3_copy = duplicate(so3_base, n_copies)
    return so3_copy
    
def se3_system_copy(n_particles, n_copies): 
    so3_base = so3_state_system(n_particles)
    _, base_size, _  = so3_base.shape
    so3_copy = duplicate(so3_base, n_copies)
    _, n_out, _ = so3_copy.shape
    
    n_gens = 6
    
    W_0 = jnp.zeros((n_gens,n_out,n_out+1))
    W_0 = W_0.at[:3,:,1:].set(so3_copy)
    # include translation generators
    
    for system in range(n_copies):
        for i in range(n_particles):
            idx = 3*i + base_size*system
            translation_idx = jnp.array([[3,4,5],[idx,idx+1,idx+2],[0,0,0]])
            W_0 = W_0.at[translation_idx[0], 
                        translation_idx[1],
                        translation_idx[2]].set(1)
    
    for i in range(n_gens):
        norm = jnp.linalg.norm(W_0[i], ord = 'fro')
        W_0 = W_0.at[i].divide(norm)
    return W_0

def so2_xy_system_copy(n_particles, n_copies):
    so3_copy = so3_system_copy(n_particles, n_copies)
    return so3_copy[:1]
