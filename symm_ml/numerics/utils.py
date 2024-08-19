from scipy.special import comb
import jax.numpy as jnp


def n_poly_points(d,m):
    '''Calculate number of points needed to approximate a d-degree multivariate polynomial in m-variables.'''
    return int(sum([comb(k+m-1, m-1) for k in range(d+1)]))

def so_n(n):
    '''Helper function for creating Lie algebra for SO(n)'''
    N_gen = n*(n-1)//2 # dim of SO(n) manifold
    gens = jnp.zeros((N_gen,n,n))

    triu_list = jnp.array(jnp.triu_indices(n,k=1)).T

    for i, idx in enumerate(triu_list):
        gens = gens.at[i,idx[0], idx[1]].set(1)
    gens = gens - gens.transpose(0,2,1)
    return gens

def se_n(n_vars, normalize=True):
    '''Helper function for creating Lie algebra for SE(n)'''
    son = so_n(n_vars)
    n_son = len(son)
    n_gen = n_son + n_vars
    W_0 = jnp.zeros((n_gen,n_vars,n_vars+1))
    W_0 = W_0.at[:n_son,:,1:].set(son)

    vars_idx = jnp.arange(n_vars)

    translation_idx = jnp.array([n_son + vars_idx,
                                vars_idx,
                                jnp.zeros_like(vars_idx)
        
    ])
    W_0 = W_0.at[translation_idx[0], 
                translation_idx[1],
                translation_idx[2]].set(1)
    
    if normalize:
        norms = jnp.linalg.norm(W_0, 'fro', axis=(1,2)).reshape(-1,1,1)
        W_0 = W_0 = W_0 / norms
    return W_0

def t_n(n_vars, normalize=True):
    '''Helper function for creating Lie algebras for the translation group T(n)'''
    W_0 = se_n(n_vars, normalize=normalize)[-n_vars:]
    return W_0