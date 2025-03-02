# from jax import config
# config.update("jax_enable_x64", True)

from diffrax import ODETerm, SemiImplicitEuler, diffeqsolve, SaveAt
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax import random

def gen_dyn(key, n_particles=10, n_dim=3, g = 1.0, k_bounds = (0.5,1.0)):
    # spring mass matrices
    L = random.uniform(key, minval=k_bounds[0], maxval=k_bounds[1], shape=(n_particles, 1))
    
    # make symmetric
    K = L @ L.T
    
    # no self-interaction
    K = K.at[jnp.diag_indices(n_particles)].set(0)

    # gravity is in the last dimension
    gravity_dir = jnp.zeros(n_dim)
    gravity_dir = gravity_dir.at[-1].set(g)

    def dq(t,p,args):
        return p

    def dp(t,q,args):
        K_tilde = jnp.diag(K.sum(axis=0))
        return (-K_tilde + K) @q - gravity_dir
    
    return (dq, dp), K

def get_data(key, dq, dp, K, n_dim = 3, dt = 0.01, T = 100, zero_dir=False):
    n_particles = K.shape[0]
    
    terms = (ODETerm(dq),
            ODETerm(dp)
            )
    solver = SemiImplicitEuler()
    
    key1, key2, key3, key4 = random.split(key, 4)
    
    global_q0 = random.normal(key1, shape=(n_dim,))
    global_p0 = random.normal(key2, shape=(n_dim,))
    
    q0 = global_q0 + random.normal(key3, shape=(n_particles,n_dim)) 
    p0 = 0.1*(global_p0 + random.normal(key4, shape=(n_particles,n_dim)))
    
    if zero_dir:
        q0 = q0.at[:,1].set(0.0)
        p0 = p0.at[:,1].set(0.0)
    
    y0 = (q0,p0)

    saveat = SaveAt(ts=jnp.arange(0,T,dt))
    sol = diffeqsolve(terms=terms, solver=solver, t0=0, t1=T+1, dt0=1e-5, y0=y0, 
                      saveat=saveat, max_steps=None)

    q, p = sol.ys
    return (q, p)