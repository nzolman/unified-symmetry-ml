from jax import jit, random, vmap, jacrev
from jax.experimental.ode import odeint
import jax.numpy as jnp

from symm_ml.dynamics.nn_utils import random_layer_params, init_network_params, get_mlp
from symm_ml.dynamics.lie_utils import so3_system_copy, se3_system_copy, so2_xy_system_copy

class MLPNODE:
    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        # self.sizes = sizes
        self.mlp = get_mlp(self.nonlinearity)
        self.mlp_batched = vmap(self.mlp, in_axes=(None, 0))
        self.T_max = 1.0
        self.dt = 0.2
        
        self._setup_fns()
        
        self.init_params = init_network_params
        
    def _setup_fns(self):
        @jit
        def node_vf(x0,t,params):
            return self.mlp(params, x0)

        @jit
        def node_vf_batched(X0,t,params):
            return self.mlp_batched(params, X0)

        @jit
        def node_predict(params, x0):
            return odeint(node_vf, x0, jnp.arange(0,self.T_max,self.dt), params)

        @jit
        def node_predict_batched(params, X0):
            return odeint(node_vf_batched, X0, jnp.arange(0,self.T_max,self.dt), params).transpose(1,0,2)

        @jit
        def data_loss(params, X0, X): 
            preds = node_predict_batched(params, X0)
            mse = jnp.mean((preds[:,1:] - X[:,1:])**2)
            return mse

        self.node_vf = node_vf
        self.node_vf_batched = node_vf_batched
        self.node_predict = node_predict
        self.node_predict_batched = node_predict_batched
        
        self.data_loss = data_loss
        
        return None



class SymmMLPNODE(MLPNODE):
    def __init__(self, nonlinearity, rep_type='so3',bounds_max=2, n_sample = int(1e5)):
        super().__init__(nonlinearity)
        self.rep_type = rep_type
        self.bounds_max = bounds_max
        self.n_sample = n_sample
        
        if rep_type == 'se3':
            self.lie_gen_fn = se3_system_copy
            self.include_rep_bias = True
        elif rep_type == 'so3':
            self.lie_gen_fn = so3_system_copy
            self.include_rep_bias = False
        elif rep_type == 'so2':
            self.lie_gen_fn = so2_xy_system_copy
            self.include_rep_bias = False
        
    def _setup_symm(self, key):
        dim_in = 12 # hardcoding for now
        N_sample = self.n_sample
        X_sample = random.uniform(key, minval=-self.bounds_max, maxval=self.bounds_max,
                        shape=(N_sample, dim_in))
        
        mlp_jac = jacrev(self.mlp,
                        argnums=1)
        mlp_jac_v = vmap(mlp_jac,
                        in_axes=(None,0))
                
        # hard-coding for now
        lie_gens = self.lie_gen_fn(n_particles=2, n_copies=1)

        lie_X = jnp.einsum('qij,Nj->Niq',lie_gens, X_sample)

        @jit
        def lhat_mlp(params):
            df_X = mlp_jac_v(params, X_sample)
            f_X = self.mlp_batched(params, X_sample)
            dlie_fX = jnp.einsum('qij,Nj->Niq',lie_gens, f_X)
            df_lieX = jnp.einsum('Nij,Njq->Niq', df_X, lie_X)
            
            lhat = df_lieX - dlie_fX
            lhat_flat = jnp.concatenate(lhat)
            return lhat_flat
        @jit
        def symm_loss(params):
            lhat_flat = lhat_mlp(params)
            return jnp.linalg.norm(lhat_flat, ord='nuc')
        
        @jit        
        def total_loss(params, X0, X, gamma=1e-4):
            mse_loss = self.data_loss(params, X0, X)
            lhat_loss = symm_loss(params)
            return mse_loss + gamma * lhat_loss
        
        self.lhat_mlp = lhat_mlp
        self.symm_loss = symm_loss
        self.total_loss = total_loss
        
        return None
