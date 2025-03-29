from jax import jit, random, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from symm_ml.dynamics.nn_utils import random_layer_params, init_network_params, get_mlp


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
