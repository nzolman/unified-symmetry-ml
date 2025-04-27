
import jax
from jax import jit, vmap, random
from jax import numpy as jnp
from jax import scipy as jscp
from jax.experimental.ode import odeint
import jaxopt

from symm_ml.symm_layers import BaseSymmLayer, BasicSymmBlock, BlockLayer, CopyBlockLayer, n_poly_points


from symm_ml.dynamics.lie_utils import so3_system_copy, se3_system_copy, so2_xy_system_copy

# ODE_TOL = dict(atol=2e-6, rtol=2e-6)
ODE_TOL = {}

def get_tot_params(n_blocks_per_layer, n_state):
    m = n_state
    tot_params = 0
    for n_blocks in n_blocks_per_layer[:-1]:
        tot_params += n_blocks * n_state*(m+1) # linear terms
        tot_params += n_blocks*n_poly_points(2,m) # quad terms
        m = n_blocks * n_state
    tot_params += n_state*(m+1) # final linear terms

    return tot_params

class SymmNet:

    def __init__(self, n_blocks_per_layer, 
                 rep_type='so3', 
                 bounds_max=2, 
                 param_bounds = [-0.1,0.1], 
                 nonlinearity=jax.nn.tanh,
                 use_copy=True,
                 enforce = False,
                 oversample=1):
        
        self.bounds_max = bounds_max
        self.param_bounds = param_bounds
        self.n_blocks_per_layer = n_blocks_per_layer
        self.n_layers = len(n_blocks_per_layer) - 1 
        self.rep_type = rep_type
        self.nonlinearity = nonlinearity
        self.use_copy = use_copy
        self.oversample = oversample
        
        self.dt = 0.2
        self.T_max = 1.0
                
        
        if rep_type == 'se3':
            self.lie_gen_fn = se3_system_copy
            self.include_rep_bias = True
        elif rep_type == 'so3':
            self.lie_gen_fn = so3_system_copy
            self.include_rep_bias = False
        elif rep_type == 'so2':
            self.lie_gen_fn = so2_xy_system_copy
            self.include_rep_bias = False
            
        self.enforce = enforce
    
    def init_params(self, key):
        all_params = []
        r_key = key.copy()
        for i in range(self.n_layers+1):
            layer = self.layers[i]
            if self.enforce:
                r_key, layer_params = layer.initialize_proj_parameters(r_key, 
                                                                        bounds=self.param_bounds)
            else:
                r_key, layer_params = layer.initialize_parameters(r_key, 
                                                                bounds=self.param_bounds)
            all_params.append(layer_params)
        return r_key, all_params
        

    def setup_layers(self, key):
        if self.use_copy:
            block_class = CopyBlockLayer
        else: 
            block_class = BlockLayer
        r_key = key.copy()
        
        layers = []
        for i in range(self.n_layers+1):
            if i == 0:
                n_copies = 1
                sample_bounds = [-self.bounds_max, self.bounds_max]
            else:
                n_copies = self.n_blocks_per_layer[i-1]
                sample_bounds = [-2,2]
                
            n_blocks = self.n_blocks_per_layer[i]
            rep_in = self.lie_gen_fn(n_particles=2, 
                                     n_copies=n_copies)
            rep_out = self.lie_gen_fn(n_particles=2, 
                                      n_copies=1)
            
            out_layer = False
            if i == self.n_layers:
                out_layer = True


            layer = block_class(n_blocks, 
                                input_rep=rep_in, 
                                output_rep=rep_out,  
                                nonlinearity=self.nonlinearity, 
                                include_rep_bias = self.include_rep_bias,
                                out_layer = out_layer,
                                enforce = self.enforce)
            
            
            # TO-DO: allow for oversampling
            r_key = layer.init_layers(r_key, *sample_bounds, oversample=self.oversample)
            
            layers.append(layer)
            
        self.layers = layers
        return r_key, layers
    
    def _setup_fns(self):
        layers = self.layers
        
        @jit
        def node(params, x):
            x_out = x
            for layer, l_params in zip(layers, params):
                x_out = layer.transform_v(x_out, l_params)
            return x_out
        
        @jit
        def lhat(params):
            l_hats = [] 
            symm_loss = 0
            n_lie_gens = layers[0].input_rep.shape[0]
            for (W, layer) in zip(params, layers):
                l_hat = layer.lhat(W).reshape(n_lie_gens,-1)
                l_hats.append(l_hat)
                
            big_l_hat = jnp.concatenate(l_hats, axis=-1)
            return big_l_hat

        @jit
        def symm_loss(params):
            big_l_hat = lhat(params)
            return jnp.linalg.norm(big_l_hat, ord = 'nuc')

        @jit
        def node_vf(x0, t, params):
            return node(params, x0)


        @jit
        def node_predict_batched(params, x0):
            # return odeint(node_vf, x0, jnp.arange(0,T_max,dt), params) # .transpose(1,0,2)
            return odeint(node_vf, x0, jnp.arange(0,self.T_max,self.dt), params,
                          **ODE_TOL).transpose(1,0,2)

        @jit
        def data_loss(params, X0, X):
            # X_pred = node_predict_batched(params, X0)
            X_pred = node_predict_batched(params, X0)
            return jnp.mean((X[:, 1:] - X_pred[:,1:])**2)

        @jit
        def total_loss(params, X0, X, gamma = 1e-3):
            mse = data_loss(params, X0, X)
            lhat_loss = symm_loss(params)
            return mse + gamma * lhat_loss
    
        self.lhat = lhat
        
        self.node = node
        self.node_vf = node_vf
        self.node_predict_batched = node_predict_batched
        self.symm_loss = symm_loss
        self.data_loss = data_loss
        self.total_loss = total_loss
        
        return None
    