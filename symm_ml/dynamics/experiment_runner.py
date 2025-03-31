import os
from tqdm import tqdm
import optax
from jax import random, tree, value_and_grad, jit
from jax import numpy as jnp
from symm_ml.dynamics.node import MLPNODE
from symm_ml.dynamics.symm_node import SymmNet, get_tot_params

from symm_ml import _parent_dir
_pend_data_dir  = os.path.join(_parent_dir, 'datasets/ODEDynamics/double_spring_pendulum/')

def get_pend_data(use_normal = True):
    if use_normal:
        path = os.path.join(_pend_data_dir, 'trajectories_1500_5_0.2_30.pz.npy')
    else:
        path = os.path.join(_pend_data_dir, 'trajectories_1500_5_0.2_30_no-y=True.pz.npy')
    data = jnp.load(path)

    train = data[:500]
    val = data[500:1000]
    test = data[1000:]
    
    return train, val, test    

class NodeExperimentRunner:
    def __init__(self, seed=0, symm=None, use_mlp=False, 
                 layer_sizes=None, nonlinearity = jnp.tanh,symm_kwargs=None,
                 lr = 3e-4,
                 symm_reg = 0):
        
        self.seed = seed
        self.key = random.PRNGKey(seed)
        self.symm = symm
        self.use_mlp = use_mlp
        self.layer_sizes = layer_sizes
        self.nonlinearity=nonlinearity
        self.symm_kwargs = symm_kwargs or {}
        self.symm_reg = symm_reg
        self.lr = lr
        self.n_state = 12 # hard coded because SymmNet currently hard coded. 
        # self.dyn_dim = dyn_dim
        
        assert symm in ['so3', 'se3', 'so2', None], f'Invalid symm value: {symm}'
        
        self.setup_net()
        self.optimizer, self.opt_state, self.train_step_fn = self.setup_train_step()
        
    def setup_net(self):
        if self.use_mlp and not self.symm:
            self.node = MLPNODE(self.nonlinearity)
            self.params_0 = self.node.init_params(self.layer_sizes, self.key)
            self.key, _ = random.split(self.key)
            
            self.n_params = sum([p.size for p in tree.leaves(self.params_0)])
            # self.loss_fn = 
            
        else:
            if not self.symm:
                symm_rep = 'so2'
            else: 
                symm_rep = self.symm
            self.node = SymmNet(n_blocks_per_layer=self.layer_sizes,
                                rep_type=symm_rep,
                                nonlinearity=self.nonlinearity,
                                **self.symm_kwargs
                                )
            self.key, _ = self.node.setup_layers(self.key)
            self.key, self.params_0 = self.node.init_params(self.key)
            self.n_params = get_tot_params(self.layer_sizes, self.n_state)
            self.node._setup_fns()
        
        if self.symm:
            self.loss_fn = self.node.total_loss
        else:
            self.loss_fn = self.node.data_loss
            
        
    def to_config(self):
        config = {'seed': self.seed,
                  'symm': self.symm,
                  'use_mlp': self.use_mlp,
                  'layer_sizes': self.layer_sizes,
                  'n_params': self.n_params,
                  'symm_reg': self.symm_reg,
                  'lr': self.lr
                  }
        return config

    def setup_train_step(self):
        init_params = self.params_0
        optimizer = optax.adam(learning_rate=self.lr)
        opt_state = optimizer.init(init_params)

        v_g = value_and_grad(self.loss_fn)
        
        if self.symm:
            @jit
            def train_step_fn(params, opt_state, batch, labels):
                loss_value, grads = v_g(params, batch, labels, self.symm_reg)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, opt_state, loss_value
        else:
            @jit
            def train_step_fn(params, opt_state, batch, labels):
                loss_value, grads = v_g(params, batch, labels)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, opt_state, loss_value
        
        return optimizer, opt_state, train_step_fn

    def train(self, init_params, data, save_freq = 1, tot_steps=10000,  verbose = True):
        opt_state = self.optimizer.init(init_params)

        params_list = [init_params]
        new_params = init_params
        
        losses = []
        for i in tqdm(range(tot_steps), disable=not verbose):
            new_params, opt_state, loss_value = self.train_step_fn(new_params, 
                                                                   opt_state, 
                                                                   data[:,0], 
                                                                   data)
            if i % save_freq == 0: 
                params_list.append(new_params)
                losses.append(loss_value)
        return params_list, losses