import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
from jax import config
config.update("jax_enable_x64", False)

import jax.numpy as jnp
from jax import nn
import matplotlib.pyplot as plt

from symm_ml.dynamics.experiment_runner import NodeExperimentRunner, get_pend_data
from symm_ml import _parent_dir


if __name__ == '__main__':
    from tqdm import tqdm
    
    seed = 0
    data_seed = 42
    save_freq = 10
    n_steps = int(1e4)
    LR = 3e-4
    gamma = 1e-4
    nonlinearity = nn.sigmoid #jnp.tanh
    
    enforce = False
    symm = 'so3'
    OVERWRITE = False

    layer_archs = [
        [12,1],
        [3,3,1],
        [3,2,2,1],
        [4,2,1],
        [5,1,1],
        [2,2,2,2,1],
    ]

    n_traj = 100
    for seed in range(10):
        for symm_layer_sizes in tqdm(layer_archs):
            
            layers_string_list = [str(i) + '-' for i in symm_layer_sizes]
            layers_string = "".join(layers_string_list)[:-1]
            _data_dir = os.path.join(_parent_dir, 'data', 'node_exps', 
                                    f'symm={symm}_enforce={enforce}_layers={layers_string}')
            os.makedirs(_data_dir, exist_ok=True)
        
            print('-'*50)
            print(seed, symm_layer_sizes)
            save_path = os.path.join(_data_dir,f'n_traj={n_traj}_seed={seed:02}.npy')
            if os.path.exists(save_path) and not OVERWRITE:
                print(f'Skipping. Path exists: {save_path}')
                continue
            
            train_data, val_data, test_data = get_pend_data(data_seed + seed,
                                                            use_normal=True)
            
            # restrict to n_traj
            train_data = train_data[:n_traj]
            
            symm_exp = NodeExperimentRunner(
                                seed=seed, 
                                symm=symm, 
                                use_mlp=False, 
                                layer_sizes=symm_layer_sizes, 
                                nonlinearity = nonlinearity,
                                symm_kwargs={
                                    'bounds_max': jnp.abs(train_data).max(),
                                    'enforce': enforce
                                },
                                lr = LR,
                                symm_reg = gamma
            )

            config = symm_exp.to_config()
            config['n_traj'] = n_traj
                        
            params_list, losses = symm_exp.train(symm_exp.params_0, 
                                                        train_data, 
                                                        save_freq = save_freq, 
                                                        tot_steps=n_steps,  
                                                        verbose = True)
            
            val_losses = jnp.array([symm_exp.node.data_loss(p, 
                                                            val_data[:,0], 
                                                            val_data) for p in params_list])
            
            config['val_losses'] = val_losses
            
            jnp.save(save_path, config)