import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from jax import config
config.update("jax_enable_x64", False)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from symm_ml.dynamics.experiment_runner import NodeExperimentRunner, get_pend_data
from symm_ml import _parent_dir

_data_dir = os.path.join(_parent_dir, 'data', 'node_exps', 'node')

if __name__ == '__main__':
    from tqdm import tqdm
    
    seed = 0
    data_seed = 42
    save_freq = 10
    n_steps = int(1e4)
    n_trajs = [5, 10, 25, 50, 100, 250, 500][::-1]
    nonlinearity  = jnp.tanh
    n_hidden = 42
    n_layers = 2
    mlp_layer_sizes = [12] + (n_layers + 1) * [n_hidden] + [12]
    symm_rep = None # 'so3' # None
    n_sample = int(1e5)
    gamma = 0 # 1e-6
    
    OVERWRITE = False
    _data_dir = os.path.join(_parent_dir, 'data', 'node_exps', f'mlp-{symm_rep}')
    os.makedirs(_data_dir, exist_ok=True)
    
    for n_traj in tqdm(n_trajs):
        for seed in range(10):
            
            print('-'*50)
            print(seed, n_traj)
            save_path = os.path.join(_data_dir,f'n_traj={n_traj}_seed={seed:02}.npy')
            if os.path.exists(save_path) and not OVERWRITE:
                print(f'Skipping. Path exists: {save_path}')
                continue
            
            train_data, val_data, test_data = get_pend_data(data_seed + seed,
                                                            use_normal=True)

            train_data = train_data[:n_traj]
            symm_kwargs = {'n_sample': n_sample,
                           'bounds_max': jnp.abs(train_data).max()
                           }
            mlp_exp = NodeExperimentRunner(
                                seed=seed, 
                                symm=symm_rep, 
                                use_mlp=True, 
                                layer_sizes=mlp_layer_sizes, 
                                nonlinearity = nonlinearity,
                                symm_kwargs=symm_kwargs,
                                lr = 3e-4,
                                symm_reg = gamma
            )

            config = mlp_exp.to_config()
            config['n_traj'] = n_traj
                        
            mlp_params_list, mlp_losses = mlp_exp.train(mlp_exp.params_0, 
                                                        train_data[:n_traj], 
                                                        save_freq = save_freq, 
                                                        tot_steps=n_steps,  
                                                        verbose = True)
            
            val_losses_mlp = jnp.array([mlp_exp.node.data_loss(p, 
                                                                val_data[:,0], 
                                                                val_data) for p in mlp_params_list])
            
            config['val_losses'] = val_losses_mlp
            
            jnp.save(save_path, config)