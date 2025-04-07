
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
from jax import config
config.update("jax_enable_x64", False)
# config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from symm_ml.dynamics.experiment_runner import NodeExperimentRunner, get_pend_data
train_data, val_data, test_data = get_pend_data(use_normal=True)

from jax import nn
from tqdm import tqdm

symm_layer_sizes = [3,2,2,1]

oversample = 4

tot_steps = int(1e4)

nonlinearity = nn.sigmoid 
n_traj = 100

gammas = jnp.logspace(-5.5,-3.5,20)

all_val_losses = []
for gamma in tqdm(gammas):
    exp = NodeExperimentRunner(
                        seed=0, 
                        symm='so2', 
                        use_mlp=False, 
                        layer_sizes=symm_layer_sizes, 
                        nonlinearity = nonlinearity,
                        symm_kwargs={'bounds_max': jnp.abs(train_data[:n_traj]).max(),
                                    'enforce': False,
                                    'oversample': oversample},
                        lr = 3e-4,
                        symm_reg = gamma,
    )

    params_list, losses = exp.train(exp.params_0, 
                                    train_data[:n_traj], 
                                    save_freq = 10, 
                                    tot_steps=tot_steps,  
                                    verbose = True)
    
    val_losses = jnp.array([exp.node.data_loss(p, val_data[:,0], val_data) for p in params_list])
    
    all_val_losses.append(val_losses)


    jnp.save('2025-04-05_so2_sweep-refined.npy', 
             {'all_val_losses': all_val_losses, 'gammas': gammas}
            )
    print(f'{gamma:.02e}',
          val_losses.min())