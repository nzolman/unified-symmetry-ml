'''
Script for finding the statistical dimension for the dictionary recovery using 
a symmetry prior as a convex optimization problem. Uses multiprocessing via
tqdm.contrib.concurrent.process_map
'''

import os
import yaml
import json
from itertools import product
from datetime import datetime
from pprint import pprint
import numpy as np
import cvxpy as cp
import jax

from tqdm.contrib.concurrent import process_map

from symm_ml import _parent_dir
from symm_ml.numerics.recover_opt import BinaryRecovery, SymmRecoverOpt


# ---------------------------------------------------------
# Experiment variables
# ---------------------------------------------------------
FN_CAT = 'test'         # category of function to consider (ortho, quad)
LIE = 't_n'             # lie algebra to use (t_n, se_n)
THRESH = 5e-3           # l_infty threshold for determining recovery
N_WORKERS = 10          # number of workers to run in parallel
# ---------------------------------------------------------


class NumpyEncoder(json.JSONEncoder):
    '''Helper class for handling the encoding numpy arrays in a JSON'''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
  


# ---------------------------------------------------------
# determine tuples of (m, r, d, seed) to use for the experiment
# m: input dimension
# r: latent variable dimension
# d: degree of polynomial
# seed: random seed to use
# ---------------------------------------------------------
if FN_CAT == 'test':
    FN_CAT = 'ortho'
    N_WORKERS = 3
    DEG_DICT = {2: range(5,20,5),
                3: range(6,8),
                4: range(5,7)
            }
    N_SEEDS = 10
    R_RANGE = range(1,4)
    SEED_RANGE = range(N_SEEDS)
    exp_tuples = []
    for d in DEG_DICT: 
        exp_tuples += [(m,r,d,seed) for (m,r,seed) in product(DEG_DICT[d], R_RANGE, SEED_RANGE)]  


elif FN_CAT == 'ortho' and LIE == 't_n': 
    DEG_DICT = {2: range(5,55,5),
                3: range(6,17),
                4: range(5,11)
            }

    N_SEEDS = 10
    R_RANGE = range(1,4)
    SEED_RANGE = range(N_SEEDS)
    exp_tuples = []
    for d in DEG_DICT: 
        exp_tuples += [(m,r,d,seed) for (m,r,seed) in product(DEG_DICT[d], R_RANGE, SEED_RANGE)]
    
elif FN_CAT == 'ortho' and LIE == 'se_n':
    DEG_DICT = {2: range(5,35,5),
                3: range(6,17),
                4: range(5,11)
                }
    N_SEEDS = 10
    R_RANGE = range(1,4)
    SEED_RANGE = range(N_SEEDS)
    exp_tuples = []
    for d in DEG_DICT: 
        exp_tuples += [(m,r,d,seed) for (m,r,seed) in product(DEG_DICT[d], R_RANGE, SEED_RANGE)]

elif FN_CAT == 'quad' and LIE == 'se_n':
    DEG_DICT = {
                1: range(5,35,5),
                2: range(5,11),
                }
    N_SEEDS = 10
    SEED_RANGE = range(N_SEEDS)
    R_RANGE=[1,2]
    exp_tuples = []
    for d in DEG_DICT: 
        # Note: r = d
        for r in R_RANGE: 
            if r <= d:
                exp_tuples += [(m,r,d,seed) for (m,r,seed) in product(DEG_DICT[d], R_RANGE, SEED_RANGE)]

OPT_KWARGS = {'solver':cp.SCS, 'max_iters': 2500}

ROOT_DIR = os.path.join(_parent_dir, 'output', f'{FN_CAT}_{LIE}')
LOG_DIR  = os.path.join(ROOT_DIR, 'logs')

disable_jit = True

def log_experiment(save_path, bin_opt, model):
    '''dump data as a json'''
    bin_data = bin_opt.get_data()
    filtration_data = model.get_data()
    
    bin_data.update(filtration_data)
    with open(save_path, "w") as outfile: 
        json.dump(bin_data, outfile, 
                  cls=NumpyEncoder, indent=4, separators=(',', ': ')
                  )

def exp_wrapper(params):
    '''perform the recovery experiment'''
    m, r, d, seed = params
    
    save_dir = os.path.join(ROOT_DIR,
                                f'r={r:04d}',
                                f'd={d:04d}',
                                f'm={m:04d}'
                            )
    
    save_path = os.path.join(save_dir, f'{seed}.json')
    
    if not os.path.exists(save_path):
        with jax.disable_jit(disable_jit): # sometimes it's faster/more memory efficeint to not jit           
            model = SymmRecoverOpt(input_dim=m, feature_dim=r, phi_deg=d, fn_cat=FN_CAT, lie_alg=LIE, seed=seed,
                                    opt_kwargs= OPT_KWARGS)
            model._init_problem()
            bin_opt = BinaryRecovery(model, thresh=THRESH)
            recovery_dim = bin_opt.fit()
            
            log_experiment(save_path, bin_opt, model)
        return recovery_dim
    else:
        return None
    
def setup_dirs():
    # setup m,r,d directories
    for d, m_range in DEG_DICT.items():
        for r in R_RANGE:
            for m in m_range:
                md_dir = os.path.join(ROOT_DIR, 
                                    f'r={r:04d}',
                                    f'd={d:04d}',
                                    f'm={m:04d}'
                                    )
                os.makedirs(md_dir, exist_ok= True)
    
    # setup logdir
    os.makedirs(LOG_DIR, exist_ok= True)
    
if __name__ == '__main__':
    from copy import deepcopy
    # Initialize all the directories
    setup_dirs()

    opt_kwargs = deepcopy(OPT_KWARGS)
    opt_kwargs['solver'] = str(OPT_KWARGS['solver'])
    log_dict = {'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'num_workers': N_WORKERS,
                'fn_cat': FN_CAT,
                'lie_algebra': LIE,
                'thresh': THRESH,
                'seeds': SEED_RANGE,
                'r_range': R_RANGE,
                'degree_dict': DEG_DICT,
                'optimizer_kwargs': opt_kwargs
                }

    # save log
    log_path = os.path.join(LOG_DIR, datetime.now().strftime("%Y-%m-%d_T%H:%M:%S") + '.yaml')
    with open(log_path, 'w') as f:
        yaml.dump(log_dict, f)
        
    pprint(log_dict)
    
    # Iterate through everything    
    process_map(exp_wrapper, exp_tuples, max_workers=N_WORKERS)