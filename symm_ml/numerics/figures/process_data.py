from tqdm import tqdm
import os
import glob
import numpy as np
import pandas as pd
import json

from symm_ml.numerics.utils import n_poly_points
from symm_ml import _parent_dir

def save_data(exp_name):
    path = os.path.join(_parent_dir, 'output', exp_name)

    save_dir = os.path.join(_parent_dir, 'data')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{exp_name}.csv')

    filepaths =glob.glob(os.path.join(path, '**', '*.json'), recursive=True)
    filepaths.sort()

    quad = ('quad' in exp_name)

    datas = []
    for fp in tqdm(filepaths):
        with open(fp, 'r') as f:
            datas.append(json.load(f))

    symm_df = pd.DataFrame(
        [{'m': data['m'], 'r': data['r'], 'd': data['d'], 'seed': data['seed'],
        'n_recover': np.min(data['N_high']),
        'n_lib': n_poly_points(d=data['d'] * (1+quad), m=data['m']
                                )}
        for data in datas
        ])


    symm_df['dt'] = np.array([np.sum([data['info'][key]['dt'] 
                    for key in data['info'].keys()])
            for data in datas
    ])


    symm_df['tot_iters'] = np.array([np.sum([data['info'][key]['n_iter'] 
                    for key in data['info'].keys()])
            for data in datas
    ])

    symm_df.to_csv(save_path, index=False)
    
    return symm_df, save_path

if __name__ == '__main__':
    for exp_name in ['ortho_t_n', 'ortho_se_n', 'quad_se_n']:
        symm_df, save_path = save_data(exp_name)