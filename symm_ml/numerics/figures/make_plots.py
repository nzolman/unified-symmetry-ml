import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from symm_ml import _parent_dir

colors = sns.color_palette('colorblind')
err_fn = lambda x: (x.min(), x.max())


def ortho(lie='se_n'):
    path = os.path.join(_parent_dir, 'data', f'ortho_{lie}.csv')
    save_path = os.path.join(_parent_dir, 'figures', f'ortho_{lie}.png')
    df = pd.read_csv(path)
    
    
    fig, axes = plt.subplots(1,3, figsize=(20,5))

    axes = axes.flatten()
    linesyles = ['solid','-.', '--', ':']
    LEGEND=20
    TICKS = 20

    linewidth = 3
    use_log = False

    for idx_r, r in enumerate(range(1,4)):
        d = {d: df[np.logical_and(df.d == d, df.r == r)] for d in range(2,5)}
        for idx_d, (key, val) in enumerate(d.items()):
            linestyle = linesyles[idx_r]
            sns.lineplot(data=val, x='m', y = 'n_recover',  
                        errorbar=err_fn, 
                        estimator="median", 
                        color = colors[idx_r],
                        label = f'r={r}',
                        linestyle=linestyle,
                        ax=axes[idx_d],
                        linewidth=linewidth
                        )
            ax = axes[idx_d]
            ax.set_title(f'deg = {key}', fontsize=TICKS)
            
            
            ax.scatter(val.m, val.n_lib, c='k')
            ax.legend(loc=2)
            
            if use_log:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylim(1,1400)
            else:
                ax.set_ylim(0,None)
                
            ax.set_ylabel(None)
            ax.set_xlabel(None)
            ax.legend(fontsize=LEGEND)
            ax.tick_params(labelsize=TICKS)
            ax.set_xlabel('dim(x)', fontsize=TICKS)


    fig.tight_layout()

    plt.savefig(save_path, 
                transparent=True, bbox_inches='tight')
    plt.close()
    
    return df, fig, axes


def quad():
    lie = 'se_n'
    path = os.path.join(_parent_dir, 'data', f'quad_{lie}.csv')
    save_path = os.path.join(_parent_dir, 'figures', f'quad_{lie}.png')
    df = pd.read_csv(path)
    
    
    fig, axes = plt.subplots(1,2, figsize=(13,5))

    axes = axes.flatten()
    linesyles = ['solid','-.', '--', ':']
    LEGEND=20
    TICKS = 20

    linewidth = 3
    use_log = False
    for r in range(1,3):
        d = {deg: df[np.logical_and(df.d == deg, df.r == r)] for deg in range(1,3)}
        for idx_d, (key, val) in enumerate(d.items()):
            
            # fix the legend to reflect what data we have.
            if r > idx_d +1:
                continue

            linestyle = linesyles[r-1]
            sns.lineplot(data=val, x='m', y = 'n_recover',  
                        errorbar=err_fn, 
                        estimator="median", 
                        color = colors[r-1],
                        label = f'r={r}',
                        linestyle=linestyle,
                        ax=axes[idx_d],
                        linewidth=linewidth
                        )
            ax = axes[idx_d]
            ax.set_title(f'deg = {key}', fontsize=TICKS)
                

            ax.scatter(val.m, val.n_lib, c='k')
            ax.legend(loc=2)

            if use_log:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylim(1,1400)
            else:
                ax.set_ylim(0,None)
                
            ax.set_ylabel(None)
            ax.set_xlabel(None)
            ax.legend(fontsize=LEGEND)
            ax.tick_params(labelsize=TICKS)
            ax.set_xlabel('dim(x)', fontsize=TICKS)

    fig.tight_layout()

    plt.savefig(save_path, 
                transparent=True, bbox_inches='tight')
    plt.close()
    
    return df, fig, axes
        
if __name__ == '__main__': 
    df, fig, axes  = ortho('se_n')
    df, fig, axes  = ortho('t_n')
    df, fig, axes = quad()