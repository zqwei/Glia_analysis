#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import fish_proc.wholeBrainDask.cellProcessing_single_WS as fwc
import dask.array as da
import pandas as pd
df = pd.read_csv('../Processing/data_list.csv')
dask_tmp = '/scratch/weiz/dask-worker-space'
memory_limit = 0 # unlimited


for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    print(save_root)
    if not os.path.exists(save_root+'Y_ave.npy'):
        continue
    brain_map = np.load(save_root+'Y_ave.npy')
    num_plane = brain_map.shape[0]
    if os.path.exists(save_root+'brain_map.png'):
        fig, ax = plt.subplots(4, num_plane//4+1, figsize=(40, 10))
        ax = ax.flatten()
        for n, _ in enumerate(brain_map):
            ax[n].imshow(_.squeeze().astype('float'), vmax=np.percentile(_[:].astype('float'), 95))
            ax[n].axis('off')
        plt.savefig(save_root+'brain_map.png')
        plt.close()
    
    ## if recompute dFF
    # _ = np.load(save_root+'cell_raw_dff_sparse.npz', allow_pickle=True)
    # A = _['A'].astype('float')
    # F_ = _['F'].astype('float')
    # A_loc = _['A_loc']
    # _ = None

    # # first remove components with low signals
    # valid_cell = F_.max(axis=-1)>20

    # A = A[valid_cell]
    # A_loc = A_loc[valid_cell]
    # F_ = F_[valid_cell]

    # F_dask = da.from_array(F_, chunks=('auto', -1))
    # baseline_ = da.map_blocks(fwc.baseline, F_dask, dtype='float', window=400, percentile=20, downsample=10).compute()
    # dFF = F_/baseline_-1

    # invalid_ = (dFF.max(axis=-1)>5) | (np.isnan(dFF.max(axis=-1))) | (baseline_.min(axis=-1)<=0)

    # np.savez(save_root+'cell_dff.npz', A=A[~invalid_].astype('float16'), A_loc=A_loc[~invalid_], dFF=dFF[~invalid_].astype('float16'))
    
    _ = np.load(save_root+'cell_dff.npz')
    A=_['A'].astype('float')
    A_loc=_['A_loc']
    dFF=_['dFF'].astype('float')
    
    A_ext = np.zeros(brain_map.shape[:-1]).astype('int')
    for n_, A_ in enumerate(A):
        A_loc_ = A_loc[n_]
        z, x, y = A_loc_
        _ = (A_>A_.max()*0.4).astype('int')*(n_+1)
        cx, cy = A_ext[z, x:x+100, y:y+100].shape
        A_ext[z, x:x+100, y:y+100]=np.maximum(A_ext[z, x:x+100, y:y+100], _[:cx, :cy])
    
    if os.path.exists(save_root+'components.png'):
        fig, ax = plt.subplots(4, A_ext.shape[0]//4+1, figsize=(40, 10))
        ax = ax.flatten()
        for n, _ in enumerate(A_ext):
            _[_>0] = _[_>0]%32+1
            ax[n].imshow(_, cmap=plt.cm.nipy_spectral)
            ax[n].axis('off')
        plt.savefig(save_root+'components.png')
        plt.close()
    
