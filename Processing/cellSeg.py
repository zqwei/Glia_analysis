#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import os, sys
import warnings
warnings.filterwarnings('ignore')
from fish_proc.wholeBrainDask.cellProcessing_single_WS import *
import dask.array as da
import numpy as np
import pandas as pd

df = pd.read_csv('data_list.csv')
dask_tmp = '/nrs/ahrens/Ziqiang/dask-worker-space'
memory_limit = 0 # unlimited
down_sample_registration = 3
baseline_percentile = 20
baseline_window = 1000   # number of frames
num_t_chunks = 30
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'

for ind, row in df.iterrows():
    dir_root = row['dat_dir'] # +'im/'
    save_root = row['save_dir']
    if os.path.exists(f'{save_root}/cell_raw_dff_sparse.npz'):
        continue
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    files = sorted(glob(dir_root+'/*.h5'))
    chunks = File(files[0],'r')['default'].shape
    nsplit = (chunks[1]//64, chunks[2]//64)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print('========================')
    print('Preprocessing')
    if not os.path.exists(save_root+'/motion_corrected_data_chunks_%03d.zarr'%(num_t_chunks-1)):
        preprocessing(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, nsplit=nsplit, \
                      num_t_chunks=num_t_chunks, dask_tmp=dask_tmp, memory_limit=memory_limit, \
                      is_bz2=False, down_sample_registration=down_sample_registration)
    print('========================')
    print('Combining motion corrected data')
    if not os.path.exists(f'{save_root}/motion_corrected_data.zarr'):
        combine_preprocessing(dir_root, save_root, num_t_chunks=num_t_chunks, dask_tmp=dask_tmp, memory_limit=memory_limit)
    if not os.path.exists(f'{save_root}/detrend_data.zarr'):
        detrend_data(dir_root, save_root, window=baseline_window, percentile=baseline_percentile, \
                     nsplit=nsplit, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print('========================')
    print('Mask')
    default_mask(dir_root, save_root, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print('========================')
    print('Demix')
    dt = 3
    is_skip = True
    demix_cells(save_root, dt, is_skip=is_skip, dask_tmp=dask_tmp, memory_limit=memory_limit)