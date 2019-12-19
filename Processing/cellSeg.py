#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import os, sys
import warnings
warnings.filterwarnings('ignore')
from fish_proc.wholeBrainDask.cellProcessing_single_WS import *
import fish_proc.wholeBrainDask.cellProcessing_single_WS as fwc
from fish_proc.utils.fileio import make_tarfile
import dask.array as da
import numpy as np
import pandas as pd
import shutil

df = pd.read_csv('data_list.csv')
dask_tmp = '/scratch/weiz/dask-worker-space'
memory_limit = 0 # unlimited
down_sample_registration = 3
baseline_percentile = 20
baseline_window = 1000   # number of frames
num_t_chunks = 25
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'
savetmp = '/scratch/weiz/'

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
    if not os.path.exists(savetmp+'/motion_corrected_data_chunks_%03d.zarr'%(num_t_chunks-1)):
        preprocessing(dir_root, savetmp, cameraNoiseMat=cameraNoiseMat, nsplit=nsplit, \
                      num_t_chunks=num_t_chunks, dask_tmp=dask_tmp, memory_limit=memory_limit, \
                      is_bz2=False, down_sample_registration=down_sample_registration)
    print('========================')
    print('Combining motion corrected data')
    if not os.path.exists(f'{savetmp}/motion_corrected_data.zarr'):
        combine_preprocessing(dir_root, savetmp, num_t_chunks=num_t_chunks, dask_tmp=dask_tmp, memory_limit=memory_limit)
    if not os.path.exists(f'{savetmp}/detrend_data.zarr'):
        detrend_data(dir_root, savetmp, window=baseline_window, percentile=baseline_percentile, \
                     nsplit=nsplit, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print('========================')
    print('Mask')
    default_mask(dir_root, savetmp, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print('========================')
    print('Demix')
    dt = 3
    is_skip = True
    demix_cells(savetmp, dt, is_skip=is_skip, dask_tmp=dask_tmp, memory_limit=memory_limit)

    # remove some files --
    Y_d = da.from_zarr(f'{savetmp}/Y_max.zarr')
    np.save(f'{save_root}/Y_max', Y_d.compute())
    Y_d = da.from_zarr(f'{savetmp}/Y_d_max.zarr')
    np.save(f'{save_root}/Y_d_max', Y_d.compute())
    Y_d = da.from_zarr(f'{savetmp}/Y_ave.zarr')
    chunks = Y_d.chunksize[:-1]
    np.save(f'{save_root}/Y_ave', Y_d.compute())
    np.save(f'{save_root}/chunks', chunks)

    for nfolder in glob(savetmp+'/Y_*.zarr/'):
        shutil.rmtree(nfolder)
    shutil.rmtree(f'{savetmp}/detrend_data.zarr')
    make_tarfile(save_root+'sup_demix_rlt.tar.gz', savetmp+'sup_demix_rlt')

    Y_d = np.load(f'{save_root}/Y_ave.npy')
    chunks = np.load(f'{save_root}/chunks.npy')
    Y_d_max = Y_d.max(axis=0, keepdims=True)

    max_ = np.percentile(Y_d_max, 40)
    mask_ = Y_d_max>max_
    mask_ = np.repeat(mask_, Y_d.shape[0], axis=0)
    mask_ = da.from_array(mask_, chunks=(1, chunks[1], chunks[2], -1))

    print('========================')
    print('DF/F computation')
    compute_cell_dff_raw(savetmp, mask_, dask_tmp=dask_tmp, memory_limit=0)
    combine_dff(savetmp)
    combine_dff_sparse(savetmp)

    shutil.move(f'{savetmp}/cell_raw_dff_sparse.npz', f'{save_root}/cell_raw_dff_sparse.npz')
    shutil.move(f'{savetmp}/cell_raw_dff.npz', f'{save_root}/cell_raw_dff.npz')

    for nfolder in glob(savetmp+'*.zarr/'):
        shutil.rmtree(nfolder)
    shutil.rmtree(savetmp+'cell_raw_dff')
    shutil.rmtree(savetmp+'sup_demix_rlt')

    brain_map = np.load(save_root+'Y_ave.npy')
    _ = np.load(save_root+'cell_raw_dff_sparse.npz', allow_pickle=True)
    A = _['A'].astype('float')
    F_ = _['F'].astype('float')
    A_loc = _['A_loc']
    _ = None

    valid_cell = F_.max(axis=-1)>20

    A = A[valid_cell]
    A_loc = A_loc[valid_cell]
    F_ = F_[valid_cell]

    F_dask = da.from_array(F_, chunks=('auto', -1))
    baseline_ = da.map_blocks(fwc.baseline, F_dask, dtype='float', window=400, percentile=20, downsample=10).compute()
    dFF = F_/baseline_-1

    invalid_ = (dFF.max(axis=-1)>5) | (np.isnan(dFF.max(axis=-1))) | (baseline_.min(axis=-1)<=0)

    A_ext = np.zeros(brain_map.shape[:-1]).astype('int')
    for n_, A_ in enumerate(A):
        if invalid_[n_]:
            continue
        A_loc_ = A_loc[n_]
        z, x, y = A_loc_
        _ = (A_>A_.max()*0.4).astype('int')*(n_+1)
        cx, cy = A_ext[z, x:x+100, y:y+100].shape
        A_ext[z, x:x+100, y:y+100]=np.maximum(A_ext[z, x:x+100, y:y+100], _[:cx, :cy])

    np.savez(save_root+'cell_dff.npz', A=A[~invalid_].astype('float16'), A_loc=A_loc[~invalid_], dFF=dFF[~invalid_].astype('float16'))
