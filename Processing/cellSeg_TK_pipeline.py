#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import os, sys
import warnings
warnings.filterwarnings('ignore')
from cellProcessing_single_WS import *

dir_root = '/nrs/ahrens/Yu/SPIM/active_dataset/20141209_1_1_cy74_GAD_1_1_simpleGU_20141209_154521/raw'
save_root = '/nrs/ahrens/Ziqiang/Pipeline_test/Yu_TK_pipeline/'
dask_tmp = '/opt/data/weiz/dask-worker-space'
memory_limit = 0 # unlimited

if not os.path.exists(save_root):
    os.makedirs(save_root)

nsplit = (4, 8)
baseline_percentile = 20
baseline_window = 1000   # number of frames
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'

print('========================')
print('Preprocessing')
if not os.path.exists(f'{save_root}/motion_corrected_data.zarr'):
    preprocessing(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, nsplit=nsplit, dask_tmp=dask_tmp, memory_limit=memory_limit, is_bz2=True)

if not os.path.exists(f'{save_root}/detrend_data.zarr'):
    detrend_data(dir_root, save_root, window=baseline_window, percentile=baseline_percentile, nsplit=nsplit, dask_tmp=dask_tmp, memory_limit=memory_limit)

print('========================')
print('Mask')
default_mask(dir_root, save_root, dask_tmp=dask_tmp, memory_limit=memory_limit)

if False:
    print('========================')
    print('Plot masks')
    cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print_client_links(cluster)
    Y_d = zarr.open(f'{save_root}/Y_max.zarr', 'r')
    mask = zarr.open(f'{save_root}/mask_map.zarr', 'r')
    for n, n_ave_ in enumerate(Y_d):
        _ = n_ave_.squeeze().copy()
        _[~mask[n].squeeze()] = 0
        plt.imshow(_, vmax=np.percentile(_, 99))
        plt.title(n)
        plt.show()
    fdask.terminate_workers(cluster, client)

print('========================')
print('Denoise')
if not os.path.exists(f'{save_root}/masked_local_pca_data.zarr'):
    local_pca_on_mask(save_root, is_dff=False, dask_tmp=dask_tmp, memory_limit=memory_limit)

print('========================')
print('Demix')
dt = 3
is_skip = False

print('========================')
print('Demix')
dt = 3
is_skip = False
params = {'cut_perc': True, 
          'cut_off_point':[99, 85, 70, 40], 
          'length_cut':[30, 20, 20, 20], 
          'max_allow_neuron_size':0.15, 
          'patch_size':[10, 10],
          'max_iter':50,
          'max_iter_fin':90,
          'update_after':40}
demix_cells(save_root, dt, params=params, is_skip=is_skip, dask_tmp=dask_tmp, memory_limit=memory_limit)
