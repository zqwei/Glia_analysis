#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import os, sys
import warnings
warnings.filterwarnings('ignore')
from cellProcessing_single_WS import *

dir_root = '/nrs/ahrens/Yu/SPIM/active_dataset/20141209_1_1_cy74_GAD_1_1_simpleGU_20141209_154521/raw'
save_root = '/nrs/ahrens/Ziqiang/Pipeline_test/Yu_TK_pipeline/'
dask_tmp = '/groups/ahrens/home/weiz/thinclient_drives/dask-worker-space'
memory_limit = 0 # unlimited

if not os.path.exists(save_root):
    os.makedirs(save_root)

nsplit = (4, 8)
baseline_percentile = 20
baseline_window = 400   # number of frames
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'

print('========================')
print('Preprocessing')
if not os.path.exists(f'{save_root}/detrend_data.zarr'):
    preprocessing(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, window=baseline_window,
                  percentile=baseline_percentile, nsplit=nsplit, dask_tmp=dask_tmp, memory_limit=memory_limit, is_bz2=True)

print('========================')
print('Mask')
# Y = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
# Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
# Y_b = Y - Y_d
# Y_b_max_mask = Y_b.max(axis=-1, keepdims=True)>2
# Y_b_min_mask = Y_b.min(axis=-1, keepdims=True)>1
# mask = Y_b_max_mask & Y_b_min_mask
# mask.to_zarr(f'{save_root}/mask_map.zarr', overwrite=True)


# Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
# Y_d_max = Y_d.max(axis=-1, keepdims=True)
# print('Save average data ---')
# Y_d_max.to_zarr(f'{save_root}/Y_max.zarr', overwrite=True)


# Y_d = zarr.open(f'{save_root}/Y_max.zarr', 'r')
# mask = zarr.open(f'{save_root}/mask_map.zarr', 'r')
# for n, n_ave_ in enumerate(Y_d):
#     _ = n_ave_.squeeze().copy()
#     _[~mask[n].squeeze()] = 0
#     plt.imshow(_, vmax=np.percentile(_, 99))
#     plt.title(n)
#     plt.show()

print('========================')
print('Denoise')
if not os.path.exists(f'{save_root}/masked_local_pca_data.zarr'):
    local_pca_on_mask(save_root, is_dff=False, dask_tmp=dask_tmp, memory_limit=memory_limit)

print('========================')
print('Demix')
dt = 3
is_skip = False

demix_cells(save_root, dt, is_skip=is_skip, dask_tmp=dask_tmp, memory_limit=memory_limit)
