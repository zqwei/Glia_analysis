#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import os, sys
import warnings
warnings.filterwarnings('ignore')
from cellProcessing_single_WS import *
import dask.array as da
import numpy as np

# dir_root = '/nrs/ahrens/jing/giving_up/20190430/fish01/6dpf_HuC-GCaMP7ff-GFAP-RGECO_GU-slow-fwd_fish01_exp01_20190430_174349/im'
# save_root = '/nrs/ahrens/Ziqiang/Jing_Glia_project/Processed_data/20190430/fish01/6dpf_HuC-GCaMP7ff-GFAP-RGECO_GU-slow-fwd_fish01_exp01_20190430_174349/'

# dir_root = '/nrs/ahrens/jing/giving_up/20190426/fish03/7dpf_HuC-H2B_GCaMP7ff_GU-slow-fwd_fish03_exp02_20190426_221213/im'
# save_root = '/nrs/ahrens/Ziqiang/Jing_Glia_project/Processed_data/20190426/fish03/7dpf_HuC-H2B_GCaMP7ff_GU-slow-fwd_fish03_exp02_20190426_221213/'

dir_root = '/nrs/ahrens/jing/giving_up/20190907/fish00/7dpf_HuC-GC7FF_GU-fwd_fish00_exp01_20190907_172639/im'
save_root = '/nrs/ahrens/Ziqiang/Jing_Glia_project/Processed_data/20190907/fish00/7dpf_HuC-GC7FF_GU-fwd_fish00_exp01_20190907_172639/'


dask_tmp = '/opt/data/weiz/dask-worker-space'
memory_limit = 0 # unlimited

if not os.path.exists(save_root):
    os.makedirs(save_root)

nsplit = (16, 32)
baseline_percentile = 20
baseline_window = 1000   # number of frames
num_t_chunks = 80
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'

if not os.path.exists(save_root):
    os.makedirs(save_root)


print('========================')
print('Preprocessing')
if not os.path.exists(f'{save_root}/motion_corrected_data.zarr'):
    preprocessing(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, nsplit=nsplit, num_t_chunks=num_t_chunks, dask_tmp=dask_tmp, memory_limit=memory_limit, is_bz2=False)

if not os.path.exists(f'{save_root}/detrend_data.zarr'):
    detrend_data(dir_root, save_root, window=baseline_window, percentile=baseline_percentile, nsplit=nsplit, dask_tmp=dask_tmp, memory_limit=memory_limit)


print('========================')
print('Mask')
default_mask(dir_root, save_root, dask_tmp=dask_tmp, memory_limit=memory_limit)


print('========================')
print('Demix')
dt = 3
is_skip = True
demix_cells(save_root, dt, is_skip=is_skip, dask_tmp=dask_tmp, memory_limit=memory_limit)


# Y_d = da.from_zarr(f'{save_root}/Y_ave.zarr')
# Y_d_max = Y_d.max(axis=0).compute()
# max_ = np.percentile(Y_d_max, 45)
# mask_ = Y_d_max>max_
# mask_ = da.from_array(mask_[np.newaxis,:], chunks=Y_d.chunksize)
# mask_ = da.repeat(mask_, Y_d.shape[0], axis=0).rechunk(Y_d.chunksize)

# print('========================')
# print('DF/F computation')
# compute_cell_dff_raw(save_root, mask_, dask_tmp=dask_tmp, memory_limit=0)
