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

nsplit = (10, 16)
baseline_percentile = 20  
baseline_window = 400   # number of frames
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'

if not os.path.exists(f'{save_root}/detrend_data.zarr'):
    preprocessing(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, window=baseline_window, 
                  percentile=baseline_percentile, nsplit=nsplit, dask_tmp=dask_tmp, memory_limit=memory_limit, is_bz2=True)

# if not os.path.exists(f'{save_root}/masked_local_pca_data.zarr'):
#     local_pca_on_mask(save_root, is_dff=False, dask_tmp=dask_tmp, memory_limit=memory_limit)

# dt = 3
# is_skip = False

# demix_cells(save_root, dt, is_skip=is_skip, dask_tmp=dask_tmp, memory_limit=memory_limit)