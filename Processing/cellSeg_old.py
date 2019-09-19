#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import os, sys
import warnings
warnings.filterwarnings('ignore')
from cellProcessing_single_WS import *
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt

# dir_root = '/nrs/ahrens/jing/giving_up/20190430/fish01/6dpf_HuC-GCaMP7ff-GFAP-RGECO_GU-slow-fwd_fish01_exp01_20190430_174349/im'
# save_root = '/nrs/ahrens/Ziqiang/Jing_Glia_project/Processed_data/20190430/fish01/6dpf_HuC-GCaMP7ff-GFAP-RGECO_GU-slow-fwd_fish01_exp01_20190430_174349/'

dir_root = '/nrs/ahrens/jing/giving_up/20190426/fish03/7dpf_HuC-H2B_GCaMP7ff_GU-slow-fwd_fish03_exp02_20190426_221213/im'
save_root = '/nrs/ahrens/Ziqiang/Jing_Glia_project/Processed_data/20190426/fish03/7dpf_HuC-H2B_GCaMP7ff_GU-slow-fwd_fish03_exp02_20190426_221213/'

dask_tmp = '/opt/data/weiz/dask-worker-space'
memory_limit = 0 # unlimited

if not os.path.exists(save_root):
    os.makedirs(save_root)

nsplit = (16, 32)
baseline_percentile = 20
baseline_window = 1000   # number of frames
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'

if not os.path.exists(save_root):
    os.makedirs(save_root)


print('========================')
print('Preprocessing')
if not os.path.exists(f'{save_root}/motion_corrected_data.zarr'):
    preprocessing(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, nsplit=nsplit, dask_tmp=dask_tmp, memory_limit=memory_limit, is_bz2=False)

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
print('Demix')
dt = 3
is_skip = False
demix_cells(save_root, dt, is_skip=is_skip, dask_tmp=dask_tmp, memory_limit=memory_limit)
