#!/groups/ahrens/home/weiz/anaconda3/envs/myenv/bin/python

import os, sys
import warnings
warnings.filterwarnings('ignore')
from cellProcessing_single_WS import *

dir_root = '/nrs/ahrens/jing/giving_up/20190426/fish03/7dpf_HuC-H2B_GCaMP7ff_GU-slow-fwd_fish03_exp02_20190426_221213/im'
save_root = '/nrs/ahrens/jing/giving_up/20190426/fish03/7dpf_HuC-H2B_GCaMP7ff_GU-slow-fwd_fish03_exp02_20190426_221213/weiz_processed/'
dask_tmp = '/nrs/ahrens/jing/giving_up/20190426/fish03/7dpf_HuC-H2B_GCaMP7ff_GU-slow-fwd_fish03_exp02_20190426_221213/weiz_processed/dask-worker-space'
memory_limit = 0 # unlimited

if not os.path.exists(save_root):
    os.makedirs(save_root)

nsplit = (10, 16)
baseline_percentile = 20  
baseline_window = 400   # number of frames
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'

preprocessing(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, window=baseline_window, 
              percentile=baseline_percentile, nsplit=nsplit, dask_tmp=dask_tmp, memory_limit=memory_limit)


local_pca_on_mask(save_root, is_dff=False, dask_tmp=dask_tmp, memory_limit=memory_limit)