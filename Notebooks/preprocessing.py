import numpy as np
import pandas as pd
import os, sys
from glob import glob
from h5py import File
import matplotlib.pyplot as plt
from fish_proc.utils.memory import get_process_memory, clear_variables
cameraNoiseMat = '/groups/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'
dir_root = '/nrs/ahrens/Yu/SPIM/active_dataset/glia_neuron_imaging/20161109/fish2/20161109_2_1_6dpf_GFAP_GC_Huc_RG_GA_CL_fb_OL_f0_0GAIN_20161109_211950/raw'


from dask.distributed import Client
client = Client("tcp://127.0.0.1:39978")
client

import dask.array as da
chunks = img.shape
data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])


from fish_proc.utils.getCameraInfo import getCameraInfo
cameraInfo = getCameraInfo(dir_root)


denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraInfo=cameraInfo))


med_win = len(denoised_data)
ref_img = denoised_data[med_win-10:med_win+10].mean(axis=0).compute(scheduler='threads')
np.save('motion_fix_', ref_img)

trans_affine = denoised_data[:4].map_blocks(lambda x: estimate_rigid3d_affine(x, fixed=ref_img), dtype='object', drop_axis=(1,2,3), chunks=(1,)).compute(scheduler='threads')