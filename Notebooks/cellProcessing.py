import numpy as np
import pandas as pd
import os, sys
from glob import glob
from h5py import File
import matplotlib.pyplot as plt
from fish_proc.utils.memory import get_process_memory, clear_variables
cameraNoiseMat = '/groups/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'
dir_root = '/nrs/ahrens/Yu/SPIM/active_dataset/glia_neuron_imaging/20161109/fish2/20161109_2_1_6dpf_GFAP_GC_Huc_RG_GA_CL_fb_OL_f0_0GAIN_20161109_211950/raw'


def preprocessing(dir_root, save_root, num_workers=20):
    '''
      1. pixel denoise
      2. registration
      3. save intermediate data of registrated images into small files (separated by z-stacks)
    '''
    import fish_proc.utils.dask_ as fdask
    from fish_proc.utils.getCameraInfo import getCameraInfo
    import dask.array as da
    from utilsPreprocessing import pixelDenoiseImag
    from utilsPreprocessing import estimate_rigid2d
    
    cluster, client = fdask.setup_workers(num_workers)
    
    files = sorted(glob(dir_root+'/*.h5'))
    len_files = len(files)
    files = files[len_files//2-1000:len_files//2+1000]
    chunks = File(files[0],'r')['default'].shape
    data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])
    
    cameraInfo = getCameraInfo(dir_root)
    denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraInfo=cameraInfo))
    
    if not os.path.exists(f'{save_root}/motion_fix_.npy'):
        med_win = len(denoised_data)
        ref_img = denoised_data[med_win-10:med_win+10].mean(axis=0).compute()
        np.save(f'{save_root}/motion_fix_', ref_img)
    else:
        ref_img = np.load(f'{save_root}/motion_fix_.npy')
    
    if not os.path.exists(f'{save_root}/trans_affs.npy'):
        trans_affine = denoised_data.map_blocks(lambda x: estimate_rigid2d(x, fixed=ref_img), dtype='float32', drop_axis=(3), chunks=(1,4,4)).compute()
        np.save(f'{save_root}/trans_affs.npy', trans_affine)
        trans_affine_ = da.from_array(trans_affine, chunks=(1,4,4))
    else:
        trans_affine_ = np.load('trans_affs.npy')
        trans_affine_ = da.from_array(trans_affine_, chunks=(1,4,4))
    
    trans_data_ = da.map_blocks(apply_transform3d, denoised_data[::100], trans_affine_, chunks=(1, *denoised_data.shape[1:]), dtype='float32')
    trans_data = trans_data_.compute()
    

# the rest will done in a lazy way..... 