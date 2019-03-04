import numpy as np
import pandas as pd
import os, sys
from glob import glob
from h5py import File
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import dask.array as da
import shutil
from utils import *
cameraNoiseMat = '/groups/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'
dir_root = '/nrs/ahrens/Yu/SPIM/active_dataset/glia_neuron_imaging/20161109/fish2/20161109_2_1_6dpf_GFAP_GC_Huc_RG_GA_CL_fb_OL_f0_0GAIN_20161109_211950/raw'


def refresh_workers(cluster, numCores=20):
    import time
    try:
        cluster.stop_all_jobs()
        time.sleep(10)
    except:
        pass
    cluster.start_workers(numCores)
    return None


def preprocessing(dir_root, save_root, numCores=20, window=100, percentile=20):
    '''
      1. pixel denoise
      2. registration -- save registration file
      3. detrend using percentile baseline
      4. local pca demix
    '''
    import fish_proc.utils.dask_ as fdask
    from fish_proc.utils.getCameraInfo import getCameraInfo
    import dask.array as da    
    
    # set worker
    cluster, client = fdask.setup_workers(numCores)
    
    files = sorted(glob(dir_root+'/*.h5'))
    chunks = File(files[0],'r')['default'].shape
    data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])
    
    # pixel denoise
    cameraInfo = getCameraInfo(dir_root)
    denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraInfo=cameraInfo))
    
    # save and compute reference image
    if not os.path.exists(f'{save_root}/motion_fix_.h5'):
        med_win = len(denoised_data)
        ref_img = denoised_data[med_win-10:med_win+10].mean(axis=0).compute()
        save_h5(f'{save_root}/motion_fix_.h5', ref_img, dtype='float16')
        refresh_workers(cluster, numCores=numCores)
    else:
        ref_img = File('motion_fix_.h5', 'r')['default'].value
    ref_img = ref_img.max(axis=0, keepdims=True)
    
    # compute affine transform
    if not os.path.exists(f'{save_root}/trans_affs.npy'):
        trans_affine = denoised_data.map_blocks(lambda x: estimate_rigid2d(x, fixed=ref_img), dtype='float32', drop_axis=(3), chunks=(1,4,4)).compute()
        refresh_workers(cluster, numCores=numCores)
        np.save(f'{save_root}/trans_affs.npy', trans_affine)
        trans_affine_ = da.from_array(trans_affine, chunks=(1,4,4))
    else:
        trans_affine_ = np.load(f'{save_root}/trans_affs.npy')
        trans_affine_ = da.from_array(trans_affine_, chunks=(1,4,4))
    
    # apply affine transform
    trans_data_ = da.map_blocks(apply_transform3d, denoised_data, trans_affine_, chunks=(1, *denoised_data.shape[1:]), dtype='float32')
    
    # compute detrend data
    chunk_x, chunk_y = chunks[-2:]
    trans_data_t = trans_data_.transpose((1, 2, 3, 0)).rechunk((1, chunk_x//4, chunk_y//4, -1))
    Y_d = trans_data_t.map_blocks(lambda v: v - baseline(v, window=window, percentile=percentile), dtype='float32')
    
    # remove meaning before svd (-- pca)
    Y_d_ave = Y_d.mean(axis=-1, keepdims=True, dtype='float32')
    Y_d = Y_d - Y_d_ave
    
    # local pca on overlap blocks 
    xy_lap = 4 # overlap by 10 pixel in blocks
    g = da.overlap.overlap(Y_d, depth={1: xy_lap, 2: xy_lap}, boundary={1: 0, 2: 0})
    Y_svd = g.map_blocks(local_pca, dtype='float32')
    Y_svd = da.overlap.trim_internal(Y_svd, {1: xy_lap, 2: xy_lap})
    
    if os.path.exists(f'{save_root}/local_pca_data.zarr'):
        shutil.rmtree(f'{save_root}/local_pca_data.zarr')
    Y_svd.to_zarr(f'{save_root}/local_pca_data.zarr')
    cluster.stop_all_jobs()
    return None


