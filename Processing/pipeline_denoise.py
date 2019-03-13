import numpy as np
import os, sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from cellProcessing import *
import zarr
from fish_proc.denoiseLocalPCA.denoise import temporal as svd_patch
from utils import save_h5
from fish_proc.utils.memory import clear_variables

save_root = './processed'    
Y_d = zarr.open(f'{save_root}/detrend_data.zarr')
mask = zarr.open(f'{save_root}/mask_map.zarr')

z = Y_d.shape[0]

for nz in range(z):
    print(f'Processing layer {nz} ---------')
    Y_ = Y_d[nz]
    mask_ = mask[nz]
    Y_[mask_.squeeze()]=0
    Y_ = Y_ - Y_.mean(axis=-1, keepdims=True)
    dx=4
    nblocks=[64, 64]
    Y_svd, __ = svd_patch(Y_, nblocks=nblocks, dx=dx, stim_knots=None, stim_delta=0, is_single_core=False)
    save_h5(f'{save_root}/masked_local_pca_data_{nz}.h5', Y_svd, dtype='float32')
    Y_svd = None
    clear_variables(Y_)
    clear_variables(Y_svd)
    
    
    
