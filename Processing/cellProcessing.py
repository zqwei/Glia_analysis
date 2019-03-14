import numpy as np
import pandas as pd
import os, sys
from glob import glob
from h5py import File
import warnings
warnings.filterwarnings('ignore')
import dask.array as da
import shutil
from utils import *
import fish_proc.utils.dask_ as fdask
from fish_proc.utils.getCameraInfo import getCameraInfo
import time
cameraNoiseMat = '/groups/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'


def refresh_workers(cluster, numCores=20):
    try:
        cluster.stop_all_jobs()
        time.sleep(10)
    except:
        pass
    cluster.start_workers(numCores)
    return None

def print_client_links(cluster):
    print(f'Scheduler: {cluster.scheduler_address}')
    print(f'Dashboard link: {cluster.dashboard_link}')
    return None


def preprocessing(dir_root, save_root, numCores=20, window=100, percentile=20, nsplit = 4):
    '''
      1. pixel denoise
      2. registration -- save registration file
      3. detrend using percentile baseline
      4. local pca denoise
    '''

    # set worker
    cluster, client = fdask.setup_workers(numCores)
    print_client_links(cluster)

    files = sorted(glob(dir_root+'/*.h5'))
    chunks = File(files[0],'r')['default'].shape
    data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])

    # pixel denoise
    cameraInfo = getCameraInfo(dir_root)
    denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraInfo=cameraInfo))

    # save and compute reference image
    print('Compute reference image ---')
    if not os.path.exists(f'{save_root}/motion_fix_.h5'):
        med_win = len(denoised_data)
        ref_img = denoised_data[med_win-50:med_win+50].mean(axis=0).compute()
        save_h5(f'{save_root}/motion_fix_.h5', ref_img, dtype='float16')
        
    print('--- Done computing reference image')

    # compute affine transform
    print('Registration to reference image ---')
    if not os.path.exists(f'{save_root}/trans_affs.npy'):
        ref_img = File(f'{save_root}/motion_fix_.h5', 'r')['default'].value
        ref_img = ref_img.max(axis=0, keepdims=True)
        trans_affine = denoised_data.map_blocks(lambda x: estimate_rigid2d(x, fixed=ref_img), dtype='float32', drop_axis=(3), chunks=(1,4,4)).compute()
        np.save(f'{save_root}/trans_affs.npy', trans_affine)
        trans_affine_ = da.from_array(trans_affine, chunks=(1,4,4))
    else:
        trans_affine_ = np.load(f'{save_root}/trans_affs.npy')
        trans_affine_ = da.from_array(trans_affine_, chunks=(1,4,4))
    print('--- Done registration reference image')

    # apply affine transform
    print('Apply registration ---')
    if not os.path.exists(f'{save_root}/motion_corrected_data.zarr'):
        if numCores<700:
            numCores_ = 700
        else:
            numCores_ = numCores
        refresh_workers(cluster, numCores=numCores_)
        trans_data_ = da.map_blocks(apply_transform3d, denoised_data, trans_affine_, chunks=(1, *denoised_data.shape[1:]), dtype='float32')
        trans_data_t = trans_data_.transpose((1, 2, 3, 0)).rechunk((1, chunks[1]//nsplit, chunks[2]//nsplit, -1))
        trans_data_t.to_zarr(f'{save_root}/motion_corrected_data.zarr')

    # detrend
    if not os.path.exists(f'{save_root}/detrend_data.zarr'):
        refresh_workers(cluster, numCores=trans_data_t.shape[0]*nsplit*nsplit+1)
        trans_data_t = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
        Y_d = trans_data_t.map_blocks(lambda v: v - baseline(v, window=window, percentile=percentile), dtype='float32')
        Y_d.to_zarr(f'{save_root}/detrend_data.zarr')

    # remove meaning before svd (-- pca)
    if not os.path.exists(f'{save_root}/Y_ave.zarr'):
        refresh_workers(cluster, numCores=60)
        Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
        Y_d_ave = Y_d.mean(axis=-1, keepdims=True, dtype='float32')
        print('Save average data ---')
        Y_d_ave.to_zarr(f'{save_root}/Y_ave.zarr')
        
    if not os.path.exists(f'{save_root}/Y_max.zarr'):
        refresh_workers(cluster, numCores=60)
        Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
        Y_d_max = Y_d.max(axis=-1, keepdims=True)
        print('Save average data ---')
        Y_d_max.to_zarr(f'{save_root}/Y_max.zarr')  
    cluster.stop_all_jobs()
    time.sleep(10)
    return None


def local_pca_on_mask(save_root, numCores=20):
    from dask.distributed import Client, LocalCluster
    if not os.path.exists(f'{save_root}/denoise_rlt'):
        os.makedirs(f'{save_root}/denoise_rlt')
    cluster, client = fdask.setup_workers(numCores)
    print_client_links(cluster)
    Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
    mask = da.from_zarr(f'{save_root}/mask_map.zarr')
    Y_svd = da.map_blocks(fb_pca_block, Y_d, mask, dtype='float32', save_folder=save_root)
    Y_svd.to_zarr(f'{save_root}/masked_local_pca_data.zarr')
    return None


def demix_cells(save_root, dt, is_skip=True, numCores = 200):
    '''
      1. local pca denoise
      2. cell segmentation
    '''
    cluster, client = fdask.setup_workers(numCores)
    print_client_links(cluster)

    Y_svd = da.from_zarr(f'{save_root}/masked_local_pca_data.zarr')
    Y_svd = Y_svd[:, :, :, ::dt]
    mask = da.from_zarr(f'{save_root}/mask_map.zarr')
    if not os.path.exists(f'{save_root}/demix_rlt/'):
        os.mkdir(f'{save_root}/demix_rlt/')
    da.map_blocks(demix_blocks, Y_svd, mask, chunks=(1, 1, 1, 1), dtype='int8', save_folder=save_root, is_skip=is_skip).compute()
    cluster.stop_all_jobs()
    return None


def check_fail_block(save_root, dt=0):
    file = glob(f'{save_root}/masked_local_pca_data.zarr/*.partial')
    print(file)


def check_demix_cells(save_root, block_id, plot_global=True, plot_mask=True):
    import matplotlib.pyplot as plt
    import pickle
    Y_d_ave = da.from_zarr(f'{save_root}/Y_max.zarr')
    mask = da.from_zarr(f'{save_root}/mask_map.zarr')
    Y_d_ave_ = Y_d_ave.blocks[block_id].squeeze().compute(scheduler='threads')
    mask_ = mask.blocks[block_id].squeeze().compute(scheduler='threads')
    v_max = np.percentile(Y_d_ave_, 90)
    _, xdim, ydim, _ = Y_d_ave.shape
    _, x_, y_, _ = Y_d_ave.chunksize
    try:
        A_ = load_A_matrix(save_root=save_root, block_id=block_id, min_size=0)
        A_comp = np.zeros(A_.shape[0])
        A_comp[A_.sum(axis=-1)>0] = np.argmax(A_[A_.sum(axis=-1)>0, :], axis=-1) + 1
        # plt.imshow(Y_d_ave_, cmap=plt.cm.gray)
        plt.imshow(A_comp.reshape(y_, x_).T, cmap=plt.cm.nipy_spectral_r)
        if plot_mask:
            plt.imshow(mask_, cmap='gray', alpha=0.5)
        plt.title('Components')
        plt.axis('off')
        plt.show()
    except:
        print('No components')
    
    plt.imshow(Y_d_ave_, vmax=v_max)
    plt.title('Max Intensity')
    plt.axis('off')
    plt.show()
    
    if plot_global:
        area_mask = np.zeros((xdim, ydim)).astype('bool')
        area_mask[block_id[1]*x_:block_id[1]*x_+x_, block_id[2]*y_:block_id[2]*y_+y_]=True
        plt.figure(figsize=(16, 16))
        plt.imshow(Y_d_ave[block_id[0]].squeeze().compute(scheduler='threads'), vmax=v_max)
        plt.imshow(area_mask, cmap='gray', alpha=0.3)
        plt.axis('off')
        plt.show()
    return None


def compute_cell_dff_pixels(dir_root, save_root, numCores=20, window=100, percentile=20):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''
    # set worker
    cluster, client = fdask.setup_workers(numCores)
    print_client_links(cluster)
    files = sorted(glob(dir_root+'/*.h5'))
    chunks = File(files[0],'r')['default'].shape
    data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])
    # pixel denoise
    cameraInfo = getCameraInfo(dir_root)
    denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraInfo=cameraInfo))
    trans_affine_ = np.load(f'{save_root}/trans_affs.npy')
    trans_affine_ = da.from_array(trans_affine_, chunks=(1,4,4))
    # apply affine transform
    trans_data_ = da.map_blocks(apply_transform3d, denoised_data, trans_affine_, chunks=(1, *denoised_data.shape[1:]), dtype='float32')
    # baseline
    chunk_x, chunk_y = chunks[-2:]
    trans_data_t = trans_data_.transpose((1, 2, 3, 0)).rechunk((1, 1, 1, -1))
    baseline_t = trans_data_t.map_blocks(lambda v: baseline(v, window=window, percentile=percentile), dtype='float32')
    min_t = trans_data_t.map_blocks(lambda v: np.min(np.percentile(v, 0.3), 0), dtype='float32')
    dff = (trans_data_t-baseline_t)/(baseline_t-min_t)
    dff.to_zarr(f'{save_root}/pixel_dff.zarr', overwrite=True)
    cluster.stop_all_jobs()
    cluster.close()
    return None


def compute_cell_dff_raw(dir_root, save_root, numCores=20, window=100, percentile=20, nsplit=8):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''
    # set worker
    cluster, client = fdask.setup_workers(numCores)
    files = sorted(glob(dir_root+'/*.h5'))
    chunks = File(files[0],'r')['default'].shape
    data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])
    # pixel denoise
    cameraInfo = getCameraInfo(dir_root)
    denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraInfo=cameraInfo))
    trans_affine_ = np.load(f'{save_root}/trans_affs.npy')
    trans_affine_ = da.from_array(trans_affine_, chunks=(1,4,4))
    # apply affine transform
    trans_data_ = da.map_blocks(apply_transform3d, denoised_data, trans_affine_, chunks=(1, *denoised_data.shape[1:]), dtype='float32')
    # baseline
    chunk_x, chunk_y = chunks[-2:]
    trans_data_t = trans_data_.transpose((1, 2, 3, 0)).rechunk((1, chunk_x//nsplit, chunk_y//nsplit, -1))
    if not os.path.exists(f'{save_root}/cell_raw_dff'):
        os.makedirs(f'{save_root}/cell_raw_dff')
    trans_data_t.map_blocks(compute_cell_raw_dff, dtype='float32', chunks=(1, 1, 1, 1), save_root=save_root, window=window, percentile=percentile).compute()
    return None


def compute_cell_dff_NMF(dir_root, save_root, numCores=20, window=100, percentile=20, nsplit=8, dt=5):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''
    # set worker
    cluster, client = fdask.setup_workers(numCores)
    files = sorted(glob(dir_root+'/*.h5'))
    chunks = File(files[0],'r')['default'].shape
    data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])
    # pixel denoise
    cameraInfo = getCameraInfo(dir_root)
    denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraInfo=cameraInfo))
    trans_affine_ = np.load(f'{save_root}/trans_affs.npy')
    trans_affine_ = da.from_array(trans_affine_, chunks=(1,4,4))
    # apply affine transform
    trans_data_ = da.map_blocks(apply_transform3d, denoised_data, trans_affine_, chunks=(1, *denoised_data.shape[1:]), dtype='float32')
    # baseline
    chunk_x, chunk_y = chunks[-2:]
    trans_data_t = trans_data_.transpose((1, 2, 3, 0)).rechunk((1, chunk_x//nsplit, chunk_y//nsplit, -1))
    pca_data = da.from_zarr(f'{save_root}/masked_local_pca_data.zarr').rechunk((1, chunk_x//nsplit, chunk_y//nsplit, -1))
    if not os.path.exists(f'{save_root}/cell_nmf_dff'):
        os.makedirs(f'{save_root}/cell_nmf_dff')
    da.map_blocks(compute_cell_denoise_dff, trans_data_t, pca_data, dtype='float32', chunks=(1, 1, 1, 1), save_root=save_root, dt=dt, window=window, percentile=percentile).compute()
    return None

