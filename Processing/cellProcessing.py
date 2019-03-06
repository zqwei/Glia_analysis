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
cameraNoiseMat = '/groups/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'


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
      4. local pca denoise
    '''

    # set worker
    cluster, client = fdask.setup_workers(numCores)
    print(client)

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

    # compute affine transform
    if not os.path.exists(f'{save_root}/trans_affs.npy'):
        ref_img = File(f'{save_root}/motion_fix_.h5', 'r')['default'].value
        ref_img = ref_img.max(axis=0, keepdims=True)
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
    save_h5(f'{save_root}/Y_2dnorm_ave.h5', Y_d_ave.compute(), dtype='float32')

    # local pca on overlap blocks
    Y_d = Y_d - Y_d_ave
    xy_lap = 4 # overlap by 10 pixel in blocks
    g = da.overlap.overlap(Y_d, depth={1: xy_lap, 2: xy_lap}, boundary={1: 0, 2: 0})
    Y_svd = g.map_blocks(local_pca, dtype='float32')
    Y_svd = da.overlap.trim_internal(Y_svd, {1: xy_lap, 2: xy_lap})
    Y_svd.to_zarr(f'{save_root}/local_pca_data.zarr')
    cluster.stop_all_jobs()
    time.sleep(10)
    return None


def mask_brain(save_root, percentile=40, dt=5, numCores=20, is_skip_snr=True, save_masked_data=True):
    from fish_proc.utils.snr import correlation_pnr
    from fish_proc.utils.noise_estimator import get_noise_fft
    cluster, client = fdask.setup_workers(numCores)
    print(client)
    Y_d_ave_ = da.from_array(File(f'{save_root}/Y_2dnorm_ave.h5', 'r')['default'], chunks=(1, -1, -1, -1))
    Y_svd_ = da.from_zarr(f'{save_root}/local_pca_data.zarr')
    Y_svd_ = Y_svd_.rechunk((1, -1, -1, -1))
    mask_ave_ =  Y_d_ave_.map_blocks(lambda: v: intesity_mask(v, percentile=percentile), dtype='bool').compute()
    mask_snr = mask_ave_.copy().squeeze(axis=-1) #drop last dim
    Cn_list = np.zeros(mask_ave_.shape).squeeze(axis=-1) #drop last dim
    for ii in range(Y_svd_.shape[0]):
        _ = Y_svd_[ii].compute().copy()
        _[~mask_ave_[ii].squeeze()] = 0
        if not is_skip_snr:
            mask_snr_ = snr_mask(_)
            _[mask_snr_] = 0
            mask_snr[ii] = np.logical_not(mask_snr_)
        Cn, _ = correlation_pnr(_, skip_pnr=True)
        Cn_list[ii] = Cn
    mask = mask_ave_ & np.expand_dims(mask_snr, -1)
    save_h5(f'{save_root}/mask_map.h5', mask, dtype='bool')
    save_h5(f'{save_root}/local_correlation_map.h5', Cn_list, dtype='float32')
    Y_svd_ = Y_svd_.rechunk((-1, -1, -1, 1))
    _ = Y_svd_.map_blocks(lambda v: mask_blocks(v, mask=mask), dtype='float32')
    refresh_workers(cluster, numCores=numCores)
    if save_masked_data:
        _.to_zarr(f'{save_root}/masked_local_pca_data.zarr', overwrite=True)
        refresh_workers(cluster, numCores=numCores)
    _[:, :, :, ::dt].to_zarr(f'{save_root}/masked_downsampled_local_pca_data.zarr')
    cluster.stop_all_jobs()
    time.sleep(10)
    return None

def demix_cells(save_root, nsplit = 8, numCores = 200):
    '''
      1. local pca denoise
      2. cell segmentation
    '''
    cluster, client = fdask.setup_workers(numCores)
    print(client)
    Y_svd_ = da.from_zarr(f'{save_root}/masked_local_pca_data.zarr')
    Cn_list = File(f'{save_root}/local_correlation_map.h5', 'r')['default'].value
    _, xdim, ydim = Cn_list.shape
    Cn_list = da.from_array(np.expand_dims(Cn_list, -1), chunks=(1, xdim//nsplit, ydim//nsplit, -1))
    Y_svd_ = Y_svd_.rechunk((1, xdim//nsplit, ydim//nsplit, -1))
    if not os.path.exists(f'{save_root}/demix_rlt/'):
        os.mkdir(f'{save_root}/demix_rlt/')
    da.map_blocks(lambda a, b:demix_blocks(a, b, save_folder=save_root), Y_svd_, Cn_list, chunks=(1, 1, 1, 1), dtype='int8').compute()
    cluster.stop_all_jobs()
    cluster.close()
    return None


def check_demix_cells(save_root, block_id, nsplit=8, plot_global=True):
    import matplotlib.pyplot as plt
    import pickle
    Y_d_ave_ = da.from_array(File(f'{save_root}/Y_2dnorm_ave.h5', 'r')['default'], chunks=(1, -1, -1, -1))
    mask_ = da.from_array(File(f'{save_root}/mask_map.h5', 'r')['default'], chunks=(1, -1, -1, -1))
    Cn_list = File(f'{save_root}/local_correlation_map.h5', 'r')['default'].value
    _, xdim, ydim = Cn_list.shape
    Cn_list = da.from_array(np.expand_dims(Cn_list, -1), chunks=(1, xdim//nsplit, ydim//nsplit, -1))
    Y_d_ave_ = Y_d_ave_.rechunk((1, xdim//nsplit, ydim//nsplit, -1))
    mask_ = mask_.rechunk((1, xdim//nsplit, ydim//nsplit, -1))
    v_max = np.percentile(Y_d_ave_, 90)
    fname = f'{save_root}/demix_rlt/period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    if os.path.exists(fname+'_rlt.pkl'):
        with open(fname+'_rlt.pkl', 'rb') as f:
            try:
                rlt_ = pickle.load(f)
                A = rlt_['fin_rlt']['a']
                A_ = A[:, (A>0).sum(axis=0)>40]
                A_comp = np.zeros(A_.shape[0])
                A_comp[A_.sum(axis=-1)>0] = np.argmax(A_[A_.sum(axis=-1)>0, :], axis=-1) + 1
                plt.imshow(Y_d_ave_block.squeeze(), cmap=plt.cm.gray)
                plt.imshow(A_comp.reshape(ydim//nsplit, xdim//nsplit).T, cmap=plt.cm.nipy_spectral_r, alpha=0.7)
                plt.axis('off')
                plt.show()
            except:
                print('None components')
    plt.imshow(Y_d_ave_block.squeeze())
    plt.axis('off')
    plt.show()
    if plot_global:
        area_mask = np.zeros((xdim, ydim)).astype('bool')
        area_mask[block_id[1]*x_:block_id[1]*x_+x_, block_id[2]*y_:block_id[2]*y_+y_]=True
        plt.figure(figsize=(16, 16))
        plt.imshow(Y_d_ave_[block_id[0]].squeeze(), vmax=v_max)
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
    print(client)
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
    trans_data_t = trans_data_.transpose((1, 2, 3, 0)).rechunk((1, chunk_x//4, chunk_y//4, -1))
    baseline_t = trans_data_t.map_blocks(lambda v: baseline(v, window=window, percentile=percentile), dtype='float32')
    min_t = trans_data_t.map_blocks(lambda v: np.min(np.percentile(v, 0.3), 0), dtype='float32')
    dff = (trans_data_t-baseline_t)/(baseline_t-min_t)
    return dff.compute()


def compute_cell_dff_raw(dir_root, save_root, numCores=20, window=100, percentile=20):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''
#     # set worker
#     cluster, client = fdask.setup_workers(numCores)
#     files = sorted(glob(dir_root+'/*.h5'))
#     chunks = File(files[0],'r')['default'].shape
#     data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])
#     # pixel denoise
#     cameraInfo = getCameraInfo(dir_root)
#     denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraInfo=cameraInfo))
#     trans_affine_ = np.load(f'{save_root}/trans_affs.npy')
#     trans_affine_ = da.from_array(trans_affine_, chunks=(1,4,4))
#     # apply affine transform
#     trans_data_ = da.map_blocks(apply_transform3d, denoised_data, trans_affine_, chunks=(1, *denoised_data.shape[1:]), dtype='float32')
#     # baseline
#     chunk_x, chunk_y = chunks[-2:]
#     trans_data_t = trans_data_.transpose((1, 2, 3, 0)).rechunk((1, chunk_x//4, chunk_y//4, -1))
#     baseline_t = trans_data_t.map_blocks(lambda v: baseline(v, window=window, percentile=percentile), dtype='float32')
    return None


def compute_cell_dff_NMF(dir_root, save_root, numCores=20, window=100, percentile=20, min_p=0.3):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''

#     # set worker
#     cluster, client = fdask.setup_workers(numCores)
#     files = sorted(glob(dir_root+'/*.h5'))
#     chunks = File(files[0],'r')['default'].shape
#     data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])
#     # pixel denoise
#     cameraInfo = getCameraInfo(dir_root)
#     denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraInfo=cameraInfo))
#     trans_affine_ = np.load(f'{save_root}/trans_affs.npy')
#     trans_affine_ = da.from_array(trans_affine_, chunks=(1,4,4))
#     # apply affine transform
#     trans_data_ = da.map_blocks(apply_transform3d, denoised_data, trans_affine_, chunks=(1, *denoised_data.shape[1:]), dtype='float32')
#     # baseline
#     trans_data_t = trans_data_.transpose((1, 2, 3, 0)).rechunk((1, 1, 1, -1))
#     baseline_t = trans_data_t.map_blocks(lambda v: baseline(v, window=window, percentile=percentile), dtype='float32')
#     Y_d = trans_data_t.map_blocks(lambda v: v - baseline(v, window=100, percentile=20), dtype='float32')
    return None


if __name__ == '__main__':
    # dir_root = '/nrs/ahrens/Yu/SPIM/active_dataset/glia_neuron_imaging/20161109/fish2/20161109_2_1_6dpf_GFAP_GC_Huc_RG_GA_CL_fb_OL_f0_0GAIN_20161109_211950/raw'
    # save_root = '.'
    # preprocessing(dir_root, save_root, numCores=100, window=1000, percentile=20)
    print('------')
