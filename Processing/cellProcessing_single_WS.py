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
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'


def print_client_links(cluster):
    print(f'Scheduler: {cluster.scheduler_address}')
    print(f'Dashboard link: {cluster.dashboard_link}')
    return None


def preprocessing(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, window=100, percentile=20, nsplit = 4):
    '''
      1. pixel denoise
      2. registration -- save registration file
      3. detrend using percentile baseline
      4. local pca denoise
    '''

    # set worker
    cluster, client = fdask.setup_workers(is_local=True)
    print_client_links(cluster)

    files = sorted(glob(dir_root+'/*.h5'))
    chunks = File(files[0],'r')['default'].shape
    data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])

    # pixel denoise
    cameraInfo = getCameraInfo(dir_root)
    denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraNoiseMat=cameraNoiseMat, cameraInfo=cameraInfo))

    # save and compute reference image
    print('Compute reference image ---')
    if not os.path.exists(f'{save_root}/motion_fix_.h5'):
        med_win = len(denoised_data)//2
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
    if not os.path.exists(f'{save_root}/motion_corrected_data.zarr'):
        print('Apply registration ---')
        trans_data_ = da.map_blocks(apply_transform3d, denoised_data, trans_affine_, chunks=(1, *denoised_data.shape[1:]), dtype='float32')
        # time * z * x * y
        trans_data_t = trans_data_.rechunk((-1, 1, chunks[1]//nsplit, chunks[2]//nsplit))
        trans_data_t.to_zarr(f'{save_root}/motion_corrected_data.zarr')

    # detrend
    if not os.path.exists(f'{save_root}/detrend_data.zarr'):
        print('Compute detrend data ---')
        trans_data_t = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
        Y_d = trans_data_t.map_blocks(lambda v: v - baseline(v, window=window, percentile=percentile), dtype='float32')
        Y_d.to_zarr(f'{save_root}/detrend_data.zarr')

    fdask.terminate_workers(cluster, client)
    time.sleep(10)
    return None


def local_pca_on_mask(save_root, is_dff=False, numCores=20):
    from dask.distributed import Client, LocalCluster
    cluster, client = fdask.setup_workers(numCores)
    print_client_links(cluster)
    Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
    if is_dff:
        Y_t = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
        Y_d = Y_d/(Y_t - Y_d)
    mask = da.from_zarr(f'{save_root}/mask_map.zarr')
    Y_svd = da.map_blocks(fb_pca_block, Y_d, mask, dtype='float32')
    Y_svd.to_zarr(f'{save_root}/masked_local_pca_data.zarr', overwrite=True)
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
    Y_d_ave = da.from_zarr(f'{save_root}/Y_max.zarr')
    mask = da.from_zarr(f'{save_root}/mask_map.zarr')
    Y_d_ave_ = Y_d_ave.blocks[block_id].squeeze().compute(scheduler='threads')
    mask_ = mask.blocks[block_id].squeeze().compute(scheduler='threads')
    v_max = np.percentile(Y_d_ave_, 99)
    _, xdim, ydim, _ = Y_d_ave.shape
    _, x_, y_, _ = Y_d_ave.chunksize
    try:
        A_ = load_A_matrix(save_root=save_root, block_id=block_id, min_size=0)
        print(A_.shape[-1])
        plt.figure(figsize=(8, 8))
        for n in range(A_.shape[-1]):
            n_max = A_[:, n].max()
            A_[A_[:,n]<n_max*0, n] = 0
        A_comp = np.zeros(A_.shape[0])
        A_comp[A_.sum(axis=-1)>0] = np.argmax(A_[A_.sum(axis=-1)>0, :], axis=-1) + 1
        A_comp[A_comp>0] = A_comp[A_comp>0]%20+1
        plt.imshow(Y_d_ave_, vmax=v_max, cmap='gray')
        plt.imshow(A_comp.reshape(y_, x_).T, cmap=plt.cm.nipy_spectral_r, alpha=1.0)
#         A_comp = A_.sum(axis=-1)
#         plt.imshow(A_comp.reshape(y_, x_).T)
#         for n in range(A_.shape[-1]):
#             plt.imshow(A_[:, n].reshape(y_, x_).T)
#             plt.show()
        if plot_mask:
            plt.imshow(mask_, cmap='gray', alpha=0.5)
        plt.title('Components')
        plt.axis('off')
        plt.show()
    except:
        print('No components')

#     plt.imshow(Y_d_ave_, vmax=v_max)
#     plt.title('Max Intensity')
#     plt.axis('off')
#     plt.show()

    if plot_global:
        area_mask = np.zeros((xdim, ydim)).astype('bool')
        area_mask[block_id[1]*x_:block_id[1]*x_+x_, block_id[2]*y_:block_id[2]*y_+y_]=True
        plt.figure(figsize=(16, 16))
        plt.imshow(Y_d_ave[block_id[0]].squeeze().compute(scheduler='threads'), vmax=v_max)
        plt.imshow(area_mask, cmap='gray', alpha=0.3)
        plt.axis('off')
        plt.show()
    return None


def check_demix_cells_layer(save_root, nlayer, nsplit=8):
    import matplotlib.pyplot as plt
    Y_d_ave = da.from_zarr(f'{save_root}/Y_max.zarr')
    Y_d_ave_ = Y_d_ave.blocks[nlayer].squeeze().compute(scheduler='threads')
    v_max = np.percentile(Y_d_ave_, 99.9)
    _, xdim, ydim, _ = Y_d_ave.shape
    _, x_, y_, _ = Y_d_ave.chunksize
    A_mat = np.zeros((xdim, ydim))
    n_comp = 1
    for nx in range(nsplit):
        for ny in range(nsplit):
            try:
                A_ = load_A_matrix(save_root=save_root, block_id=(nlayer, nx, ny, 0), min_size=0)
                for n in range(A_.shape[-1]):
                    n_max = A_[:, n].max()
                    A_[A_[:,n]<n_max*0.2, n] = 0
                A_comp = np.zeros(A_.shape[0])
                A_comp[A_.sum(axis=-1)>0] = np.argmax(A_[A_.sum(axis=-1)>0, :], axis=-1) + n_comp
                A_mat[x_*nx:x_*(nx+1), y_*ny:y_*(ny+1)] =A_comp.reshape(y_, x_).T
#                 A_mat[x_*nx:x_*(nx+1), y_*ny:y_*(ny+1)] = A_.sum(axis=-1).reshape(y_, x_).T
                n_comp = A_mat.max()+1
            except:
                pass

    plt.figure(figsize=(8, 8))
    A_mat[A_mat>0] = A_mat[A_mat>0]%60+1
    plt.imshow(A_mat, cmap=plt.cm.nipy_spectral_r)
#     plt.imshow(A_mat, vmax=A_mat.max()*0.6)
    plt.title('Components')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(Y_d_ave_, vmax=v_max)
    plt.title('Max Intensity')
    plt.axis('off')
    plt.show()
    return None


def compute_cell_dff_pixels(save_root, numCores=20):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''
    # set worker
    cluster, client = fdask.setup_workers(numCores)
    print_client_links(cluster)
    trans_data_t = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
    Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
    baseline_t = da.map_blocks(baseline_from_Yd, trans_data_t, Y_d, dtype='float32')
    dff = Y_d/baseline_t
    dff.to_zarr(f'{save_root}/pixel_dff.zarr', overwrite=True)
    cluster.stop_all_jobs()
    cluster.close()
    return None


def compute_cell_dff_raw(save_root, numCores=20):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''
    # set worker
    if not os.path.exists(f'{save_root}/cell_raw_dff'):
        os.mkdir(f'{save_root}/cell_raw_dff')
    cluster, client = fdask.setup_workers(numCores)
    print_client_links(cluster)
    trans_data_t = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
    Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
    baseline_t = da.map_blocks(baseline_from_Yd, trans_data_t, Y_d, dtype='float32')
    if not os.path.exists(f'{save_root}/cell_raw_dff'):
        os.makedirs(f'{save_root}/cell_raw_dff')
    da.map_blocks(compute_cell_raw_dff, baseline_t, Y_d, dtype='float32', chunks=(1, 1, 1, 1), save_root=save_root).compute()#.compute(scheduler='single-threaded')
    cluster.stop_all_jobs()
    cluster.close()
    return None


def compute_cell_dff_NMF(save_root, numCores=20, dt=3):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''
    # set worker
    if not os.path.exists(f'{save_root}/cell_nmf_dff'):
        os.mkdir(f'{save_root}/cell_nmf_dff')
    cluster, client = fdask.setup_workers(numCores)
    print_client_links(cluster)
    trans_data_t = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
    Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
    baseline_t = da.map_blocks(baseline_from_Yd, trans_data_t, Y_d, dtype='float32')
    pca_data = da.from_zarr(f'{save_root}/masked_local_pca_data.zarr')
    if not os.path.exists(f'{save_root}/cell_nmf_dff'):
        os.makedirs(f'{save_root}/cell_nmf_dff')
    da.map_blocks(compute_cell_denoise_dff, baseline_t, pca_data, dtype='float32', chunks=(1, 1, 1, 1), save_root=save_root, dt=dt).compute()
    cluster.stop_all_jobs()
    cluster.close()
    return None
