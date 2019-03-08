import numpy as np
import pandas as pd
import os, sys
from glob import glob
from h5py import File
from fish_proc.utils.getCameraInfo import getCameraInfo

cameraNoiseMat = '/groups/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'

def pixelDenoiseImag(img, cameraInfo=None):
    from fish_proc.pixelwiseDenoising.simpleDenioseTool import simpleDN
    from scipy.ndimage.filters import median_filter
    win_ = 3
    pixel_x0, pixel_x1, pixel_y0, pixel_y1 = [int(_) for _ in cameraInfo['camera_roi'].split('_')]
    pixel_x = (pixel_x0, pixel_x1)
    pixel_y = (pixel_y0, pixel_y1)
    offset = np.load(cameraNoiseMat +'/offset_mat.npy').astype('float32')
    gain = np.load(cameraNoiseMat +'/gain_mat.npy').astype('float32')
    offset_ = offset[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    gain_ = gain[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    if img.ndim == 3:
        filter_win = (1, win_, win_)
    if img.ndim == 4:
        filter_win = (1, 1, win_, win_)
    return median_filter(simpleDN(img, offset=offset_, gain=gain_), size=filter_win)


def estimate_rigid3d(moving, fixed=None, affs=None):
    from fish_proc.imageRegistration.imTrans import ImAffine
    from numpy import expand_dims
    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0
    affs = trans.estimate_rigid3d(fixed, moving.squeeze(), tx_tr=affs).affine
    return expand_dims(affs, 0)


def estimate_rigid3d_affine(moving, fixed=None, affs=None):
    from fish_proc.imageRegistration.imTrans import ImAffine
    from numpy import expand_dims
    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0
    return trans.estimate_rigid3d(fixed, moving.squeeze(), tx_tr=affs)


def estimate_translation3d(moving, fixed=None):
    from fish_proc.imageRegistration.imTrans import ImAffine
    from numpy import expand_dims
    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0
    affs = trans.estimate_translation3d(fixed, moving.squeeze()).affine
    return expand_dims(affs, 0)


def estimate_rigid2d(moving, fixed=None, affs=None, to3=True):
    from fish_proc.imageRegistration.imTrans import ImAffine
    from numpy import expand_dims
    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.factors = [8, 4, 2]
    trans.sigmas = [3.0, 2.0, 1.0]
    trans.ss_sigma_factor = 1.0
    affs = trans.estimate_rigid2d(fixed.max(0), moving.squeeze().max(0), tx_tr=affs).affine
    if to3:
        _ = np.eye(4)
        _[1:, 1:] = affs
        affs = _
    return expand_dims(affs, 0)


def estimate_translation2d(moving, fixed=None, to3=True):
    from fish_proc.imageRegistration.imTrans import ImAffine
    from numpy import expand_dims
    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0
    trans.factors = [8, 4, 2]
    trans.sigmas = [3.0, 1.0, 1.0]
    affs = trans.estimate_translation2d(fixed.max(0), moving.squeeze().max(0)).affine
    if to3:
        _ = np.eye(4)
        _[1:, 1:] = affs
        affs = _
    return expand_dims(affs, 0)


def apply_transform3d(mov, affs):
    from scipy.ndimage.interpolation import affine_transform
    print(mov.squeeze().shape)
    print(affs.squeeze().shape)
    return np.expand_dims(affine_transform(mov.squeeze(), affs.squeeze()), 0)


def save_h5(filename, data, dtype='float32'):
    with File(filename, 'w') as f:
        f.create_dataset('default', data=data.astype(dtype), compression='gzip', chunks=True, shuffle=True)
        f.close()
        

def save_h5_rescale(filename, data, reset_max_int=65535):
    ## np.iinfo(np.uint16).max = 65535
    with File(filename, 'w') as f:
        data_max = data.max()
        data_min = data.min()
        data = (data - data_min)/(data_max - data_min)*reset_max_int
        f.create_dataset('default', data=data.astype(np.uint16), compression='gzip', chunks=True, shuffle=True)
        f.create_dataset('scale', data=np.array([data_min, data_max]), chunks=True, shuffle=True)
        f.close()

        
def baseline(data, window=100, percentile=15, downsample=1, axis=-1):
    """
    Get the baseline of a numpy array using a windowed percentile filter with optional downsampling
    data : Numpy array
        Data from which baseline is calculated
    window : int
        Window size for baseline estimation. If downsampling is used, window shrinks proportionally
    percentile : int
        Percentile of data used as baseline
    downsample : int
        Rate of downsampling used before estimating baseline. Defaults to 1 (no downsampling).
    axis : int
        For ndarrays, this specifies the axis to estimate baseline along. Default is -1.
    """
    from scipy.ndimage.filters import percentile_filter
    from scipy.interpolate import interp1d
    from numpy import ones

    size = ones(data.ndim, dtype='int')
    size[axis] *= window//downsample

    if downsample == 1:
        bl = percentile_filter(data, percentile=percentile, size=size)
    else:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, None, downsample)
        data_ds = data[slices]
        baseline_ds = percentile_filter(data_ds, percentile=percentile, size=size)
        interper = interp1d(range(0, data.shape[axis], downsample), baseline_ds, axis=axis, fill_value='extrapolate')
        bl = interper(range(data.shape[axis]))
    return bl


def robust_sp_trend(mov):
    from fish_proc.denoiseLocalPCA.detrend import trend
    return trend(mov)


# test code for local_pca
def local_pca_test(block, block_id=None):
    # this check number of cpus can be used on each worker
    import multiprocess as mp
    return np.ones([1]*len(block_id))*mp.cpu_count()


def local_pca_block(block, block_id=None):
    from fish_proc.denoiseLocalPCA.denoise import temporal as svd_patch
    from numpy import expand_dims
    dx=4
    nblocks=[20, 20]
    if np.prod(block.shape) == 1:
        Y_svd = block[0]
    else:
        Y_svd, _ = svd_patch(block.squeeze(), nblocks=nblocks, dx=dx, stim_knots=None, stim_delta=0, is_single_core=True)
    return expand_dims(Y_svd, 0)

def local_pca(block):
    from fish_proc.denoiseLocalPCA.denoise import temporal as svd_patch
    from numpy import expand_dims
    dx=4
    nblocks=[64, 64]
    Y_svd, _ = svd_patch(block.squeeze(), nblocks=nblocks, dx=dx, stim_knots=None, stim_delta=0, is_single_core=False)
    return Y_svd


# mask functions
def intesity_mask(blocks, percentile=40):
    return blocks>np.percentile(blocks, percentile)


def snr_mask(Y_svd, std_per=20, snr_per=10):    
    Y_svd = Y_svd.squeeze()
    d1, d2, _ = Y_svd.shape
    mean_ = Y_svd.mean(axis=-1,keepdims=True)
    sn, _ = get_noise_fft(Y_svd - mean_,noise_method='logmexp')
    SNR_ = Y_svd.var(axis=-1)/sn**2    
    Y_d_std = Y_svd.std(axis=-1)
    std_thres = np.percentile(Y_d_std.ravel(), std_per)
    mask = Y_d_std<=std_thres
    snr_thres = np.percentile(np.log(SNR_).ravel(), snr_per)
    mask = np.logical_or(mask, np.log(SNR_)<snr_thres)
    return mask.squeeze()


def mask_blocks(block, mask=None):
    _ = block.copy()
    _[~mask] = 0
    return _


def demix_blocks(block, Cblock, save_folder='.', block_id=None):
    import pickle
    import sys
    from fish_proc.demix import superpixel_analysis as sup
    is_demix = False
    orig_stdout = sys.stdout
    
    fname = f'{save_folder}/demix_rlt/period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
        
    f = open(fname+'_info.txt', 'w')
    sys.stdout = f
    
    if (Cblock>0).sum()==0:
        print('No components in this block', flush=True)
        sys.stdout = orig_stdout
        f.close()
        return np.zeros([1]*4)
    
    if (Cblock[Cblock>0]>0.90).mean()<0.5:
        cut_off_point=np.percentile(Cblock.ravel(), [99, 95, 85, 65])
    else:
        cut_off_point = np.array([0.99, 0.95, 0.90])
    pass_num_max = (cut_off_point>0).sum()
    cut_off_point = cut_off_point[:pass_num_max]
    print(cut_off_point, flush=True)
    pass_num = pass_num_max
    while not is_demix and pass_num>=0:
        try:
            rlt_= sup.demix_whole_data(block.squeeze(), cut_off_point[pass_num_max-pass_num:], length_cut=[20,15,15,15],
                                       th=[1,1,1,1], pass_num=pass_num, residual_cut = [0.6,0.6,0.6,0.6],
                                       corr_th_fix=0.3, max_allow_neuron_size=0.3, merge_corr_thr=cut_off_point[-1],
                                       merge_overlap_thr=0.6, num_plane=1, patch_size=[20, 20], plot_en=False,
                                       TF=False, fudge_factor=1, text=False, bg=False, max_iter=50,
                                       max_iter_fin=100, update_after=10) 
            is_demix = True
        except:
            print(f'fail at pass_num {pass_num}', flush=True)
            is_demix = False
            pass_num -= 1
            
    sys.stdout = orig_stdout
    f.close()
    
    try:    
        with open(fname+'_rlt.pkl', 'wb') as f:
            pickle.dump(rlt_, f)
    except:
        pass

    return np.zeros([1]*4)


def demix_file_name_block(save_root='.', block_id=None):
    fname = f'{save_root}/demix_rlt/period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    return fname+'_rlt.pkl'


def load_A_matrix(save_root='.', block_id=None, min_size=40):
    fname = demix_file_name_block(save_root=save_root, block_id=block_id)
    with open(fname+'_rlt.pkl', 'rb') as f:
        try:
            rlt_ = pickle.load(f)
            A = rlt_['fin_rlt']['a']
            return A[:, (A>0).sum(axis=0)>min_size]
        except:
            return None


def compute_cell_raw_dff(block, save_root='.', window=100, percentile=20, block_id=None):
    from fish_proc.utils.demix import recompute_C_matrix
    fname = demix_file_name_block(save_root=save_root, block_id=block_id)
    if not os.path.exists(fname):
        return np.zeros([1]*4)
    else:
        A, b = load_A_matrix(save_root=save_root, block_id=block_id)
        if A is None:
            return np.zeros([1]*4)
        if A.shape[1] == 0:
            return np.zeros([1]*4)
    
    fsave = f'{save_root}/cell_raw_dff/period_Y_demix_block_'
    for _ in block_id:
        fsave += '_'+str(_)
    fsave += '_rlt.h5'
    
    block_ = block.squeeze(axis=0) # remove z, remaining x, y, t
    d1, d2, _ = block_.shape
    F0 = baseline(block_, window=window, percentile=percentile)
    dF = block_ - F0
    min_t = np.percentile(block, 0.3, axis=-1)
    min_t[min_t>0] = 0
    F0 = F0 - min_t
    cell_F0 = recompute_C_matrix(F0, A)
    cell_dF = recompute_C_matrix(dF, A)
    A = A.reshape((d1, d2, -1), order="F")
    with File(fsave, 'w') as f:
        f.create_dataset('A', data=A, compression='gzip', chunks=True, shuffle=True)
        f.create_dataset('cell_dF', data=cell_dF, compression='gzip', chunks=True, shuffle=True)
        f.create_dataset('cell_F0', data=cell_F0, compression='gzip', chunks=True, shuffle=True)
        f.close()
    return np.zeros([1]*4)


def load_Ab_matrix(save_root='.', block_id=None, min_size=40):
    fname = demix_file_name_block(save_root=save_root, block_id=block_id)
    with open(fname+'_rlt.pkl', 'rb') as f:
        try:
            rlt_ = pickle.load(f)
            A = rlt_['fin_rlt']['a']
            return A[:, (A>0).sum(axis=0)>min_size], rlt_['fin_rlt']['b']
        except:
            return None, None


def pos_sig_correction(mov, dt, axis_=-1):
    return mov - (mov[:, :, dt]).min(axis=axis_, keepdims=True)


def compute_cell_denoise_dff(block, pca_block, save_root='.', dt=5, window=100, percentile=20, block_id=None):
    from fish_proc.utils.demix import recompute_C_matrix
    fname = demix_file_name_block(save_root=save_root, block_id=block_id)
    if not os.path.exists(fname):
        return np.zeros([1]*4)
    else:
        A = load_Ab_matrix(save_root=save_root, block_id=block_id)
        if A is None:
            return np.zeros([1]*4)
        if A.shape[1] == 0:
            return np.zeros([1]*4)
    
    fsave = f'{save_root}/cell_nmf_dff/period_Y_demix_block_'
    for _ in block_id:
        fsave += '_'+str(_)
    fsave += '_rlt.h5'
    
    block_ = block.squeeze(axis=0) # remove z, remaining x, y, t
    d1, d2, _ = block_.shape
    F0 = baseline(block_, window=window, percentile=percentile)
    min_t = np.percentile(block, 0.3, axis=-1)
    min_t[min_t>0] = 0
    cell_F0 = recompute_C_matrix(F0 - min_t, A)
    dF = pos_sig_correction(pca_block.squeeze(axis=0), dt) - b.reshape((d1, d2, 1), order="F")
    cell_dF = recompute_C_matrix(dF, A)
    A = A.reshape((d1, d2, -1), order="F")
    with File(fsave, 'w') as f:
        f.create_dataset('A', data=A, compression='gzip', chunks=True, shuffle=True)
        f.create_dataset('cell_dF', data=cell_dF, compression='gzip', chunks=True, shuffle=True)
        f.create_dataset('cell_F0', data=cell_F0, compression='gzip', chunks=True, shuffle=True)
        f.close()
    return np.zeros([1]*4)
