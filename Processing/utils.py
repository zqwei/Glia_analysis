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


def baseline_from_Yd(block_t, block_d):
    min_t = np.percentile(block_t, 0.3, axis=-1, keepdims=True)
    min_t[min_t>0] = 0
    return block_t - block_d - min_t


def baseline_correct(block_b, block_t):
    min_t = np.percentile(block_t, 0.3, axis=-1, keepdims=True)
    min_t[min_t>0] = 0
    min_b = np.min(block_b-min_t, axis=-1, keepdims=True)
    min_b[min_b<=0] = min_b[min_b<=0] - 0.01
    min_b[min_b>0] = 0
    return block_b - min_t - min_b


def robust_sp_trend(mov):
    from fish_proc.denoiseLocalPCA.detrend import trend
    return trend(mov)


def local_pca_block(block, mask, save_folder='.', block_id=None):
    from fish_proc.denoiseLocalPCA.denoise import temporal as svd_patch
    from numpy import expand_dims
    import sys
    import gc
    gc.collect()
    
    orig_stdout = sys.stdout    
    fname = f'{save_folder}/denoise_rlt/block_'
    for _ in block_id:
        fname += '_'+str(_)    
    f = open(fname+'_info.txt', 'w')
    sys.stdout = f

    if np.prod(block.shape) == 1:
        Y_svd = block[0]
        print('Testing data in dask', flush=True)
    else:
        if mask is None:
            print('No mask is applied', flush=True)
        else:
            if mask.sum() == 0:
                print('No valid pixels')
                sys.stdout = orig_stdout
                f.close()
                return np.zeros(block.shape)
            print('mask shape')
            print(mask.shape, flush=True)
        print('block shape')
        print(block.shape, flush=True)
        dx=4
        nblocks=[8, 8]
        Y_svd, _ = svd_patch(block-block.mean(axis=-1, keepdims=True), nblocks=nblocks, dx=dx, stim_knots=None, stim_delta=0, is_single_core=True)
    sys.stdout = orig_stdout
    f.close()
    gc.collect()
    if block.ndim == 4:
        return expand_dims(Y_svd, 0)
    else:
        return Y_svd
    
    
def fb_pca_block(block, mask_block, block_id=None):
    # using fb pca instead of local pca from fish
    from fbpca import pca
    from numpy import expand_dims
    if mask_block.sum()==0:
        return np.zeros(block.shape)    
    M = (block-block.mean(axis=-1, keepdims=True)).squeeze()
    M[~mask_block.squeeze()] = 0
    dimsM = M.shape
    M = M.reshape((np.prod(dimsM[:-1]),dimsM[-1]),order='F')
    k = min(min(M.shape)//4, 600)
    [U, S, Va] = pca(M.T, k=k, n_iter=20, raw=True)
    M_pca = U.dot(np.diag(S).dot(Va))
    M_pca = M_pca.T.reshape(dimsM, order='F')
    return expand_dims(M_pca, 0)


# mask functions
def intesity_mask(blocks, percentile=40):
    return blocks>np.percentile(blocks, percentile)


def intesity_mask_block(blocks, percentile):
    return blocks>np.percentile(blocks, percentile.squeeze())


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
    if block.ndim != 4 or mask.ndim !=4:
        print('error in block shape or mask shape')
        return None
    _ = block.copy()
    _[~mask.squeeze(axis=-1)] = 0
    return _


def demix_blocks(block, mask_block, save_folder='.', is_skip=True, block_id=None):
    # this uses old parameter set up
    import pickle
    import sys
    from fish_proc.demix import superpixel_analysis as sup
    from fish_proc.utils.snr import local_correlations_fft
    is_demix = False
    orig_stdout = sys.stdout
        
    fname = f'{save_folder}/demix_rlt/period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
        
    if os.path.exists(fname+'_rlt.pkl') and is_skip:
        return np.zeros([1]*4)
        
    f = open(fname+'_info.txt', 'w')
    sys.stdout = f
    
    if mask_block.sum() ==0:
        print('No valid pixel in this block', flush=True)
        sys.stdout = orig_stdout
        f.close()
        os.remove(fname+'_info.txt')
        return np.zeros([1]*4)
    
    M = block.squeeze().copy()
    M[~mask_block.squeeze()] = 0
    Cblock = local_correlations_fft(M, is_mp=False)

    if (Cblock>0).sum()==0:
        print('No components in this block', flush=True)
        sys.stdout = orig_stdout
        f.close()
        os.remove(fname+'_info.txt')
        return np.zeros([1]*4)
    
    cut_off_point = [0.95, 0.9, 0.85, 0.70]
    pass_num = 4
    pass_num_max = 4
    while not is_demix and pass_num>=0:
        try:
            rlt_= sup.demix_whole_data(M, cut_off_point[pass_num_max-pass_num:], length_cut=[20, 20, 40, 40],
                                       th=[1,1,1,1], pass_num=pass_num, residual_cut = [0.6,0.6,0.6,0.6],
                                       corr_th_fix=0.3, max_allow_neuron_size=0.05, merge_corr_thr=0.90,
                                       merge_overlap_thr=0.6, num_plane=1, patch_size=[10, 10], plot_en=False,
                                       TF=False, fudge_factor=1, text=False, bg=False, max_iter=50,
                                       max_iter_fin=90, update_after=40) 
            is_demix = True
        except:
            print(f'fail at pass_num {pass_num}', flush=True)
            is_demix = False
            pass_num -= 1
            
    sys.stdout = orig_stdout
    f.close()
    os.remove(fname+'_info.txt')
    
    try:    
        with open(fname+'_rlt.pkl', 'wb') as f:
            pickle.dump(rlt_, f)
    except:
        pass

    return np.zeros([1]*4)


def demix_blocks_old(block, mask_block, save_folder='.', is_skip=True, block_id=None):
    # this uses old parameter set up
    import pickle
    import sys
    from fish_proc.demix import superpixel_analysis as sup
    from fish_proc.utils.snr import local_correlations_fft
    is_demix = False
    orig_stdout = sys.stdout
        
    fname = f'{save_folder}/demix_rlt/period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
        
    if os.path.exists(fname+'_rlt.pkl') and is_skip:
        return np.zeros([1]*4)
        
    f = open(fname+'_info.txt', 'w')
    sys.stdout = f
    
    if mask_block.sum() ==0:
        print('No valid pixel in this block', flush=True)
        sys.stdout = orig_stdout
        f.close()
        return np.zeros([1]*4)
    
    M = block.squeeze().copy()
    M[~mask_block.squeeze()] = 0
    Cblock = local_correlations_fft(M, is_mp=False)

    if (Cblock>0).sum()==0:
        print('No components in this block', flush=True)
        sys.stdout = orig_stdout
        f.close()
        return np.zeros([1]*4)
    
    if (Cblock[Cblock>0]>0.90).mean()<0.5:
        cut_off_point=np.percentile(Cblock.ravel(), [99, 95, 85, 75])
    else:
        cut_off_point = np.array([0.99, 0.95, 0.90])
    pass_num_max = (cut_off_point>0).sum()
    cut_off_point = cut_off_point[:pass_num_max]
    print(cut_off_point, flush=True)
    pass_num = pass_num_max
    while not is_demix and pass_num>=0:
        try:
            rlt_= sup.demix_whole_data(M, cut_off_point[pass_num_max-pass_num:], length_cut=[60, 40, 40, 40],
                                       th=[1,1,1,1], pass_num=pass_num, residual_cut = [0.6,0.6,0.6,0.6],
                                       corr_th_fix=0.3, max_allow_neuron_size=0.05, merge_corr_thr=cut_off_point[-1],
                                       merge_overlap_thr=0.6, num_plane=1, patch_size=[10, 10], plot_en=False,
                                       TF=False, fudge_factor=1, text=False, bg=False, max_iter=50,
                                       max_iter_fin=90, update_after=40) 
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
    import pickle
    fname = demix_file_name_block(save_root=save_root, block_id=block_id)
    with open(fname, 'rb') as f:
        try:
            rlt_ = pickle.load(f)
            A = rlt_['fin_rlt']['a']
            return A[:, (A>0).sum(axis=0)>min_size]
        except:
            return None


def compute_cell_raw_dff(block_F0, block_dF, save_root='.', block_id=None):
    from fish_proc.utils.demix import recompute_C_matrix
    import pickle
    fname = demix_file_name_block(save_root=save_root, block_id=block_id)
    if not os.path.exists(fname):
        return np.zeros([1]*4)
    else:
        A = load_A_matrix(save_root=save_root, block_id=block_id)
        if A is None:
            return np.zeros([1]*4)
        if A.shape[1] == 0:
            return np.zeros([1]*4)
    
    fsave = f'{save_root}/cell_raw_dff/period_Y_demix_block_'
    for _ in block_id:
        fsave += '_'+str(_)
    fsave += '_rlt.h5'
    _, d1, d2, _ = block_F0.shape
    cell_F0 = recompute_C_matrix(block_F0.squeeze(axis=0), A)
    cell_dF = recompute_C_matrix(block_dF.squeeze(axis=0), A)
    A = A.reshape((d1, d2, -1), order="F")
    with File(fsave, 'w') as f:
        f.create_dataset('A', data=A, compression='gzip', chunks=True, shuffle=True)
        f.create_dataset('cell_dF', data=cell_dF, compression='gzip', chunks=True, shuffle=True)
        f.create_dataset('cell_F0', data=cell_F0, compression='gzip', chunks=True, shuffle=True)
        f.close()
    return np.zeros([1]*4)


def load_Ab_matrix(fname, min_size=40):
    import pickle
    with open(fname, 'rb') as f:
        try:
            rlt_ = pickle.load(f)
            A = rlt_['fin_rlt']['a']
            return A[:, (A>0).sum(axis=0)>min_size], rlt_['fin_rlt']['b']
        except:
            return None, None


def pos_sig_correction(mov, dt, axis_=-1):
    return mov - (mov[:, :, dt]).min(axis=axis_, keepdims=True)


def compute_cell_denoise_dff(block_F0, block_dF, save_root='.', dt=5, block_id=None):
    from fish_proc.utils.demix import recompute_C_matrix
    import pickle
    fname = demix_file_name_block(save_root=save_root, block_id=block_id)
    if not os.path.exists(fname):
        return np.zeros([1]*4)
    else:
        A, b = load_Ab_matrix(fname, min_size=40)
        if A is None:
            return np.zeros([1]*4)
        if A.shape[1] == 0:
            return np.zeros([1]*4)
    fsave = f'{save_root}/cell_nmf_dff/period_Y_demix_block_'
    for _ in block_id:
        fsave += '_'+str(_)
    fsave += '_rlt.h5'
    _, d1, d2, _ = block_F0.shape
    cell_F0 = recompute_C_matrix(block_F0.squeeze(axis=0), A)
    dF = pos_sig_correction(block_dF.squeeze(axis=0), dt) - b.reshape((d1, d2, 1), order="F")
    cell_dF = recompute_C_matrix(dF, A)
    A = A.reshape((d1, d2, -1), order="F")
    with File(fsave, 'w') as f:
        f.create_dataset('A', data=A, compression='gzip', chunks=True, shuffle=True)
        f.create_dataset('cell_dF', data=cell_dF, compression='gzip', chunks=True, shuffle=True)
        f.create_dataset('cell_F0', data=cell_F0, compression='gzip', chunks=True, shuffle=True)
        f.close()
    return np.zeros([1]*4)
