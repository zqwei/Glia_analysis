import numpy as np
import pandas as pd
import os, sys
from glob import glob
from h5py import File
from fish_proc.utils.getCameraInfo import getCameraInfo

def pixelDenoiseImag(img, cameraNoiseMat='', cameraInfo=None):
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


def load_bz2file(file, dims):
    import bz2
    import numpy as np
    data = bz2.BZ2File(file,'rb').read()
    im = np.frombuffer(data,dtype='int16')
    return im.reshape(dims[-1::-1])


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


def baseline(data, window=100, percentile=15, downsample=10, axis=-1):
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


def fb_pca_block(block, mask_block, block_id=None):
    # using fb pca instead of local pca from fish
    # from fbpca import pca
    from sklearn.utils.extmath import randomized_svd
    from numpy import expand_dims
    if mask_block.sum()==0:
        return np.zeros(block.shape)
    M = (block-block.mean(axis=-1, keepdims=True)).squeeze()
    M[~mask_block.squeeze()] = 0
    dimsM = M.shape
    M = M.reshape((np.prod(dimsM[:-1]),dimsM[-1]),order='F')
    k = min(min(M.shape)//4, 300)
    # [U, S, Va] = pca(M.T, k=k, n_iter=20, raw=True)
    [U, S, Va] = randomized_svd(M.T, k, n_iter=10, power_iteration_normalizer='QR')
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


def demix_blocks(block, mask_block, save_folder='.', is_skip=True, params=None, block_id=None):
    # this uses old parameter set up
    import pickle
    from pathlib import Path
    import sys
    from fish_proc.demix import superpixel_analysis as sup
    from fish_proc.utils.snr import local_correlations_fft
    is_demix = False
    
    # set fname for blocks
    fname = f'{save_folder}/demix_rlt/period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    # set no processing conditions
    if os.path.exists(fname+'_rlt.pkl') and is_skip:
        return np.zeros([1]*4)
    if mask_block.sum() ==0:
        Path(fname+'_empty_rlt.tmp').touch()
        return np.zeros([1]*4)
    M = block.squeeze().copy()
    M[~mask_block.squeeze()] = 0
    Cblock = local_correlations_fft(M, is_mp=False)
    if (Cblock>0).sum()==0:
        Path(fname+'_empty_rlt.tmp').touch()
        return np.zeros([1]*4)
    
    # get parameters
    if params is None:
        cut_off_point = np.percentile(Cblock[:], [99, 95, 80, 50])
        length_cut=[60, 40, 40, 40]
        max_allow_neuron_size=0.15
        patch_size=[10, 10]
        max_iter=50
        max_iter_fin=90
        update_after=40
    else:
        if params['cut_perc']:
            cut_off_point = np.percentile(Cblock[:], params['cut_off_point'])
        else:
            cut_off_point = params['cut_off_point']
        length_cut=params['length_cut']
        max_allow_neuron_size=params['max_allow_neuron_size']
        patch_size=params['patch_size']
        max_iter=params['max_iter']
        max_iter_fin=params['max_iter_fin']
        update_after=params['update_after']
    pass_num = len(cut_off_point)
    pass_num_max = len(cut_off_point)
    th = [1] * pass_num_max
    residual_cut = [0.6] * pass_num_max
    # demix
    while not is_demix and pass_num>=0:
        try:
            rlt_= sup.demix_whole_data(M, cut_off_point[pass_num_max-pass_num:], length_cut=length_cut[pass_num_max-pass_num:],
                                       th=th, pass_num=pass_num, residual_cut=residual_cut, corr_th_fix=0.3, max_allow_neuron_size=max_allow_neuron_size,
                                       merge_overlap_thr=0.6, patch_size=patch_size, text=False, bg=False, max_iter=max_iter,
                                       max_iter_fin=max_iter_fin, update_after=update_after)
            is_demix = True
        except:
            print(f'fail at pass_num {pass_num}', flush=True)
            is_demix = False
            pass_num -= 1
    try:
        with open(fname+'_rlt.pkl', 'wb') as f:
            pickle.dump(rlt_, f)
    except:
        Path(fname+'_empty_rlt.tmp').touch()
        pass
    return np.zeros([1]*4)


def sup_blocks(block, mask_block, save_folder='.', is_skip=True, block_id=None):
    # this uses old parameter set up
    import pickle
    from pathlib import Path
    import sys
    from fish_proc.demix import superpixel_analysis as sup
    from fish_proc.utils.snr import local_correlations_fft
    is_demix = False
    
    # set fname for blocks
    fname = 'period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    # set no processing conditions
    sup_fname = f'{save_folder}/sup_demix_rlt/'+fname
    demix_fname = f'{save_folder}/demix_rlt/'+fname
    # file exist -- skip
    if os.path.exists(sup_fname+'_rlt.npz') and is_skip:
        return np.zeros([1]*4)
    # no pixels -- skip
    if mask_block.sum() ==0:
        Path(sup_fname+'_empty_rlt.tmp').touch()
        return np.zeros([1]*4)
    M = block.squeeze().copy()
    M[~mask_block.squeeze()] = 0
    Cblock = local_correlations_fft(M, is_mp=False)
    # no corrlated pixel -- skip
    if (Cblock>0).sum()==0:
        Path(sup_fname+'_empty_rlt.tmp').touch()
        return np.zeros([1]*4)
    
    cut_off_point = np.percentile(Cblock[:], 10) # set 1% correlation as threshould
    cut_off_point = max(cut_off_point, 0.01) # force it larger than 0.01
    _, x_, y_, _ = block.shape
    
    # load demixed A matrix
    try:
        A_ = load_A_matrix(save_root=save_folder, ext='', block_id=block_id, min_size=0)
    except:
        A_ = np.zeros((x_*y_, 1))
    if A_ is None:
        A_ = np.zeros((x_*y_, 1))
    # reset small weights to zeros
    for n in range(A_.shape[-1]):
        n_max = A_[:, n].max()
        A_[A_[:,n]<n_max*0.2, n] = 0
    # valid_pixels
    valid_pixels = A_.sum(-1)>0
    valid_pixels = valid_pixels.reshape(x_, y_, order='F')
    M[valid_pixels] = 0
    
    if (M.sum(-1)>0).sum()==0:
        np.savez(sup_fname+'_rlt.npz', A=A_, A_ext=np.zeros(x_*y_))
    try:
        rlt_= sup.demix_whole_data(M, [cut_off_point], length_cut=[10],th=[0], pass_num=1, residual_cut=[0.6], 
                                   corr_th_fix=0.3, max_allow_neuron_size=.99, merge_overlap_thr=0.6, 
                                   patch_size=[10, 10], text=False, bg=False, max_iter=0,max_iter_fin=0, update_after=0)
    except Exception as e:
        print(e)
        print(block_id)
    A_ext=rlt_['fin_rlt']['a']
    np.savez(sup_fname+'_rlt.npz', A=A_, A_ext=A_ext)
    return np.zeros([1]*4)


def demix_file_name_block(save_root='.', ext='', block_id=None):
    fname = f'{save_root}/demix_rlt{ext}/period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    return fname+'_rlt.pkl'


def load_A_matrix(save_root='.', ext='', block_id=None, min_size=40):
    import pickle
    fname = demix_file_name_block(save_root=save_root, ext=ext, block_id=block_id)
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
