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


def demix_blocks(block, mask_block, save_folder='.', is_skip=True, block_id=None):
    from skimage.exposure import equalize_adapthist as clahe
    from skimage.morphology import square, dilation
    from skimage.segmentation import watershed
    from sklearn.decomposition import NMF
    from scipy.sparse import csr_matrix

    # set fname for blocks
    fname = 'period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    # set no processing conditions
    sup_fname = f'{save_folder}/sup_demix_rlt/'+fname

    block_img = mask_block.squeeze()
    if block_img.max()==0:
        np.savez(sup_fname+'_rlt.npz', A=np.zeros([np.prod(dims[:-1]),1]))
        return np.zeros([1]*4)
    try:
        img_adapteq = clahe(block_img/block_img.max(), clip_limit=0.03)
    except:
        img_adapteq = block_img
    # initial segments
    segments_watershed = watershed(img_adapteq, markers=100, compactness=0.01)
    min_ = segments_watershed.min()
    max_ = segments_watershed.max()
    vec_segments_watershed = segments_watershed.reshape(-1, 1, order='F')
    a_ini = np.zeros((vec_segments_watershed.shape[0], max_+1))
    for n in range(min_, max_+1):
        if (vec_segments_watershed[:,0]==n).sum()>10:
            _ = (segments_watershed==n)
            _ = dilation(_, square(5)).astype('float')
            _ = _.reshape(-1, order='F')
            a_ini[_>0, n]=1
    a_ini = a_ini[:, a_ini.sum(0)>0]>0

    Yt = block.squeeze()
    dims = Yt.shape;
    T = dims[-1];
    Yt_r = Yt.reshape(np.prod(dims[:-1]),T,order = "F");
    Yt_r[Yt_r<0]=0
    Yt_r = csr_matrix(Yt_r);
    model = NMF(n_components=1, init='custom')
    U_mat = []
    V_mat = []
    if Yt_r.sum()==0:
        np.savez(sup_fname+'_rlt.npz', A=np.zeros([np.prod(dims[:-1]),1]))
        return np.zeros([1]*4)
    for ii, comp in enumerate(a_ini.T):
        y_temp = Yt_r[comp,:].astype('float')
        if y_temp.sum()==0:
            continue
        u_ = np.zeros((np.prod(dims[:-1]),1)).astype('float32')
        u_[list(comp)] = model.fit_transform(y_temp, W=np.array(y_temp.mean(axis=1)),H = np.array(y_temp.mean(axis=0)))
        U_mat.append(u_)
        V_mat.append(model.components_.T)
    if len(U_mat)>1:
        U_mat = np.concatenate(U_mat, axis=1)
        V_mat = np.concatenate(V_mat, axis=1)
    else:
        U_mat = np.zeros([np.prod(dims[:-1]),1])
        V_mat = np.zeros([T,1])

    if U_mat.sum()>0:
        model_ = NMF(n_components=U_mat.shape[-1], init='custom', solver='cd', max_iter=20)
        U = model_.fit_transform(Yt_r.astype('float'), W=U_mat.astype('float64'), H=V_mat.T.astype('float64'))
    else:
        U = np.zeros([np.prod(dims[:-1]),1])
    temp = np.sqrt((U**2).sum(axis=0,keepdims=True))
    U = U/temp
    U[U<U.max(axis=-1, keepdims=True)*.2]=0
    np.savez(sup_fname+'_rlt.npz', A=U)
    return np.zeros([1]*4)


def demix_file_name_block(save_root='.', ext='', block_id=None):
    fname = f'{save_root}/demix_rlt{ext}/period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    return fname+'_rlt.pkl'


def sup_file_name_block(save_root='.', ext='', block_id=None):
    fname = f'{save_root}/sup_demix_rlt{ext}/period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    return fname+'_rlt.npz'


def load_A_matrix(save_root='.', ext='', block_id=None, min_size=40):
    fname = sup_file_name_block(save_root=save_root, ext=ext, block_id=block_id)
    _ = np.load(fname, allow_pickle=True)
    return _['A']


def pos_sig_correction(mov, dt, axis_=-1):
    return mov - (mov[:, :, dt]).min(axis=axis_, keepdims=True)


def compute_cell_raw_dff(block_F0, block_dF, save_root='.', ext='', block_id=None):
    from fish_proc.utils.demix import recompute_C_matrix
    _, x_, y_, _ = block_F0.shape
    A_= load_A_matrix(save_root=save_root, ext=ext, block_id=block_id, min_size=0)
    if A_.sum()==0:
        return np.zeros([1]*4) # return if no components
    if np.abs(block_dF).sum()==0:
        return np.zeros([1]*4) # return if out of brain
    
    fsave = f'{save_root}/cell_raw_dff/period_Y_demix_block_'
    for _ in block_id:
        fsave += '_'+str(_)
    fsave += '_rlt.h5'
    
    F0 = block_F0.squeeze(axis=0).reshape((x_*y_, -1), order='F')
    dF = block_dF.squeeze(axis=0).reshape((x_*y_, -1), order='F')
    cell_F0 = np.linalg.inv(A_.T.dot(A_)).dot(np.matmul(A_.T, F0))
    cell_dF = np.linalg.inv(A_.T.dot(A_)).dot(np.matmul(A_.T, dF))
    
    with File(fsave, 'w') as f:
        f.create_dataset('A_loc', data=np.array([block_id[0], x_*block_id[1], y_*block_id[2]]))
        f.create_dataset('A', data=A_.reshape((x_, y_, -1), order="F"))
        f.create_dataset('cell_dFF', data=cell_dF/cell_F0)
        f.close()
    return np.zeros([1]*4)
