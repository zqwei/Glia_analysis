import numpy as np
import os, sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import fish_proc.wholeBrainDask.cellProcessing_single_WS as fwc
import dask.array as da
import pandas as pd
from glob import glob
from tqdm import tqdm


def ep2frame(camtrig, thres=3.8):
    arr_ = (camtrig>thres).astype('int')
    return np.where((arr_[:-1]-arr_[1:])==-1)[0]+1


def WNtest(zdFF, lags=30):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    num_cell = zdFF.shape[0]
    valid_ = np.zeros(num_cell).astype('int')
    for n_cell in tqdm(range(num_cell)):
        _, p = acorr_ljungbox(zdFF[n_cell], lags=lags)
        valid_[n_cell]=(p<.05).sum()
    return valid_


def layer_corr(zdFF, A_loc, corr_thres=0.25, corr_size=10):
    from scipy.stats import spearmanr
    num_cell = zdFF.shape[0]
    num_z = A_loc[:,0].max()+1
    valid_ = np.zeros(num_cell).astype('bool')
    for nz in tqdm(range(num_z)):
        nz_list = np.where(A_loc[:,0]==nz)[0]
        corr_, p_ = spearmanr(zdFF[nz_list], axis=1)
        valid_thres = ((p_<0.05).sum(axis=-1)>corr_size) & ((np.abs(corr_)>corr_thres).sum(axis=-1)>corr_size)
        valid_[nz_list[valid_thres]]=True
    return valid_


def _rankdata_(X, axis=1):
    from bottleneck import rankdata
    X = rankdata(X, axis=axis)
    return (X - X.mean(axis=axis, keepdims=True)) / X.std(axis=axis, keepdims=True)/np.sqrt(X.shape[axis]),


def rankdata_(X, axis=1):
    from fish_proc.utils.np_mp import parallel_to_chunks
    X_r, = parallel_to_chunks(_rankdata_, X, axis=1)
    return X_r


def matmul(a, b):
    import mkl
    mkl.set_num_threads(96)
    print(f'Test thread at: {mkl.get_max_threads()}')
    import numpy as np
    s = np.matmul(a, b)
    mkl.set_num_threads(1)
    print(f'Test thread at: {mkl.get_max_threads()}')
    import numpy as np
    return s





# def covariance_gufunc(x, y):
#     return ((x - x.mean(axis=-1, keepdims=True))
#             * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)


# def pearson_correlation_gufunc(x, y):
#     return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))


# def spearman_correlation_gufunc(x, y):
#     from bottleneck import rankdata
#     x_ranks = rankdata(x, axis=-1)
#     y_ranks = rankdata(y, axis=-1)
#     return pearson_correlation_gufunc(x_ranks, y_ranks)


# def spearman_correlation(x, y, dim):
#     import xarray as xr
#     return xr.apply_ufunc(
#         spearman_correlation_gufunc, x, y,
#         input_core_dims=[[dim], [dim]],
#         dask='parallelized',
#         output_dtypes=[float])