import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.stats import zscore
from factor_analyzer import Rotator

def factor_(dFF, n_c=10, noise_thres=0.8):
    dFFz=zscore(dFF, axis=-1)
    FA_=FactorAnalysis(n_components=n_c, svd_method='randomized', random_state=None, iterated_power=7)
    # first fitting with all neurons
    FA_.fit(dFFz.T)
    valid_cell=FA_.noise_variance_<noise_thres
    # second fitting with less-noisy neurons
    if valid_cell.sum()>10:
        FA_.fit(dFFz[valid_cell].T)
        # rotation
        lam=FA_.components_.T
        Rot_=Rotator(power=4)
        loadings, rotation_mtx, phi=Rot_._promax(lam)
        scores=FA_.transform(dFFz[valid_cell].T).T
        scores_rot=np.matmul(np.linalg.inv(rotation_mtx), scores)
        return valid_cell, lam, loadings, rotation_mtx, phi, scores, scores_rot
    else:
        return None, None, None, None, None, None, None


def thres_factor_(x, y, valid_cell, loadings, l_thres_=0.5, shape_thres_=20, min_size_c=10):
    nc = loadings.shape[1]
    loadings_ = loadings.copy()
    loadings_thres = np.abs(loadings).max(axis=0, keepdims=True)
    loadings_[np.abs(loadings)<loadings_thres*l_thres_]=0
    
    for n_ in range(nc):
        l_ = loadings_[:,n_]
        sub_ = l_!=0
        x_shape = np.ptp(x[valid_cell][sub_])
        y_shape = np.ptp(y[valid_cell][sub_])
        if (y_shape==0) or (x_shape==0) or (x_shape/y_shape>shape_thres_) or (y_shape/x_shape>shape_thres_):
            loadings_[:,n_]=0
    valid_c = (np.abs(loadings_)>0).sum(axis=0)>min_size_c
    loadings_=loadings_[:, valid_c]
    return loadings_, valid_c