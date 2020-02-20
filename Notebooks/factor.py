import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.stats import zscore
from factor_analyzer import Rotator

def factor_(dFF, n_c=10, noise_thres=0.8):
    dFFz=zscore(dFF, axis=-1)
    FA_=FactorAnalysis(n_components=n_c, svd_method='randomized',random_state=None)
    # first fitting with all neurons
    FA_.fit(dFFz.T)
    valid_cell=FA_.noise_variance_<noise_thres
    # second fitting with less-noisy neurons
    FA_.fit(dFFz[valid_cell].T)
    # rotation
    lam=FA_.components_.T
    Rot_=Rotator(power=4)
    loadings, rotation_mtx, phi=Rot_._promax(lam)
    scores=FA_.transform(dFFz[valid_cell].T).T
    scores_rot=np.matmul(np.linalg.inv(rotation_mtx), scores)
    return valid_cell, lam, loadings, rotation_mtx, phi, scores, scores_rot