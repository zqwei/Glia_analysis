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
    valid_ = np.zeros(num_cell).astype('bool')
    for n_cell in tqdm(range(num_cell)):
        _, p = acorr_ljungbox(zdFF[n_cell], lags=lags)
        if (p<.05).sum()>0:
            valid_[n_cell]=True
    return valid_


def layer_corr(zdFF, A_loc, corr_thres=0.2, corr_size=10):
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