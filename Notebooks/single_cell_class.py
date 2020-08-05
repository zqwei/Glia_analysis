import numpy as np
import os, sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from fish_proc.utils.np_mp import parallel_to_single
from tqdm import tqdm
from scipy.stats import ranksums, wilcoxon
from statsmodels.multivariate.manova import MANOVA


def pulse_stats(dff_, pulse_trial, nopulse_trial):
    dff_ = dff_.squeeze()
    num_pulse = len(pulse_trial)
    num_nopulse = len(nopulse_trial)
    dff_pulse = np.zeros((num_pulse, 7))
    dff_nopulse = np.zeros((num_nopulse, 7))
    
    for n, trial in enumerate(pulse_trial):
        dff_pulse[n] = dff_[trial-2:trial+5] - dff_[trial-2:trial].mean()
    
    for n, trial in enumerate(nopulse_trial):
        dff_nopulse[n] = dff_[trial+3:trial+10] - dff_[trial+3:trial+5].mean() 
    
    p_mean = np.zeros(3)
    _, p_mean[0] = ranksums(dff_pulse.sum(axis=-1), dff_nopulse.sum(axis=-1))
    _, p_mean[1] = wilcoxon(dff_pulse.sum(axis=-1))
    _, p_mean[2] = wilcoxon(dff_nopulse.sum(axis=-1))
    
    p_vec = np.zeros((3, 7))
    for n in range(7):
        _, p_vec[0, n] = ranksums(dff_pulse[:, n], dff_nopulse[:, n])
        _, p_vec[1, n] = wilcoxon(dff_pulse[:, n])
        _, p_vec[2, n] = wilcoxon(dff_nopulse[:, n])
    
    if (p_mean[0]<0.05) or ((p_vec[0]<0.05).sum()>3):
        x_ = np.vstack([dff_pulse, dff_nopulse])
        y_ = np.r_[np.zeros(num_pulse), np.ones(num_nopulse)]
        try:
            p_manova = MANOVA(x_, y_).mv_test().results['x0']['stat']
        except:
            p_manova = None
    else:
        p_manova = None
    
    return np.array([p_mean, p_vec, p_manova, dff_pulse.mean(axis=0), dff_nopulse.mean(axis=0)])[None,:],


def motor_stats(dff_, swim_trial, noswim_trial, swim_len, pre_len):
    dff_ = dff_.squeeze()
    num_swim = len(swim_trial)
    num_noswim = len(noswim_trial)
    dff_swim = np.zeros((num_swim, swim_len+pre_len))
    dff_noswim = np.zeros((num_noswim, swim_len+pre_len))
    
    for n, trial in enumerate(swim_trial):
        dff_swim[n] = dff_[trial-pre_len:trial+swim_len] - dff_[trial-pre_len:trial].mean()
    
    for n, trial in enumerate(noswim_trial):
        dff_noswim[n] = dff_[trial:trial+pre_len+swim_len] - dff_[trial:trial+pre_len].mean() 
    
    p_mean = np.zeros(3)
    _, p_mean[0] = ranksums(dff_swim.sum(axis=-1), dff_noswim.sum(axis=-1))
    _, p_mean[1] = wilcoxon(dff_swim.sum(axis=-1))
    _, p_mean[2] = wilcoxon(dff_noswim.sum(axis=-1))
    
    p_vec = np.zeros((3, swim_len+pre_len))
    for n in range(swim_len+pre_len):
        _, p_vec[0, n] = ranksums(dff_swim[:, n], dff_noswim[:, n])
        _, p_vec[1, n] = wilcoxon(dff_swim[:, n])
        _, p_vec[2, n] = wilcoxon(dff_noswim[:, n])
    
    if (p_mean[0]<0.05) or ((p_vec[0]<0.05).sum()>3):
        x_ = np.vstack([dff_swim, dff_noswim])
        y_ = np.r_[np.zeros(num_swim), np.ones(num_noswim)]
        try:
            p_manova = MANOVA(x_, y_).mv_test().results['x0']['stat']
        except:
            p_manova = None
    else:
        p_manova = None
    
    return np.array([p_mean, p_vec, p_manova, dff_swim.mean(axis=0), dff_noswim.mean(axis=0)])[None,:],


def smooth(a, kernel):
    return np.convolve(a, kernel, 'full')[kernel.shape[0]//2:-(kernel.shape[0]//2)]

def boxcarKernel(sigma=60):
    kernel = np.ones(sigma)
    return kernel/kernel.sum()


def gaussKernel(sigma=20):
    kernel = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(np.arange(-sigma*3,sigma*3+1)**2)/(2*sigma**2))
    return kernel/kernel.sum()