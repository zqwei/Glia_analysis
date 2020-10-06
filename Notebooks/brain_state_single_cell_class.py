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

def comp_pulse_stats(dff_, cond_trial, comp_trial, pre, post):
    dff_ = dff_.squeeze()
    num_cond = len(cond_trial)
    num_comp = len(comp_trial)
    dff_cond = np.empty((num_cond, pre+post))
    dff_cond[:] = np.nan
    dff_comp = np.empty((num_comp, pre+post))
    dff_comp[:] = np.nan
    
    valid_ = np.zeros(num_cond).astype('bool')
    for n, (trial, trial_end) in enumerate(cond_trial):
        if len(dff_[trial-pre:trial+post])==(pre+post):
            if trial_end<0:
                dff_cond[n] = dff_[trial-pre:trial+post] - dff_[trial-pre:trial].mean()
            else:
                dff_cond[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end] - dff_[trial-pre:trial].mean()
            valid_[n] = True
    dff_cond = dff_cond[valid_]
    
    valid_ = np.zeros(num_comp).astype('bool')
    for n, (trial, trial_end) in enumerate(comp_trial):
        if len(dff_[trial-pre:trial+post])==(pre+post):
            if trial_end<0:
                dff_comp[n] = dff_[trial-pre:trial+post] - dff_[trial-pre:trial].mean()
            else:
                dff_comp[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end] - dff_[trial-pre:trial].mean()
            valid_[n] = True
    dff_comp = dff_comp[valid_]
        
    p_vec = np.zeros((3, pre+post))
    for n in range(pre+post):
        n_valid_cond = ~np.isnan(dff_cond[:, n])
        n_valid_comp = ~np.isnan(dff_comp[:, n])
        _, p_vec[0, n] = ranksums(dff_cond[n_valid_cond, n], dff_comp[n_valid_comp, n])
        try:
            _, p_vec[1, n] = wilcoxon(dff_cond[n_valid_cond, n])
        except:
            p_vec[1, n] = 1
        try:
            _, p_vec[2, n] = wilcoxon(dff_comp[n_valid_comp, n])
        except:
            p_vec[2, n] = 1
        
    return np.array([p_vec, np.nanmean(dff_cond,axis=0), np.nanmean(dff_comp,axis=0)])[None,:],


def comp_pulse_stats_chunks(dff, cond_trial=None, comp_trial=None, pre=None, post=None):
    num_cells = dff.shape[0]
    cell_stats = np.zeros((num_cells, 3)).astype('O')
    for n in range(num_cells):
        _ = comp_pulse_stats(dff[n], cond_trial=cond_trial, comp_trial=comp_trial, pre=pre, post=post)
        cell_stats[n]=_[0]
    return cell_stats

