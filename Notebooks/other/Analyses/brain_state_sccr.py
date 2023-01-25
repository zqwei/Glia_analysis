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


def moving_average(x, win):
    return pd.Series(x).rolling(win, win_type='boxcar', center=True, min_periods=1).mean()


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
                dff_cond[n] = dff_[trial-pre:trial+post]
            else:
                dff_cond[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end]
            valid_[n] = True
    dff_cond = dff_cond[valid_]
    
    valid_ = np.zeros(num_comp).astype('bool')
    for n, (trial, trial_end) in enumerate(comp_trial):
        if len(dff_[trial-pre:trial+post])==(pre+post):
            if trial_end<0:
                dff_comp[n] = dff_[trial-pre:trial+post]
            else:
                dff_comp[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end]
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


def comp_pulse_stats_ref(dff_, cond_trial, comp_trial, cond_trial_ref, comp_trial_ref, pre, post):
    trend_win = 4
    dff_ = dff_.squeeze()
    num_cond = len(cond_trial)
    num_comp = len(comp_trial)
    if cond_trial_ref is None:
        num_cond_ref = 0
    else:
        num_cond_ref = len(cond_trial_ref)
    if comp_trial_ref is None:
        num_comp_ref = 0
    else:
        num_comp_ref = len(comp_trial_ref)
    dff_cond = np.empty((num_cond, pre+post))
    dff_cond[:] = np.nan
    dff_comp = np.empty((num_comp, pre+post))
    dff_comp[:] = np.nan
    dff_cond_ref = np.empty((num_cond_ref, pre+post))
    dff_cond_ref[:] = np.nan
    dff_comp_ref = np.empty((num_comp_ref, pre+post))
    dff_comp_ref[:] = np.nan
    
    valid_ = np.zeros(num_cond).astype('bool')
    for n, (trial, trial_end) in enumerate(cond_trial):
        if len(dff_[trial-pre:trial+post])==(pre+post):
            if trial_end<0:
                dff_cond[n] = dff_[trial-pre:trial+post]
            else:
                dff_cond[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end]
            valid_[n] = True
    dff_cond = dff_cond[valid_]    
    
    valid_ = np.zeros(num_cond_ref).astype('bool')
    if num_cond_ref>0:
        for n, (trial, trial_end) in enumerate(cond_trial_ref):
            if len(dff_[trial-pre:trial+post])==(pre+post):
                if trial_end<0:
                    dff_cond_ref[n] = dff_[trial-pre:trial+post]
                else:
                    dff_cond_ref[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end]
                valid_[n] = True
        dff_cond_ref = dff_cond_ref[valid_]
    if valid_.sum()>0:
        trend = np.nanmean(dff_cond_ref, axis=0)
        trend = moving_average(trend, trend_win)[None, :]
        dff_cond_ = dff_cond - trend
    else:
        dff_cond_ = dff_cond
    
    valid_ = np.zeros(num_comp).astype('bool')
    for n, (trial, trial_end) in enumerate(comp_trial):
        if len(dff_[trial-pre:trial+post])==(pre+post):
            if trial_end<0:
                dff_comp[n] = dff_[trial-pre:trial+post]
            else:
                dff_comp[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end]
            valid_[n] = True
    dff_comp = dff_comp[valid_]
    
    valid_ = np.zeros(num_comp_ref).astype('bool')
    if num_comp_ref>0:
        for n, (trial, trial_end) in enumerate(comp_trial_ref):
            if len(dff_[trial-pre:trial+post])==(pre+post):
                if trial_end<0:
                    dff_comp_ref[n] = dff_[trial-pre:trial+post]
                else:
                    dff_comp_ref[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end]
                valid_[n] = True
        dff_comp_ref = dff_comp_ref[valid_]
    if valid_.sum()>0:
        trend = np.nanmean(dff_comp_ref, axis=0)
        trend = moving_average(trend, trend_win)[None, :]
        dff_comp_ = dff_comp - trend
    else:
        dff_comp_ = dff_comp
        
    p_vec = np.zeros((3, pre+post))
    for n in range(pre+post):
        n_valid_cond = ~np.isnan(dff_cond[:, n])
        n_valid_comp = ~np.isnan(dff_comp[:, n])
        if (n_valid_cond.sum()>0) and (n_valid_comp.sum()>0):
            _, p_vec[0, n] = ranksums(dff_cond[n_valid_cond, n], dff_comp[n_valid_comp, n])
        else:
            p_vec[0, n] = 1
        try:
            _, p_vec[1, n] = wilcoxon(dff_cond[n_valid_cond, n])
        except:
            p_vec[1, n] = 1
        try:
            _, p_vec[2, n] = wilcoxon(dff_comp[n_valid_comp, n])
        except:
            p_vec[2, n] = 1
    
    p_vec_ = np.zeros((3, pre+post))
    for n in range(pre+post):
        n_valid_cond = ~np.isnan(dff_cond[:, n])
        n_valid_comp = ~np.isnan(dff_comp[:, n])
        if (n_valid_cond.sum()>0) and (n_valid_comp.sum()>0):
            _, p_vec_[0, n] = ranksums(dff_cond_[n_valid_cond, n], dff_comp_[n_valid_comp, n])
        else:
            p_vec_[0, n] = 1
        try:
            _, p_vec_[1, n] = wilcoxon(dff_cond_[n_valid_cond, n])
        except:
            p_vec_[1, n] = 1
        try:
            _, p_vec_[2, n] = wilcoxon(dff_comp_[n_valid_comp, n])
        except:
            p_vec_[2, n] = 1
        
    return np.array([p_vec, np.nanmean(dff_cond,axis=0), np.nanmean(dff_comp,axis=0), p_vec_, np.nanmean(dff_cond_,axis=0), np.nanmean(dff_comp_,axis=0)])[None,:],


def comp_pulse_stats_ref_v2(dff_, cond_trial, comp_trial, cond_trial_ref, comp_trial_ref, pre, post):
    trend_win = 30
    dff_ = dff_.squeeze()
    num_cond = len(cond_trial)
    num_comp = len(comp_trial)
    if cond_trial_ref is None:
        num_cond_ref = 0
    else:
        num_cond_ref = len(cond_trial_ref)
    if comp_trial_ref is None:
        num_comp_ref = 0
    else:
        num_comp_ref = len(comp_trial_ref)
    dff_cond = np.empty((num_cond, pre+post))
    dff_cond[:] = np.nan
    dff_comp = np.empty((num_comp, pre+post))
    dff_comp[:] = np.nan
    dff_cond_ref = np.empty((num_cond_ref, pre+post))
    dff_cond_ref[:] = np.nan
    dff_comp_ref = np.empty((num_comp_ref, pre+post))
    dff_comp_ref[:] = np.nan
    
    valid_ = np.zeros(num_cond).astype('bool')
    for n, (trial, trial_end) in enumerate(cond_trial):
        if len(dff_[trial-pre:trial+post])==(pre+post):
            if trial_end<0:
                dff_cond[n] = dff_[trial-pre:trial+post] # -dff_[trial-pre:trial].mean()
            else:
                dff_cond[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end] #-dff_[trial-pre:trial].mean()
            valid_[n] = True
    dff_cond = dff_cond[valid_]    
    
    valid_ = np.zeros(num_cond_ref).astype('bool')
    if num_cond_ref>0:
        for n, (trial, trial_end) in enumerate(cond_trial_ref):
            if len(dff_[trial-pre:trial+post])==(pre+post):
                if trial_end<0:
                    dff_cond_ref[n] = dff_[trial-pre:trial+post] #-dff_[trial-pre:trial].mean()
                else:
                    dff_cond_ref[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end] #-dff_[trial-pre:trial].mean()
                valid_[n] = True
        dff_cond_ref = dff_cond_ref[valid_]
    if valid_.sum()>0:
        trend = np.nanmean(dff_cond_ref, axis=0)
        trend = moving_average(trend, trend_win)[None, :]
        dff_cond_ = dff_cond - trend
    else:
        dff_cond_ = dff_cond
    
    valid_ = np.zeros(num_comp).astype('bool')
    for n, (trial, trial_end) in enumerate(comp_trial):
        if len(dff_[trial-pre:trial+post])==(pre+post):
            if trial_end<0:
                dff_comp[n] = dff_[trial-pre:trial+post] #-dff_[trial-pre:trial].mean()
            else:
                dff_comp[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end] #-dff_[trial-pre:trial].mean()
            valid_[n] = True
    dff_comp = dff_comp[valid_]
    
    valid_ = np.zeros(num_comp_ref).astype('bool')
    if num_comp_ref>0:
        for n, (trial, trial_end) in enumerate(comp_trial_ref):
            if len(dff_[trial-pre:trial+post])==(pre+post):
                if trial_end<0:
                    dff_comp_ref[n] = dff_[trial-pre:trial+post] #-dff_[trial-pre:trial].mean()
                else:
                    dff_comp_ref[n, :trial_end-trial+pre] = dff_[trial-pre:trial_end] #-dff_[trial-pre:trial].mean()
                valid_[n] = True
        dff_comp_ref = dff_comp_ref[valid_]
    if valid_.sum()>0:
        trend = np.nanmean(dff_comp_ref, axis=0)
        trend = moving_average(trend, trend_win)[None, :]
        dff_comp_ = dff_comp - trend
    else:
        dff_comp_ = dff_comp
        
    p_vec = np.zeros((3, pre+post))
    for n in range(pre+post):
        n_valid_cond = ~np.isnan(dff_cond[:, n])
        n_valid_comp = ~np.isnan(dff_comp[:, n])
        if (n_valid_cond.sum()>0) and (n_valid_comp.sum()>0):
            _, p_vec[0, n] = ranksums(dff_cond[n_valid_cond, n], dff_comp[n_valid_comp, n])
        else:
            p_vec[0, n] = 1
        try:
            _, p_vec[1, n] = wilcoxon(dff_cond[n_valid_cond, n])
        except:
            p_vec[1, n] = 1
        try:
            _, p_vec[2, n] = wilcoxon(dff_comp[n_valid_comp, n])
        except:
            p_vec[2, n] = 1
    
    p_vec_ = np.zeros((3, pre+post))
    for n in range(pre+post):
        n_valid_cond = ~np.isnan(dff_cond[:, n])
        n_valid_comp = ~np.isnan(dff_comp[:, n])
        if (n_valid_cond.sum()>0) and (n_valid_comp.sum()>0):
            _, p_vec_[0, n] = ranksums(dff_cond_[n_valid_cond, n], dff_comp_[n_valid_comp, n])
        else:
            p_vec_[0, n] = 1
        try:
            _, p_vec_[1, n] = wilcoxon(dff_cond_[n_valid_cond, n])
        except:
            p_vec_[1, n] = 1
        try:
            _, p_vec_[2, n] = wilcoxon(dff_comp_[n_valid_comp, n])
        except:
            p_vec_[2, n] = 1
        
    return np.array([p_vec, np.nanmean(dff_cond,axis=0), np.nanmean(dff_comp,axis=0), p_vec_, np.nanmean(dff_cond_,axis=0), np.nanmean(dff_comp_,axis=0)])[None,:],


def comp_pulse_stats_ref_chunks(dff, cond_trial=None, comp_trial=None, cond_trial_ref=None, comp_trial_ref=None, pre=None, post=None):
    num_cells = dff.shape[0]
    cell_stats = np.zeros((num_cells, 6)).astype('O')
    for n in range(num_cells):
        _ = comp_pulse_stats_ref_v2(dff[n], cond_trial=cond_trial, comp_trial=comp_trial, cond_trial_ref=cond_trial_ref, comp_trial_ref=comp_trial_ref, pre=pre, post=post)
        cell_stats[n]=_[0]
    return cell_stats


def comp_swim_stats(dff_, cond_trial, comp_trial, pre, post):
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


def comp_swim_stats_chunks(dff, cond_trial=None, comp_trial=None, pre=None, post=None):
    num_cells = dff.shape[0]
    cell_stats = np.zeros((num_cells, 3)).astype('O')
    for n in range(num_cells):
        _ = comp_swim_stats(dff[n], cond_trial=cond_trial, comp_trial=comp_trial, pre=pre, post=post)
        cell_stats[n]=_[0]
    return cell_stats