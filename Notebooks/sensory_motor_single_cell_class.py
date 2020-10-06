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
    
    valid_ = np.zeros(num_pulse).astype('bool')
    for n, trial in enumerate(pulse_trial):
        if len(dff_[trial-2:trial+5])==7:
            dff_pulse[n] = dff_[trial-2:trial+5] - dff_[trial-2:trial].mean()
            valid_[n] = True
    dff_pulse = dff_pulse[valid_]
    
    valid_ = np.zeros(num_nopulse).astype('bool')
    for n, trial in enumerate(nopulse_trial):
        if len(dff_[trial+3:trial+10])==7:
            dff_nopulse[n] = dff_[trial+3:trial+10] - dff_[trial+3:trial+5].mean() 
            valid_[n] = True
    dff_nopulse = dff_nopulse[valid_]
    
    p_mean = np.zeros(3)
    _, p_mean[0] = ranksums(dff_pulse.sum(axis=-1), dff_nopulse.sum(axis=-1))
    try:
        _, p_mean[1] = wilcoxon(dff_pulse.sum(axis=-1))
    except:
        p_mean[1] = 1
    try:
        _, p_mean[2] = wilcoxon(dff_nopulse.sum(axis=-1))
    except:
        p_mean[2] = 1
        
    p_vec = np.zeros((3, 7))
    for n in range(7):
        _, p_vec[0, n] = ranksums(dff_pulse[:, n], dff_nopulse[:, n])
        try:
            _, p_vec[1, n] = wilcoxon(dff_pulse[:, n])
        except:
            p_vec[1, n] = 1
        try:
            _, p_vec[2, n] = wilcoxon(dff_nopulse[:, n])
        except:
            p_vec[2, n] = 1
    
    if (p_mean[0]<0.05) or ((p_vec[0]<0.05).sum()>3):
        x_ = np.vstack([dff_pulse, dff_nopulse])
        y_ = np.r_[np.zeros(dff_pulse.shape[0]), np.ones(dff_nopulse.shape[0])]
        try:
            p_manova = MANOVA(x_, y_).mv_test().results['x0']['stat']
        except:
            p_manova = None
    else:
        p_manova = None
    
    return np.array([p_mean, p_vec, p_manova, dff_pulse.mean(axis=0), dff_nopulse.mean(axis=0)])[None,:],


def multi_pulse_stats(dff_, pulse_trial, nopulse_trial, t_pre, t_post):
    dff_ = dff_.squeeze()
    num_pulse = len(pulse_trial)
    num_nopulse = len(nopulse_trial)
    dff_pulse = np.zeros((num_pulse, t_pre+t_post))
    dff_nopulse = np.zeros((num_nopulse, t_pre+t_post))
    trial_len_ = t_pre+t_post
    
    valid_ = np.zeros(num_pulse).astype('bool')
    for n, trial in enumerate(pulse_trial):
        if len(dff_[trial-t_pre:trial+t_post])==trial_len_:
            dff_pulse[n] = dff_[trial-t_pre:trial+t_post] - dff_[trial-t_pre:trial].mean()
            valid_[n] = True
    dff_pulse = dff_pulse[valid_]
    if valid_.sum()==0:
        return np.array([None, None, None, None, None])[None,:],
    
    valid_ = np.zeros(num_nopulse).astype('bool')
    for n, trial in enumerate(nopulse_trial):
        if len(dff_[trial-t_pre:trial+t_post])==trial_len_:
            dff_nopulse[n] = dff_[trial-t_pre:trial+t_post] - dff_[trial-t_pre:trial].mean() 
            valid_[n] = True
    dff_nopulse = dff_nopulse[valid_]
    if valid_.sum()==0:
        return np.array([None, None, None, None, None])[None,:],
    
    p_mean = np.zeros(3)
    _, p_mean[0] = ranksums(dff_pulse.sum(axis=-1), dff_nopulse.sum(axis=-1))
    try:
        _, p_mean[1] = wilcoxon(dff_pulse.sum(axis=-1))
    except:
        p_mean[1] = 1
    try:
        _, p_mean[2] = wilcoxon(dff_nopulse.sum(axis=-1))
    except:
        p_mean[2] = 1 
    
    p_vec = np.zeros((3, trial_len_))
    for n in range(trial_len_):
        _, p_vec[0, n] = ranksums(dff_pulse[:, n], dff_nopulse[:, n])
        try:
            _, p_vec[1, n] = wilcoxon(dff_pulse[:, n])
        except:
            p_vec[1, n] = 1
        try:
            _, p_vec[2, n] = wilcoxon(dff_nopulse[:, n])
        except:
            p_vec[2, n] = 1
    
    if (p_mean[0]<0.05) or ((p_vec[0]<0.05).sum()>3):
        x_ = np.vstack([dff_pulse, dff_nopulse])
        y_ = np.r_[np.zeros(dff_pulse.shape[0]), np.ones(dff_nopulse.shape[0])]
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
    
    valid_ = np.zeros(num_swim).astype('bool')
    for n, trial in enumerate(swim_trial):
        if len(dff_[trial-pre_len:trial+swim_len])==(pre_len+swim_len):
            dff_swim[n] = dff_[trial-pre_len:trial+swim_len] - dff_[(trial-pre_len):trial].mean()
            valid_[n] = True
    dff_swim = dff_swim[valid_]
    
    valid_ = np.zeros(num_noswim).astype('bool')
    for n, trial in enumerate(noswim_trial):
        if len(dff_[trial:trial+pre_len+swim_len])==(pre_len+swim_len):
            dff_noswim[n] = dff_[trial:trial+pre_len+swim_len] - dff_[(trial):(trial+pre_len)].mean()
            valid_[n] = True
    dff_noswim = dff_noswim[valid_]
    
    p_mean = np.zeros(3)
    _, p_mean[0] = ranksums(dff_swim.sum(axis=-1), dff_noswim.sum(axis=-1))
    try:
        _, p_mean[1] = wilcoxon(dff_swim.sum(axis=-1))
    except:
        p_mean[1] = 1
    try:
        _, p_mean[2] = wilcoxon(dff_noswim.sum(axis=-1))
    except:
        p_mean[2] = 1
    
    p_vec = np.zeros((3, swim_len+pre_len))
    for n in range(swim_len+pre_len):
        _, p_vec[0, n] = ranksums(dff_swim[:, n], dff_noswim[:, n])
        try:
            _, p_vec[1, n] = wilcoxon(dff_swim[:, n])
        except:
            p_vec[1, n] = 1
        try:
            _, p_vec[2, n] = wilcoxon(dff_noswim[:, n])
        except:
            p_vec[2, n] = 1
    
    p_manova = None
    
    if (p_mean[0]<0.05) or ((p_vec[0]<0.05).sum()>(swim_len+pre_len)//2+1):
        x_ = np.vstack([dff_swim, dff_noswim])
        y_ = np.r_[np.zeros(dff_swim.shape[0]), np.ones(dff_noswim.shape[0])]
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


def comp_stats(dff_, cond_trial, comp_trial, pre, post):
    dff_ = dff_.squeeze()
    num_cond = len(cond_trial)
    num_comp = len(comp_trial)
    dff_cond = np.zeros((num_cond, pre+post))
    dff_comp = np.zeros((num_comp, pre+post))
    
    valid_ = np.zeros(num_cond).astype('bool')
    for n, trial in enumerate(cond_trial):
        if len(dff_[trial-pre:trial+post])==(pre+post):
            dff_cond[n] = dff_[trial-pre:trial+post] - dff_[trial-pre:trial].mean()
            valid_[n] = True
    dff_cond = dff_cond[valid_]
    
    valid_ = np.zeros(num_comp).astype('bool')
    for n, trial in enumerate(comp_trial):
        if len(dff_[trial-pre:trial+post])==(pre+post):
            dff_comp[n] = dff_[trial-pre:trial+post] - dff_[trial-pre:trial].mean()
            valid_[n] = True
    dff_comp = dff_comp[valid_]
    
    p_mean = np.zeros(3)
    _, p_mean[0] = ranksums(dff_cond.sum(axis=-1), dff_comp.sum(axis=-1))
    _, p_mean[1] = wilcoxon(dff_cond.sum(axis=-1))
    _, p_mean[2] = wilcoxon(dff_comp.sum(axis=-1))
    
    p_vec = np.zeros((3, pre+post))
    for n in range(pre+post):
        _, p_vec[0, n] = ranksums(dff_cond[:, n], dff_comp[:, n])
        _, p_vec[1, n] = wilcoxon(dff_cond[:, n])
        _, p_vec[2, n] = wilcoxon(dff_comp[:, n])
    
    p_manova = None
    
    if (p_mean[0]<0.05) or ((p_vec[0]<0.05).sum()>(pre+post)//2+1):
        x_ = np.vstack([dff_cond, dff_comp])
        y_ = np.r_[np.zeros(dff_cond.shape[0]), np.ones(dff_comp.shape[0])]
        try:
            p_manova = MANOVA(x_, y_).mv_test().results['x0']['stat']
        except:
            p_manova = None
    else:
        p_manova = None
    
    return np.array([p_mean, p_vec, p_manova, dff_cond.mean(axis=0), dff_comp.mean(axis=0)])[None,:],


def open_ephys_metadata(xml):
    import xml.etree.ElementTree as et
    import collections
    import pandas as pd
    def tryfloat (x):
        try: return float(x)
        except: return(x)
    tree = et.parse(xml)
    root = tree.getroot()
    StimConds = []
    for r in root.getchildren():
        StimCond = collections.OrderedDict()
        for e in r:
            StimCond[e.tag] = (tryfloat(e.text))
        StimConds.append(StimCond)
    columns = list(StimConds[0].keys())
    columns.remove('epoch')
    index = [s['epoch'] for s in StimConds]
    return pd.DataFrame(StimConds, index=index, columns=columns)


def comp_stats_chunks(dff, cond_trial=None, comp_trial=None, pre=None, post=None):
    num_cells = dff.shape[0]
    cell_stats = np.zeros((num_cells, 5)).astype('O')
    for n in range(num_cells):
        _ = comp_stats(dff[n], cond_trial=cond_trial, comp_trial=comp_trial, pre=pre, post=post)
        cell_stats[n]=_[0]
    return cell_stats


def motor_stats_chunks(dff, swim_trial=None, noswim_trial=None, swim_len=None, pre_len=None):
    num_cells = dff.shape[0]
    cell_stats = np.zeros((num_cells, 5)).astype('O')
    for n in range(num_cells):
        _ = motor_stats(dff[n], swim_trial=swim_trial, noswim_trial=noswim_trial, swim_len=swim_len, pre_len=pre_len)
        cell_stats[n]=_[0]
    return cell_stats


def pulse_stats_chunks(dff, pulse_trial=None, nopulse_trial=None):
    num_cells = dff.shape[0]
    cell_stats = np.zeros((num_cells, 5)).astype('O')
    for n in range(num_cells):
        _ = pulse_stats(dff[n], pulse_trial=pulse_trial, nopulse_trial=nopulse_trial)
        cell_stats[n]=_[0]
    return cell_stats


def multi_pulse_stats_chunks(dff, pulse_trial=None, nopulse_trial=None, t_pre=None, t_post=None):
    num_cells = dff.shape[0]
    cell_stats = np.zeros((num_cells, 5)).astype('O')
    for n in range(num_cells):
        _ = multi_pulse_stats(dff[n], pulse_trial=pulse_trial, nopulse_trial=nopulse_trial, t_pre=t_pre, t_post=t_post)
        cell_stats[n]=_[0]
    return cell_stats