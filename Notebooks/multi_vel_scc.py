import numpy as np


def kruskal_(x, ind):
    from scipy.stats import kruskal
    u, index_ = np.unique(ind, return_inverse=True)
    len_ = len(u)
    dat_ = []
    for ii in range(len_):
        dat_.append(x[index_==ii])
    return kruskal(*dat_)


def mean_(x, ind, axis=0):
    u, index_ = np.unique(ind, return_inverse=True)
    len_ = len(u)
    dat_ = []
    for ii in range(len_):
        dat_.append(x[index_==ii].mean(axis=axis))
    return np.array(dat_)


def comp_pulse_stats(dff_, trials, conds, pre, post):
    dff_ = dff_.squeeze()
    num_ = len(trials)
    dff_cond = np.empty((num_, pre+post))
    dff_cond[:] = np.nan
    
    valid_ = np.zeros(num_).astype('bool')
    for n, trial in enumerate(trials):
        if len(dff_[trial-pre:trial+post])==(pre+post):
            dff_cond[n] = dff_[trial-pre:trial+post]
            valid_[n] = True
    dff_cond = dff_cond[valid_]
    conds = conds[valid_]
    
    p_vec = np.zeros((1, pre+post))
    for n in range(pre+post):
        _, p_vec[0, n] = kruskal_(dff_cond, conds)
        
    return np.array([p_vec, mean_(dff_cond, conds, axis=0)])[None,:],


def comp_pulse_stats_chunks(dff, trials=None, conds=None, pre=None, post=None):
    num_cells = dff.shape[0]
    cell_stats = np.zeros((num_cells, 2)).astype('O')
    for n in range(num_cells):
        _ = comp_pulse_stats(dff[n], trials=trials, conds=conds, pre=pre, post=post)
        cell_stats[n]=_[0]
    return cell_stats


