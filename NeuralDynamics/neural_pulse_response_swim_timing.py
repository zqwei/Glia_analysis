'''
For each fish, find CL trials with a recovery swim and correlate each
cell's pre-swim activity (in 6-frame bins going back from the recovery
swim) against how far in time that bin is from the swim onset, saving
the per-cell correlation to cell_corr_recovery_swim_time.npz.
'''

import numpy as np
import os, sys
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from scipy.stats import zscore, spearmanr
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v8.csv')

for n_fish, row in df.iterrows():
    # row = df.iloc[n_fish]
    save_root = row['save_dir']+'/'
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain]
    _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
    probe_amp=_['probe_amp']
    swim_t_frame=_['swim_t_frame']
    len_dff = dFF_.shape[1]
    epoch_frame=_['epoch_frame'][:len_dff]
    pulse_frame=_['pulse_frame'][:len_dff]
    visu_frame=_['visu_frame'][:len_dff]
    lswim_frame=_['lswim_frame'][:len_dff]
    rswim_frame=_['rswim_frame'][:len_dff]
    visu_frame_=_['visu_frame_'][:len_dff]
    CL_idx = epoch_frame<=1
    rl = row['rl']
    rr = row['rr']
    if rl >= rr:
        swim_frame_ = lswim_frame
    else:
        swim_frame_ = rswim_frame
    pulse_frame_=pulse_frame.copy()
    pulse_frame_[epoch_frame%5<3]=0
    recovery_swim_thres = np.percentile(swim_frame_[swim_frame_>0], 10)
    
    # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    len_ = len(epoch_on)-1
    
    CL_trial = []
    time_ticks = []
    
    # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    len_ = len(epoch_on)-1
    recovery_swim_thres = np.percentile(swim_frame_[swim_frame_>0], 0)
    for n_ in range(len_):
        on_ = epoch_on[n_]
        off_ = epoch_on[n_+1]-1
        # check if this is a complete trial
        # skip incomplete trials
        if len(np.unique(epoch_frame[on_:off_]))<5:
            continue
        epoch_ = epoch_frame[on_:off_]
        # remove the trials without swim during evoke
        if swim_frame_[on_:off_][epoch_%5==1].sum()==0:
            continue
        swm_ = swim_frame_[on_:off_]
        pulse_ = pulse_frame_[on_:off_]
        CL_trial_ = epoch_[10]//5==0
    
        
        # remove OL active trials -- fish swim during pause
        if (not CL_trial_) & (swm_[epoch_%5==2].sum()>0):
            continue
    
        # remove CL passive trials
        if swm_[epoch_%5==2].sum()==0:
            last_swm = np.where(swm_[epoch_%5<2])[0][-1]
            last_swm = last_swm - (epoch_%5<2).sum()
        else:
            last_swm = 0
        # if CL_trial_ & (last_swm<-3):
        #     continue
        
        # length of pause -- it seems like some of them are extremely long
        # remove the long pause trial
        if (not CL_trial_) & ((epoch_%5==2).sum()>10):
            continue
        # if (CL_trial_) & ((epoch_%5==2).sum()<20):
        #     continue
        # if ((epoch_%5==2).sum()>100):
        #     continue
    
        # set probe on time
        catch_trial_ = pulse_.sum()==0
        # pause trial
        if catch_trial_:
            continue
        else:
            probe_on_ = np.where(pulse_>0)[0][0]
            probe_off_ = np.where(pulse_>0)[0][-1]
    
        # probe_on_ = probe_on_+on_
        # print(probe_on_)
    
        # swm_ = np.cumsum(swim_frame_[probe_on_-trial_pre:probe_on_+trial_post])
        # if (swm_>0).sum()>swim_thres:
        #     continue
    
        # catch_trial.append(catch_trial_)
        # print(swim_frame_[on_:off_][epoch_%5<2].sum())
        
        evoke_on_ = (epoch_%5==0).sum()
        _recovery_swim = swm_[epoch_%5==3]>recovery_swim_thres
        if _recovery_swim.sum()>0:
            _t_ = np.where(_recovery_swim)[0][0]
            _t_ = _t_ + (epoch_%5<=2).sum()
            probe_on_switch = np.where((pulse_[:-1]==0) & (pulse_[1:]>0))[0]
            if (probe_on_switch<_t_).sum()==0:
                continue
            last_epoch_on = probe_on_switch[probe_on_switch<_t_].max()
            if last_epoch_on<36:
                continue
            CL_trial.append(CL_trial_)
            time_ticks.append([on_+last_epoch_on, on_+_t_])
    CL_trial = np.array(CL_trial)
    time_ticks = np.array(time_ticks)
    resp = []
    dat_ = []
    
    for last_epoch_on, _t_ in time_ticks:
        for n in range(7):
            _time_ = last_epoch_on-36+n*6
            _dat_ = dFF_[:, _time_:_time_+6].sum(axis=1)-dFF_[:, _time_+1]*6
            dat_.append(_dat_)
            resp.append((_t_ - _time_)/3)
    
    dat_ = np.array(dat_).T
    resp = np.array(resp)[None, :]
    num_cells = dat_.shape[0]
    split_array = np.array_split(np.arange(num_cells).astype('int'), num_cells//1000)
    r_list = []
    p_list = []
    for n_split in tqdm(split_array):
        r, p = spearmanr(dat_[n_split], resp, axis=1)
        r_list.append(r[-1][:-1])
        p_list.append(p[-1][:-1])
    r_list = np.concatenate(r_list)
    p_list = np.concatenate(p_list)
    np.savez(save_root+'cell_corr_recovery_swim_time.npz', r_list=r_list, p_list=p_list)