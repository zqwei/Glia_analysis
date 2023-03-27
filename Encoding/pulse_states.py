import numpy as np
import os, sys
import pandas as pd
from pulse_coding import utest_block
df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v6.csv')

def process_file(ind):
    from scipy.stats import spearmanr
    
    row = df.iloc[ind]
    save_root = row['save_dir']+'/'
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    trial_post = 35
    trial_pre = 5
    pre_ext = 0
    swim_thres = 30
    
    _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
    probe_amp=_['probe_amp']
    swim_t_frame=_['swim_t_frame']
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF']
    len_dff = dFF_.shape[1]
    epoch_frame=_['epoch_frame'][:len_dff]
    pulse_frame=_['pulse_frame'][:len_dff]
    visu_frame=_['visu_frame'][:len_dff]
    lswim_frame=_['lswim_frame'][:len_dff]
    rswim_frame=_['rswim_frame'][:len_dff]
    visu_frame_=_['visu_frame_'][:len_dff]

    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    dFF_ = dFF_[cell_in_brain]

    CL_idx = epoch_frame<=1
    rl = row['rl']
    rr = row['rr']
    if rl >= rr:
        swim_frame_ = lswim_frame
    else:
        swim_frame_ = rswim_frame
    pulse_frame_=pulse_frame.copy()
    pulse_frame_[epoch_frame%5<3]=0
    
     # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    len_ = len(epoch_on)-1
    dFF_trial = []
    swim_mask_trial = []
    CL_trial = []

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
        if (not CL_trial_) & (swim_frame_[on_:off_][epoch_%5==2].sum()>0):
            continue

        # remove CL passive trials
        if swim_frame_[on_:off_][epoch_%5==2].sum()==0:
            last_swm = np.where(swim_frame_[on_:off_][epoch_%5<2])[0][-1]
            last_swm = last_swm - (epoch_%5<2).sum()
        else:
            last_swm = 0
        if CL_trial_ & (last_swm<-10):
            continue

        # length of pause -- it seems like some of them are extremely long
        # remove the long pause trial
        if (not CL_trial_) & ((epoch_%5==2).sum()>10):
            continue
        if (CL_trial_) & ((epoch_%5==2).sum()<20):
            continue
        # print((epoch_%5==2).sum())
        if ((epoch_%5==2).sum()>100):
            continue

        # set probe on time
        catch_trial_ = pulse_.sum()==0
        # pause trial
        if catch_trial_:
            # probe_on_ = (epoch_<3).sum()
            continue
        else:
            probe_on_ = np.where(pulse_>0)[0][0]

        probe_on_ = probe_on_+on_

        swm_ = np.cumsum(swim_frame_[probe_on_-trial_pre:probe_on_+trial_post])
        if (swm_>0).sum()>swim_thres:
            continue

        # catch_trial.append(catch_trial_)
        CL_trial.append(CL_trial_)
        dFF_trial.append(dFF_[:, probe_on_-trial_pre:probe_on_+trial_post])
        swim_mask_trial.append(swm_<=0)
    
    dFF_ = None
    
    dFF_trial = np.array(dFF_trial)
    dFF_trial = dFF_trial.transpose([1, 2, 0])#.astype('float')
    dFF_trial_ = dFF_trial - dFF_trial[:, :trial_pre, :].mean(axis=1, keepdims=True)
    swim_mask_trial = np.array(swim_mask_trial).T
    # catch_trial = np.array(catch_trial)
    CL_trial = np.array(CL_trial)
    
    p_ = utest_block(dFF_trial[:, :, CL_trial], dFF_trial[:, :, ~CL_trial], swim_mask_trial[:, CL_trial], swim_mask_trial[:, ~CL_trial])
    
    np.save(save_root+'cell_state_pulse', p_)
    return None


if __name__ == "__main__":
    ind = int(sys.argv[1])
    row = df.iloc[ind]
    save_root = row['save_dir']+'/'
    process_file(ind)