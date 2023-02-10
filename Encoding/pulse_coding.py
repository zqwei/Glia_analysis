import numpy as np
import os, sys
import dask.array as da
import pandas as pd
df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v0.csv')


def spearmanr_block(a, b, axis=0):
    from scipy.stats import spearmanr
    corr_, pval_ = spearmanr(a, b, axis=axis)
    res = np.zeros((a.shape[0], 2))
    res[:, 0] = corr_[-1, :-1]
    res[:, 1] = pval_[-1, :-1]
    return res

def utest(x, y, mask_x, mask_y):
    from scipy.stats import mannwhitneyu, ks_2samp, spearmanr
    len_ = x.shape[0] # x shape: time x trials
    p_res = np.zeros(len_)
    for n in range(len_):
        if (mask_x[n].sum()>5) & (mask_y[n].sum()>5):
            try:
                _, p = mannwhitneyu(x[n, mask_x[n]], y[n, mask_y[n]])
            except:
                p = 1.
        else:
            p = 1.
        p_res[n] = p
    return p_res

def utest_block(x, y, mask_x, mask_y):
    from tqdm import tqdm
    num_cells = x.shape[0]
    num_time = x.shape[1]
    p_res = np.zeros((num_cells, num_time))
    for n_cell in tqdm(range(num_cells)):
        x_ = x[n_cell]
        y_ = y[n_cell]
        p_res[n_cell] = utest(x_, y_, mask_x, mask_y)
    return p_res

def process_n_file_data(ind, trial_post = 35, trial_pre = 5, 
                        pre_ext = 0, swim_thres = 30):
    import numpy.ma as ma
    from scipy.stats import spearmanr
    row = df.iloc[ind]
    save_root = row['save_dir']+'/'
    print(ind, save_root)
    _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
    probe_amp=_['probe_amp']
    swim_t_frame=_['swim_t_frame']
    
    # dFF_ = da.from_zarr(save_root+'cell_dff.zarr')
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF']
    trial_length = trial_post + trial_pre + pre_ext
    
    # epoch_frame=_['epoch_frame']
    # lswim_frame=_['lswim_frame']
    # rswim_frame=_['rswim_frame']
    # pulse_frame=_['pulse_frame']
    # visu_frame=_['visu_frame']
    # visu_frame_=_['visu_frame_']
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
     # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    epoch_on = np.where((epoch_frame[1:]==3) & (epoch_frame[:-1]==2))[0]+1
    epoch_off = np.where((epoch_frame[1:]!=3) & (epoch_frame[:-1]==3))[0]
    len_ = min(len(epoch_on), len(epoch_off))
    dFF_trial = []
    swim_mask_trial = []
    for n_ in range(len_):
        on_ = epoch_on[n_]-trial_pre
        off_ = epoch_on[n_]+trial_post
        vis_ = visu_frame[on_:off_]
        probe_ = pulse_frame[on_:off_]
        # if (vis_<0).mean()>0.1: # catch trial
        #     continue
        if (probe_==0).mean()<0.9: # non catch trial skip
            continue
        swm_ = np.cumsum(swim_frame_[on_-pre_ext:off_])
        if (swm_>0).sum()>swim_thres:
            continue
        if dFF_[:, on_-pre_ext:off_].shape[1] != trial_length:
            continue
        dFF_trial.append(dFF_[:, on_-pre_ext:off_])
        swim_mask_trial.append(swm_<=0)
    # CL_CT_dFF = da.stack(dFF_trial)
    CL_CT_dFF = np.array(dFF_trial)
    CL_CT_swim_mask = np.array(swim_mask_trial)
    dFF_trial = []
    swim_mask_trial = []
    for n_ in range(len_):
        on_ = epoch_on[n_]-trial_pre
        off_ = epoch_on[n_]+trial_post+10
        vis_ = visu_frame[on_:off_]
        probe_ = pulse_frame[on_:off_]
        # if (vis_<0).mean()<=0.1: # non catch trial
        #     continue
        if (probe_==0).mean()>0.9: # catch trial skip
            continue
        # probe_on_ = np.where(probe_==probe_amp)[0][0]+epoch_on[n_]-trial_pre # aligned to probe on
        probe_on_ = np.where(probe_>0)[0][0]+epoch_on[n_]-trial_pre # aligned to probe on
        on_ = probe_on_-trial_pre
        off_ = probe_on_+trial_post
        swm_ = np.cumsum(swim_frame_[on_-pre_ext:off_])
        if (swm_>0).sum()>swim_thres:
            continue
        if dFF_[:, on_-pre_ext:off_].shape[1] != trial_length:
            continue
        swim_mask_trial.append(swm_<=0)
        dFF_trial.append(dFF_[:, on_-pre_ext:off_])
    # CL_dFF = da.stack(dFF_trial)
    CL_dFF = np.array(dFF_trial)
    CL_swim_mask = np.array(swim_mask_trial)
    # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    epoch_on = np.where((epoch_frame[1:]==8) & (epoch_frame[:-1]==7))[0]+1
    epoch_off = np.where((epoch_frame[1:]!=8) & (epoch_frame[:-1]==8))[0]
    displacement_dFF_catch_trial = []
    displacement_dFF = []
    len_ = min(len(epoch_on), len(epoch_off))
    dFF_trial = []
    swim_mask_trial = []
    for n_ in range(len_):
        on_ = epoch_on[n_]-trial_pre
        off_ = epoch_on[n_]+trial_post
        vis_ = visu_frame[on_:off_]
        probe_ = pulse_frame[on_:off_]
        # if (vis_<0).mean()>0.1: # catch trial
        #     continue
        if (probe_==0).mean()<0.9: # non catch trial skip
            continue
        swm_ = np.cumsum(swim_frame_[on_-pre_ext:off_])
        if (swm_>0).sum()>swim_thres:
            continue
        if dFF_[:, on_-pre_ext:off_].shape[1] != trial_length:
            continue
        swim_mask_trial.append(swm_<=0)
        dFF_trial.append(dFF_[:, on_-pre_ext:off_])
    # OL_CT_dFF = da.stack(dFF_trial)
    OL_CT_dFF = np.array(dFF_trial)
    OL_CT_swim_mask = np.array(swim_mask_trial)
    dFF_trial = []
    swim_mask_trial = []
    for n_ in range(len_):
        on_ = epoch_on[n_]-trial_pre
        off_ = epoch_on[n_]+trial_post+10
        vis_ = visu_frame[on_:off_]
        probe_ = pulse_frame[on_:off_]
        # if (vis_<0).mean()<=0.1: # non catch trial
        #     continue
        if (probe_==0).mean()>0.9: # catch trial skip
            continue
        # probe_on_ = np.where(probe_==probe_amp)[0][0]+epoch_on[n_]-trial_pre # aligned to probe on
        probe_on_ = np.where(probe_>0)[0][0]+epoch_on[n_]-trial_pre # aligned to probe on
        on_ = probe_on_-trial_pre
        off_ = probe_on_+trial_post    
        swm_ = np.cumsum(swim_frame_[on_-pre_ext:off_])
        if (swm_>0).sum()>swim_thres:
            continue
        if dFF_[:, on_-pre_ext:off_].shape[1] != trial_length:
            continue
        swim_mask_trial.append(swm_<=0)
        dFF_trial.append(dFF_[:, on_-pre_ext:off_])
    # OL_dFF = da.stack(dFF_trial)
    OL_dFF = np.array(dFF_trial)
    OL_swim_mask = np.array(swim_mask_trial)
    
    CL_swim_mask = CL_swim_mask.T
    CL_CT_swim_mask = CL_CT_swim_mask.T
    OL_swim_mask = OL_swim_mask.T
    OL_CT_swim_mask = OL_CT_swim_mask.T
    
    # CL_dFF = CL_dFF.compute().transpose((1, 2, 0))
    # CL_CT_dFF = CL_CT_dFF.compute().transpose((1, 2, 0))
    # OL_dFF = OL_dFF.compute().transpose((1, 2, 0))
    # OL_CT_dFF = OL_CT_dFF.compute().transpose((1, 2, 0))
    
    CL_dFF = CL_dFF.transpose((1, 2, 0))#.astype('float')
    CL_CT_dFF = CL_CT_dFF.transpose((1, 2, 0))#.astype('float')
    OL_dFF = OL_dFF.transpose((1, 2, 0))#.astype('float')
    OL_CT_dFF = OL_CT_dFF.transpose((1, 2, 0))#.astype('float')
    
    CL_dFF_    = CL_dFF - CL_dFF[:, :trial_pre, :].mean(axis=1, keepdims=True)
    CL_CT_dFF_ = CL_CT_dFF - CL_CT_dFF[:, :trial_pre, :].mean(axis=1, keepdims=True)
    OL_dFF_    = OL_dFF - OL_dFF[:, :trial_pre, :].mean(axis=1, keepdims=True)
    OL_CT_dFF_ = OL_CT_dFF - OL_CT_dFF[:, :trial_pre, :].mean(axis=1, keepdims=True)

    mask_arr = ma.masked_array(CL_CT_dFF_.mean(axis=0), mask =~CL_CT_swim_mask)
    CL_dFF_no_CT    = CL_dFF_ - mask_arr.mean(axis=-1, keepdims=True)
    mask_arr = ma.masked_array(OL_CT_dFF_.mean(axis=0), mask =~OL_CT_swim_mask) 
    OL_dFF_no_CT    = OL_dFF_ - mask_arr.mean(axis=-1, keepdims=True)
    return (CL_swim_mask, CL_CT_swim_mask, OL_swim_mask, OL_CT_swim_mask, 
            CL_dFF, CL_CT_dFF, OL_dFF, OL_CT_dFF,
            CL_dFF_, CL_CT_dFF_, OL_dFF_, OL_CT_dFF_,
            CL_dFF_no_CT, OL_dFF_no_CT)

    
def process_n_file(ind):
    (CL_swim_mask, CL_CT_swim_mask, OL_swim_mask, OL_CT_swim_mask, 
     CL_dFF, CL_CT_dFF, OL_dFF, OL_CT_dFF,
     CL_dFF_, CL_CT_dFF_, OL_dFF_, OL_CT_dFF_,
     CL_dFF_no_CT, OL_dFF_no_CT) = process_n_file_data(ind, trial_post = 35, 
                                                       trial_pre = 5, pre_ext = 0, 
                                                       swim_thres = 30)
    
    if not os.path.exists(save_root+'cell_active_with_mean.npz'):
        CL_res = utest_block(CL_dFF, CL_CT_dFF, CL_swim_mask, CL_CT_swim_mask)
        OL_res = utest_block(OL_dFF, OL_CT_dFF, OL_swim_mask, OL_CT_swim_mask)
        np.savez(save_root+'cell_active_with_mean.npz', CL_res=CL_res, OL_res=OL_res)
        
    if not os.path.exists(save_root+'cell_active_with_mean.npz'):
        CL_res_ = utest_block(CL_dFF_, CL_CT_dFF_, CL_swim_mask, CL_CT_swim_mask)
        OL_res_ = utest_block(OL_dFF_, OL_CT_dFF_, OL_swim_mask, OL_CT_swim_mask)    
        np.savez(save_root+'cell_active.npz', CL_res_=CL_res_, OL_res_=OL_res_)
    
    baseline_condition_res_ = utest_block(CL_CT_dFF_, OL_CT_dFF_, CL_CT_swim_mask, OL_CT_swim_mask)
    pulse_condition_res_ = utest_block(CL_dFF_, OL_dFF_, CL_swim_mask, OL_swim_mask)
    baseline_condition_res = utest_block(CL_CT_dFF, OL_CT_dFF, CL_CT_swim_mask, OL_CT_swim_mask)
    pulse_condition_res = utest_block(CL_dFF, OL_dFF, CL_swim_mask, OL_swim_mask)
    np.savez(save_root+'cell_states.npz', 
             baseline_condition_res_=baseline_condition_res_, 
             pulse_condition_res_=pulse_condition_res_, 
             baseline_condition_res=baseline_condition_res, 
             pulse_condition_res=pulse_condition_res)
    
    pulse_condition_res_no_CT = utest_block(CL_dFF_no_CT, OL_dFF_no_CT, CL_swim_mask, OL_swim_mask)
    np.savez(save_root+'cell_states_no_CT.npz', pulse_condition_res_no_CT=pulse_condition_res_no_CT)


if __name__ == "__main__":
    ind = int(sys.argv[1])
    row = df.iloc[ind]
    save_root = row['save_dir']+'/'
    print(save_root)
    process_n_file(ind)
    # if not os.path.exists(save_root+'cell_active.npz'):
    #     process_n_file(ind)
    # if not os.path.exists(save_root+'cell_active_with_mean.npz'):
    #     process_n_file(ind)
    # if not os.path.exists(save_root+'cell_states.npz'):
    #     process_n_file(ind)
    # if not os.path.exists(save_root+'cell_states_no_CT.npz'):
    #     process_n_file(ind)