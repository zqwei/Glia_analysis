import numpy as np
import numpy.ma as ma
import os, sys
import dask.array as da
import pandas as pd
# df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v6.csv')
df = pd.read_csv('../Datalists/data_list_in_analysis_pulse_cells_v2.csv')


def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


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

def process_n_file_data(ind, trial_post = 40, pre_ext = 0):
    from scipy.stats import spearmanr
    row = df.iloc[ind]
    save_root = row['save_dir']+'/'
    print(ind, save_root)
    _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
    probe_amp=_['probe_amp']
    swim_t_frame=_['swim_t_frame']
    
    # dFF_ = da.from_zarr(save_root+'cell_dff.zarr')
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
    rl, _ = spearmanr(lswim_frame[CL_idx], visu_frame[CL_idx])
    rr, _ = spearmanr(rswim_frame[CL_idx], visu_frame[CL_idx])
    if rl >= rr:
        swim_frame_ = lswim_frame
    else:
        swim_frame_ = rswim_frame
        
    visu_frame_ = visu_frame.copy()
    visu_frame_[np.abs(visu_frame_)<1e-5] = 0

    # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    evoke_on = np.where((epoch_frame[1:]==1) & (epoch_frame[:-1]==0))[0]+1
    epoch_on = np.where((epoch_frame[1:]==2) & (epoch_frame[:-1]==1))[0]+1
    epoch_off = np.where((epoch_frame[1:]!=2) & (epoch_frame[:-1]==2))[0]
    len_ = min(len(epoch_on), len(epoch_off))
    
    dFF_trial = []
    swim_mask_trial = []
    for n_ in range(len_):
        if swim_frame_[evoke_on[n_]:epoch_on[n_]].sum()==0: #no swim trials during evoke
            continue
        epoch_off_ = epoch_off[n_] # pause off frame
        if epoch_off_+3>len(epoch_frame):
            continue
        if epoch_frame[epoch_off_+3]!=3:
            continue
        # find end point of visual inputs
        on_ = epoch_on[n_]
        # last swim in pause epoch
        if swim_frame_[on_:epoch_off_].sum()>0:
            last_swim_ = np.where(swim_frame_[on_:epoch_off_]>0)[0][-1]+1
        else:
            last_swim_ = 1
        if last_swim_ >= (epoch_off_-on_)-3:
            continue
        # remove probe trials
        if (visu_frame_[epoch_off_:epoch_off_+30]<0).sum()>0: 
            continue
        # last forward grating
        if (visu_frame_[on_:epoch_off_]<0).sum()>0:
            last_vis_ = np.where(visu_frame_[on_:epoch_off_]<0)[0][-1]+1
        else:
            last_vis_ = 1
        on_ = on_ + max(last_swim_, last_vis_)
        off_ = on_+trial_post
        swm_ = np.cumsum(swim_frame_[on_-pre_ext:off_])
        swim_mask_trial.append(swm_<=0)
        dFF_trial.append(dFF_[:, on_-pre_ext:off_])
    CL_CT_dFF = da.stack(dFF_trial)
    CL_CT_swim_mask = np.array(swim_mask_trial)

    # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    evoke_on = np.where((epoch_frame[1:]==6) & (epoch_frame[:-1]==5))[0]+1
    epoch_on = np.where((epoch_frame[1:]==7) & (epoch_frame[:-1]==6))[0]+1
    epoch_off = np.where((epoch_frame[1:]!=7) & (epoch_frame[:-1]==7))[0]
    len_ = min(len(epoch_on), len(epoch_off))

    dFF_trial = []
    swim_mask_trial = []
    for n_ in range(len_):
        if swim_frame_[evoke_on[n_]:epoch_on[n_]].sum()==0: #no swim trials during evoke
            continue
        epoch_off_ = epoch_off[n_] # pause off frame
        if epoch_off_+3>len(epoch_frame):
            continue
        if epoch_frame[epoch_off_+3]!=8:
            continue
        # find end point of visual inputs
        on_ = epoch_on[n_]
        # last swim in pause epoch
        if swim_frame_[on_:epoch_off_].sum()>0:
            last_swim_ = np.where(swim_frame_[on_:epoch_off_]>0)[0][-1]+1
        else:
            last_swim_ = 1
        if last_swim_ >= (epoch_off_-on_)-3:
            continue
        # remove probe trials
        if (visu_frame_[epoch_off_:epoch_off_+30]<0).sum()>0: 
            continue
        # last forward grating
        if (visu_frame_[on_:epoch_off_]<0).sum()>0:
            last_vis_ = np.where(visu_frame_[on_:epoch_off_]<0)[0][-1]+1
        else:
            last_vis_ = 1
        on_ = on_ + max(last_swim_, last_vis_)
        off_ = on_+trial_post
        swm_ = np.cumsum(swim_frame_[on_-pre_ext:off_])
        swim_mask_trial.append(swm_<=0)
        dFF_trial.append(dFF_[:, on_-pre_ext:off_])
    OL_CT_dFF = da.stack(dFF_trial)
    OL_CT_swim_mask = np.array(swim_mask_trial)
    
    CL_CT_swim_mask = CL_CT_swim_mask.T
    OL_CT_swim_mask = OL_CT_swim_mask.T

    CL_CT_dFF = CL_CT_dFF.compute().transpose((1, 2, 0))
    OL_CT_dFF = OL_CT_dFF.compute().transpose((1, 2, 0))        
        
    return (CL_CT_swim_mask, OL_CT_swim_mask, CL_CT_dFF, OL_CT_dFF)

    
def process_n_file(ind):
    (CL_CT_swim_mask, OL_CT_swim_mask, 
     CL_CT_dFF, OL_CT_dFF) = process_n_file_data(ind, trial_post = 40, pre_ext = 0)    
    baseline_ = utest_block(CL_CT_dFF, OL_CT_dFF, CL_CT_swim_mask, OL_CT_swim_mask)
    np.savez(save_root+'cell_states_baseline.npz', baseline_=baseline_)


if __name__ == "__main__":
    ind = int(sys.argv[1])
    row = df.iloc[ind]
    save_root = row['save_dir']+'/'
    if not os.path.exists(save_root+'cell_states_baseline.npz'):
        process_n_file(ind)