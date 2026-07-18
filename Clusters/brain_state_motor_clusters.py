'''
Build motor-related cell clusters: select motor/state-modulated cells per
fish, align their dFF to probe/recovery-swim trials, then cluster the
resulting dynamics (agglomerative + DBSCAN on cell location) and save the
per-fish cluster labels to fmotor_cluster_label.npz.
'''

import numpy as np
import pandas as pd
import numpy.ma as ma
from scipy.stats import zscore
from scipy.stats import spearmanr
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
sns.set(font_scale=1.5, style='ticks')
df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v8.csv')
cluster_folder = '/nrs/ahrens/Ziqiang/Jing_Glia_project/cell_spatial_clusters_af/'

cell_locs = []
CL_dFF_list = []
OL_dFF_list = []
CL_dFF_list_ = []
OL_dFF_list_ = []
swim_CL_dFF_list = []
swim_OL_dFF_list = []
swim_CL_dFF_list_ = []
swim_OL_dFF_list_ = []
animal_indx = []

for ind, row in df.iterrows():
    print(ind)
    save_root = row['save_dir']+'/'
    cells_center = np.load(save_root+'cell_center_registered.npy')
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    r_motor = np.load(save_root+'cell_motor_corr.npz')['r_cell']
    r_pulse = np.load(save_root+'cell_pulse_series_corr.npz')['r_cell']
    cell_idx = np.abs(r_motor)>0.4
    res = np.load(save_root+'cell_state_pulse.npy')
    sig_time = (res<0.05).sum(axis=1)
    cell_idx_ = sig_time>5
    res = np.load(save_root+'cell_states_no_CT.npz')['pulse_condition_res_no_CT']
    sig_time = (res<0.05).sum(axis=1)
    cell_idx_ = cell_idx_ | (sig_time>5)
    cell_idx = cell_idx & cell_idx_
    cells_center = cells_center[cell_in_brain][cell_idx]
    cell_locs.append(cells_center)
    np.save(save_root + 'cell_state_motor_filtered.npy', cell_idx)
    
    trial_post = 35
    trial_pre = 5
    pre_ext = 0
    swim_thres = 30
    _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
    probe_amp=_['probe_amp']
    swim_t_frame=_['swim_t_frame']
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain][cell_idx]
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
    _tmp_ = np.zeros((dFF_.shape[0], trial_pre+trial_post))
    _tmp_[:] = np.nan # this is used to marked the trial where recovery swim not happen
    recovery_swim_thres = np.percentile(swim_frame_[swim_frame_>0], 10)
    
    # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    len_ = len(epoch_on)-1
    dFF_trial = []
    dFF_trial_ = []
    dFF_swim = []
    dFF_swim_ = []
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
            # probe_on_ = (epoch_%5<3).sum()
            continue
        else:
            probe_on_ = np.where(pulse_>0)[0][0]

        probe_on_ = probe_on_+on_

        swm_ = np.cumsum(swim_frame_[probe_on_-trial_pre:probe_on_+trial_post])
        if (swm_>0).sum()>swim_thres:
            continue

        # catch_trial.append(catch_trial_)
        off_t = (epoch_%5<1).sum()+on_
        CL_trial.append(CL_trial_)
        dFF_trial.append(dFF_[:, probe_on_-trial_pre:probe_on_+trial_post])
        # dFF_trial_.append(dFF_[:, probe_on_-trial_pre:probe_on_+trial_post] - dFF_[:, probe_on_-trial_pre:probe_on_].mean(axis=-1, keepdims=True))
        dFF_trial_.append(dFF_[:, probe_on_-trial_pre:probe_on_+trial_post] - dFF_[:, off_t-trial_pre:off_t].mean(axis=-1, keepdims=True))
        swim_mask_trial.append(swm_<=0)
        _recovery_swim = swim_frame_[on_:off_][epoch_%5==3]>recovery_swim_thres #>0.01
        if _recovery_swim.sum()>0:
            _t_ = np.where(_recovery_swim)[0][0]
            _t_ = _t_ + (epoch_%5<=2).sum()+on_
            dFF_swim.append(dFF_[:, _t_-trial_post:_t_+trial_pre])
            # dFF_swim_.append(dFF_[:, _t_-trial_post:_t_+trial_pre] - dFF_[:, probe_on_-trial_pre:probe_on_].mean(axis=-1, keepdims=True))
            dFF_swim_.append(dFF_[:, _t_-trial_post:_t_+trial_pre] - dFF_[:, off_t-trial_pre:off_t].mean(axis=-1, keepdims=True))
        else:
            dFF_swim.append(_tmp_)
            dFF_swim_.append(_tmp_)
        
    dFF_trial = np.array(dFF_trial).transpose([1, 2, 0])
    # dFF_trial_ = dFF_trial - dFF_trial[:, :trial_pre, :].mean(axis=1, keepdims=True)
    dFF_trial_ = np.array(dFF_trial_).transpose([1, 2, 0])
    dFF_swim = np.array(dFF_swim).transpose([1, 2, 0])
    dFF_swim_ = np.array(dFF_swim_).transpose([1, 2, 0])
    # dFF_swim = dFF_swim.transpose([1, 2, 0])
    # dFF_swim_ = dFF_swim - dFF_trial[:, :trial_pre, :].mean(axis=1, keepdims=True)
    swim_mask_trial = np.array(swim_mask_trial).T
    # catch_trial = np.array(catch_trial)
    CL_trial = np.array(CL_trial)
    
    dat_ = dFF_trial[:, :, CL_trial]
    swim_mask_ = swim_mask_trial[:, CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    CL_dFF_list.append(mask_arr.mean(axis=-1))
    dat_ = dFF_swim[:, :, CL_trial]
    swim_CL_dFF_list.append(np.nanmean(dat_, axis=-1))

    dat_ = dFF_trial[:, :, ~CL_trial]
    swim_mask_ = swim_mask_trial[:, ~CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    OL_dFF_list.append(mask_arr.mean(axis=-1))
    dat_ = dFF_swim[:, :, ~CL_trial]
    swim_OL_dFF_list.append(np.nanmean(dat_, axis=-1))
    
    
    dat_ = dFF_trial_[:, :, CL_trial]
    swim_mask_ = swim_mask_trial[:, CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    CL_dFF_list_.append(mask_arr.mean(axis=-1))
    dat_ = dFF_swim_[:, :, CL_trial]
    swim_CL_dFF_list_.append(np.nanmean(dat_, axis=-1))


    dat_ = dFF_trial_[:, :, ~CL_trial]
    swim_mask_ = swim_mask_trial[:, ~CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    OL_dFF_list_.append(mask_arr.mean(axis=-1))
    dat_ = dFF_swim_[:, :, ~CL_trial]
    swim_OL_dFF_list_.append(np.nanmean(dat_, axis=-1))
    
    animal_indx.append([ind]*dFF_trial.shape[0])

CL_dFF_list = np.concatenate(CL_dFF_list, axis=0)
OL_dFF_list = np.concatenate(OL_dFF_list, axis=0)
CL_dFF_list_ = np.concatenate(CL_dFF_list_, axis=0)
OL_dFF_list_ = np.concatenate(OL_dFF_list_, axis=0)
swim_CL_dFF_list = np.concatenate(swim_CL_dFF_list, axis=0)
swim_OL_dFF_list = np.concatenate(swim_OL_dFF_list, axis=0)
swim_CL_dFF_list_ = np.concatenate(swim_CL_dFF_list_, axis=0)
swim_OL_dFF_list_ = np.concatenate(swim_OL_dFF_list_, axis=0)
cell_locs = np.concatenate(cell_locs)
animal_indx_ = np.concatenate(animal_indx)

## saving collected data
np.savez(cluster_folder+'cell_list_dynamics_early_motor.npz', \
         CL_dFF_list=CL_dFF_list, \
         OL_dFF_list=OL_dFF_list, \
         CL_dFF_list_=CL_dFF_list_, \
         OL_dFF_list_=OL_dFF_list_, \
         cell_locs=cell_locs, \
         animal_indx_=animal_indx_)

## Cell dynamics based clustering
trial_all = np.concatenate([CL_dFF_list_, OL_dFF_list_], axis=1)
idxx = animal_indx_<10
dat_ = trial_all[idxx].data
# dat_ = trial_all.data
zdat_ =  zscore(dat_, axis=-1)

############### For fish
n_animal = 4
func_labels = AgglomerativeClustering(n_clusters=30).fit(zdat_[animal_indx_==n_animal]).labels_
func_labels_ = func_labels.copy()
c_thres = 0.7
c_ave_thres = 0.5

for label_ in range(30):
    mat_ = zdat_[animal_indx_==n_animal][func_labels==label_]
    c_, _ = spearmanr(mat_.T)
    np.fill_diagonal(c_, -1)
    if ((c_.max(axis=-1)>c_thres).sum())<50:
        func_labels_[func_labels==label_] = -1
    else:
        ave_ = mat_[c_.max(axis=-1)>c_thres].mean(axis=0)
        c_ave = spearmanr(mat_.T, ave_)[0][-1, :-1]
        where_ = np.where(func_labels==label_)[0][(c_.max(axis=-1)<c_thres) & (c_ave<c_ave_thres)]
        func_labels_[where_]=-1

idx_ = animal_indx_==n_animal
label_sub = np.zeros((len(func_labels_), 2)).astype('int') - 1
cluster_idx = 0
cluster_sub_idx = 0

for label_ in range(30):
    if (func_labels_==label_).sum()==0:
        continue
    clustering = DBSCAN(eps=100, min_samples=10).fit(cell_locs[idx_, 1:][func_labels_==label_])
    slabels_ = clustering.labels_
    nlabels_ = slabels_.max().astype('int')+1
    where_ = np.where(func_labels_==label_)[0]
    if (slabels_>=0).sum()==0: # remove spatially scattered clusters
        continue
    slabels_[slabels_>=0] = slabels_[slabels_>=0]+cluster_sub_idx
    label_sub[where_, 0] = cluster_idx
    label_sub[where_, 1] = slabels_
    cluster_idx = cluster_idx+1
    cluster_sub_idx = cluster_sub_idx+nlabels_+1

row = df.iloc[n_animal]
save_root = row['save_dir']+'/'
np.savez(save_root + 'fmotor_cluster_label.npz', label_sub=label_sub, cell_idx=cell_idx)
print(save_root)