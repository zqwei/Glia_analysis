'''
Generate the processed data for Figure_4cd_neural_prediction_swim.ipynb.

Fig 4c: whole-brain map of cells whose activity anti-correlates with
recovery-swim-onset timing (r<0, p<0.05), reduced to xy/xz percentile
projections. Ported from
Clusters/depreciated/Figure_5_swim_predictor/neural_prediction_swim.ipynb.
Uses each fish's cell_corr_recovery_swim_time.npz (r_list/p_list against
swim-onset timing), which is already cached at every fish's save_dir --
the expensive per-cell trial-parsing + Spearman correlation step that
produces that file is not repeated here.

Fig 4d: per-fish regression of neural activity onto recovery-swim-onset
time (predictive cells only: r>0, p<0.03), reusing the same cached
cell_corr_recovery_swim_time.npz. Ported from
Clusters/depreciated/Figure_5_swim_predictor/neural_prediction_swim_timing.ipynb.

Run from this src/ folder: `python Fig_4cd_generate_processed_data.py`.
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('../../Datalists/data_list_in_analysis_neuron_v8.csv')
atlas_shape = (183, 1325, 2509)  # matches /nrs/ahrens/Ziqiang/Atlas/atlas.npy


####################################
### Fig 4c: recovery-swim-time anti-correlation map -> xy/xz projections
####################################

cell_locs = []
r_list = []
for _, row in df.iterrows():
    save_root = row['save_dir']+'/'
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    cells_center = np.load(save_root+'cell_center_registered.npy')[cell_in_brain]
    corr_ = np.load(save_root+'cell_corr_recovery_swim_time.npz', allow_pickle=True)
    r, p = corr_['r_list'], corr_['p_list']
    idx = p < 0.05
    r_list.append(r[idx])
    cell_locs.append(cells_center[idx])
r_list = np.concatenate(r_list)
cell_locs = np.concatenate(cell_locs)

rz, ry, rx = 5, 5, 5
n_list = np.concatenate([cell_locs, r_list[:, None]], axis=1)[r_list < 0]
ind_loc = (n_list[:, 1]<atlas_shape[1]-1) & (n_list[:, 1]>0)
ind_loc = ind_loc & (n_list[:, 0]<atlas_shape[0]-1) & (n_list[:, 0]>0)
ind_loc = ind_loc & (n_list[:, 2]<atlas_shape[2]-1) & (n_list[:, 2]>0)
n_list = n_list[ind_loc]
num_cells = ind_loc.sum()
map_corr = np.zeros(atlas_shape)
map_weight = np.zeros(atlas_shape)

for n in range(num_cells):
    z, y, x, corr_ = n_list[n]
    z, x, y = np.array([z, x, y]).astype('int')
    map_corr[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += -corr_
    map_weight[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += 1

result_tmp = map_corr/map_weight
result_tmp[map_weight<3] = 0
pulse_recovery_neg_xy = np.percentile(result_tmp, 99.5, axis=0)
pulse_recovery_neg_xz = np.percentile(result_tmp, 99.5, axis=1)
del map_corr, map_weight, result_tmp


####################################
### Fig 4d: per-fish neural-activity -> swim-onset-time regression
####################################

animal_list = []
for _, row in df.iterrows():
    save_root = row['save_dir']+'/'
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    corr_ = np.load(save_root+'cell_corr_recovery_swim_time.npz', allow_pickle=True)
    r, p = corr_['r_list'], corr_['p_list']
    idx_r = (r>0) & (p<0.03)

    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain][idx_r].mean(axis=0)
    ephys_ = np.load(save_root+'KA_ephys.npz', allow_pickle=True)
    len_dff = dFF_.shape[0]
    epoch_frame = ephys_['epoch_frame'][:len_dff]
    pulse_frame = ephys_['pulse_frame'][:len_dff]
    lswim_frame = ephys_['lswim_frame'][:len_dff]
    rswim_frame = ephys_['rswim_frame'][:len_dff]

    rl, rr = row['rl'], row['rr']
    swim_frame_ = lswim_frame if rl >= rr else rswim_frame
    pulse_frame_ = pulse_frame.copy()
    pulse_frame_[epoch_frame%5<3] = 0
    recovery_swim_thres = np.percentile(swim_frame_[swim_frame_>0], 0)

    # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    len_ = len(epoch_on)-1

    CL_trial = []
    time_ticks = []
    for n_ in range(len_):
        on_ = epoch_on[n_]
        off_ = epoch_on[n_+1]-1
        if len(np.unique(epoch_frame[on_:off_]))<5:
            continue
        epoch_ = epoch_frame[on_:off_]
        if swim_frame_[on_:off_][epoch_%5==1].sum()==0:
            continue
        swm_ = swim_frame_[on_:off_]
        pulse_ = pulse_frame_[on_:off_]
        CL_trial_ = epoch_[10]//5==0

        # remove OL active trials -- fish swim during pause
        if (not CL_trial_) & (swm_[epoch_%5==2].sum()>0):
            continue
        # remove the long pause trial
        if (not CL_trial_) & ((epoch_%5==2).sum()>10):
            continue

        catch_trial_ = pulse_.sum()==0
        if catch_trial_:
            continue
        else:
            probe_on_ = np.where(pulse_>0)[0][0]

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

    dat_, resp, CL_trial_list = [], [], []
    for m, (last_epoch_on, _t_) in enumerate(time_ticks):
        for n in range(7):
            _time_ = last_epoch_on-36+n*6
            _dat_ = dFF_[_time_:_time_+6].sum()-dFF_[_time_+1]*6
            dat_.append(_dat_)
            resp.append((_t_-_time_)/3)
            CL_trial_list.append(CL_trial[m])
    dat_ = np.array(dat_)
    resp = np.array(resp)
    CL_trial_list = np.array(CL_trial_list)

    reg = GradientBoostingRegressor(random_state=0)
    reg.fit(dat_[:, None], resp[:, None])
    resp_est = reg.predict(dat_[:, None])
    animal_list.append((resp_est, resp, CL_trial_list))

resp_est_all = np.concatenate([a[0] for a in animal_list])
resp_all = np.concatenate([a[1] for a in animal_list])
CL_trial_all = np.concatenate([a[2] for a in animal_list])
fish_id_all = np.concatenate([np.full(len(a[0]), i) for i, a in enumerate(animal_list)])


np.savez('../processed_data/Fig_4cd_neural_prediction_swim.npz',
         pulse_recovery_neg_xy=pulse_recovery_neg_xy,
         pulse_recovery_neg_xz=pulse_recovery_neg_xz,
         resp_est=resp_est_all,
         resp=resp_all,
         CL_trial=CL_trial_all,
         fish_id=fish_id_all)
