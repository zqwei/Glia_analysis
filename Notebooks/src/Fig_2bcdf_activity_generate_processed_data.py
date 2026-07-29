'''
Generate the processed data for Figure_2bcdf_activity_plot.ipynb: for each of
Fig 2b (pulse-position responsive), 2c (integrative, positive), 2d
(pulse-position suppressed), and 2f (motor-responsive, positive), a
CL/OL activity heatmap (already sorted + sliced, matching the source
notebooks' own display code) and the CL/OL per-cell trial-averaged trace
arrays (mean+shaded-err computed later, in the notebook, via plot_shade_err).

2b/2c/2d ported from
Notebooks/additional/Figure_2_brain_dynamics/Figure_2_brain_clusters_af_pulse_v2.ipynb,
which itself only loads two small already-precomputed caches (no raw per-fish
data needed): cell_list_dynamics_early_pulse.npz (CL/OL trial-averaged dFF
per cell, all fish) and pulse_int_precomputed.npz (per-cell pulse/int
correlation r/p values).

2f ported from
Notebooks/additional/Figure_2_brain_dynamics/Figure_2_ex_motor_dynamics_v1.ipynb,
which computes CL/OL trial-averaged dFF from one example fish's raw data
(row index 2 in the datalist) via the same trial-parsing logic used
elsewhere in this project (e.g. Fig_4cd), then correlates against motor/pulse
regressors already cached per-fish (cell_motor_corr.npz,
cell_pulse_series_corr.npz).

Note: the 2d (pulse-neg) trace panel's cell-selection threshold is ported
verbatim from the source notebook: `p_r_thres = -0.05; pulse_r < -p_r_thres`,
i.e. `pulse_r < 0.05` -- a much weaker cutoff than the brain map's own
`pulse_r < 0` (see Fig_2bcdf_4f_generate_pulse_maps.py). Kept as-is since
that's what the given source notebook does; flagged for the user to confirm
this is intentional.

Run from this src/ folder: `python Fig_2bcdf_activity_generate_processed_data.py`.
'''

import numpy as np
import pandas as pd
from scipy.stats import zscore

cluster_folder = '/nrs/ahrens/Ziqiang/Jing_Glia_project/processed_af_data/cell_spatial_clusters_af/'
out = {}


####################################
### 2b/2c/2d: pulse-position / integrative cells (all fish, precomputed)
####################################

_ = np.load(cluster_folder+'cell_list_dynamics_early_pulse.npz', allow_pickle=True)
CL_dFF_list = _['CL_dFF_list']
OL_dFF_list = _['OL_dFF_list']

_ = np.load(cluster_folder+'pulse_int_precomputed.npz', allow_pickle=True)
pulse_r = _['pulse_r']
int_r = _['int_r']

zdat_ = zscore(np.concatenate([CL_dFF_list, OL_dFF_list], axis=1), axis=-1)

# Fig 2b: pulse-position responsive cells (positive)
num_cells = 5000
sort_idx = np.argsort(-pulse_r)[:num_cells]
out['pulse_pos_CL_heatmap'] = zdat_[sort_idx][:, :40]
out['pulse_pos_OL_heatmap'] = zdat_[sort_idx][:, 40:]
p_r_thres = 0.6
out['pulse_pos_CL_trace'] = CL_dFF_list[pulse_r>p_r_thres]
out['pulse_pos_OL_trace'] = OL_dFF_list[pulse_r>p_r_thres]

# Fig 2d: pulse-position suppressed cells (negative)
sort_idx = np.argsort(pulse_r)[:num_cells]
out['pulse_neg_CL_heatmap'] = zdat_[sort_idx][:, :40]
out['pulse_neg_OL_heatmap'] = zdat_[sort_idx][:, 40:]
p_r_thres = -0.05
out['pulse_neg_CL_trace'] = CL_dFF_list[pulse_r<-p_r_thres]
out['pulse_neg_OL_trace'] = OL_dFF_list[pulse_r<-p_r_thres]

# Fig 2c: integrative cells (positive)
num_cells = 6000
sort_idx = np.argsort(-int_r)[:num_cells]
out['int_pos_CL_heatmap'] = zdat_[sort_idx][:, :40]
out['int_pos_OL_heatmap'] = zdat_[sort_idx][:, 40:]
int_r_thres = 0.6
out['int_pos_CL_trace'] = CL_dFF_list[int_r>int_r_thres]
out['int_pos_OL_trace'] = OL_dFF_list[int_r>int_r_thres]

del CL_dFF_list, OL_dFF_list, zdat_, pulse_r, int_r


####################################
### 2f: motor-responsive cells, positive (single example fish, row 2)
####################################

df = pd.read_csv('../../Datalists/data_list_in_analysis_neuron_v8.csv')
row = df.iloc[2]
save_root = row['save_dir']+'/'

trial_post = 35
trial_pre = 5
swim_thres = 30

cell_in_brain = np.load(save_root+'cell_in_brain.npy')
_ = np.load(save_root+'KA_ephys.npz', allow_pickle=True)
dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain]
len_dff = dFF_.shape[1]
epoch_frame = _['epoch_frame'][:len_dff]
pulse_frame = _['pulse_frame'][:len_dff]
lswim_frame = _['lswim_frame'][:len_dff]
rswim_frame = _['rswim_frame'][:len_dff]

rl, rr = row['rl'], row['rr']
swim_frame_ = lswim_frame if rl >= rr else rswim_frame
pulse_frame_ = pulse_frame.copy()
pulse_frame_[epoch_frame%5<3] = 0
recovery_swim_thres = np.percentile(swim_frame_[swim_frame_>0], 20)

epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
len_ = len(epoch_on)-1
dFF_trial = []
CL_trial = []

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

    if (not CL_trial_) & (swim_frame_[on_:off_][epoch_%5==2].sum()>0):
        continue
    if swim_frame_[on_:off_][epoch_%5==2].sum()==0:
        last_swm = np.where(swim_frame_[on_:off_][epoch_%5<2])[0][-1]
        last_swm = last_swm - (epoch_%5<2).sum()
    else:
        last_swm = 0
    if CL_trial_ & (last_swm<-10):
        continue
    if (not CL_trial_) & ((epoch_%5==2).sum()>10):
        continue
    if (CL_trial_) & ((epoch_%5==2).sum()<20):
        continue
    if ((epoch_%5==2).sum()>100):
        continue

    catch_trial_ = pulse_.sum()==0
    if catch_trial_:
        continue
    else:
        probe_on_ = np.where(pulse_>0)[0][0]
    probe_on_ = probe_on_+on_

    swm_ = np.cumsum(swim_frame_[probe_on_-trial_pre:probe_on_+trial_post])
    if (swm_>0).sum()>swim_thres:
        continue

    _recovery_swim = swim_frame_[on_:off_][epoch_%5==3]>recovery_swim_thres
    if _recovery_swim.sum()>0:
        _t_ = np.where(_recovery_swim)[0][0]
        CL_trial.append(CL_trial_)
        tmp_ = np.zeros((dFF_.shape[0], trial_post+62+trial_pre))
        tmp_[:] = np.nan
        if CL_trial_:
            _t_min_ = min(_t_+6, trial_post)
            _t_ = _t_ + (epoch_%5<=2).sum()+on_
            tmp_[:, trial_post-_t_min_:] = dFF_[:, _t_-_t_min_:_t_+trial_pre+62]
        else:
            _t_min_ = min(_t_+6, trial_post+62)
            _t_ = _t_ + (epoch_%5<=2).sum()+on_
            tmp_[:, trial_post+62-_t_min_:] = dFF_[:, _t_-_t_min_:_t_+trial_pre]
        dFF_trial.append(tmp_)

dFF_trial = np.array(dFF_trial).transpose([1, 2, 0])
CL_trial = np.array(CL_trial)
CL_dFF_list = np.nanmean(dFF_trial[:, :, CL_trial], axis=-1)
OL_dFF_list = np.nanmean(dFF_trial[:, :, ~CL_trial], axis=-1)

_ = np.load(save_root+'cell_motor_corr.npz', allow_pickle=True)
r_list = _['r_cell']
_ = np.load(save_root+'cell_pulse_series_corr.npz', allow_pickle=True)
pulse_r = _['r_cell']

zdat_ = zscore(np.concatenate([CL_dFF_list, OL_dFF_list], axis=1), axis=-1)

num_cells = 5000
idxx_ = np.abs(pulse_r)<0.01
sort_idx = np.argsort(-r_list[idxx_])[:num_cells]
out['motor_pos_CL_heatmap'] = zdat_[idxx_][sort_idx][:, :40+62]
out['motor_pos_OL_heatmap'] = zdat_[idxx_][sort_idx][:, 40+62:]

idxx_ = (np.abs(pulse_r)<0.05) & (r_list>0.5)
out['motor_pos_CL_trace'] = CL_dFF_list[idxx_]
out['motor_pos_OL_trace'] = OL_dFF_list[idxx_]


np.savez('../processed_data/Fig_2bcdf_activity_plot.npz', **out)
