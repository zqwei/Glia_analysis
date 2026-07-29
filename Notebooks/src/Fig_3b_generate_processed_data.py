'''
Generate the processed data for Figure_3b_example_cluster_dynamics.ipynb: for
each of three example brain regions -- Cb, OT, IO -- produce (1) the
long-trial CL/OL population-average dynamics trace (probe-aligned, extending
through the full recovery period) and (2) the pulse-response statistics
(short window aligned to the last pulse before the recovery swim).

The two panels use different data sources, faithfully matching their two
different original notebooks:

- Long-trial dynamics (1): fish 0 only, raw per-fish selection straight from
  `Notebooks/additional/Figure_3_dynamics/long_trial_ex_cluster_dynamics_Figure_4A.ipynb`
  (`fsensory_cluster_label.npy` + `cell_state_pulse_filtered.npy`). Region ->
  label_ confirmed via `Notebooks/additional/cluster_dynamics/ex_brain_cluster_TL.ipynb`'s
  comment ("4, IO; 9, OT; 10, SLoMO; 8, PT"): OT=9, IO=4. No `fsensory_cluster_label`
  value for Cb was found in any notebook, so per the user's call, **Cb reuses
  label_=10 (SLoMO) plus its z-filter** -- the one region this source notebook
  fully documents end-to-end.
- Pulse-response statistics (2): Notebooks/old_figure_panels/Figure_4DE_af_brain_cluster_at_swim_on.py,
  already parameterized by a `loc` region string + the per-region cached
  `fish_{n}_{loc}.npy` array in
  /nrs/ahrens/Ziqiang/Jing_Glia_project/processed_af_data/cluster_dynamics/,
  aggregated across fish 0/1/2.

Run from this src/ folder: `python Fig_3b_generate_processed_data.py`.
'''

import numpy as np
import pandas as pd

df = pd.read_csv('../../Datalists/data_list_in_analysis_neuron_v8.csv')
dat_save_folder = '/nrs/ahrens/Ziqiang/Jing_Glia_project/processed_af_data/cluster_dynamics/'

regions = ['Cb', 'OT', 'IO']
fish_list = [0, 1, 2]

# region -> (label_ in fsensory_cluster_label.npy, optional z-range filter on
# cells_center[cell_idx, 2]); Cb reuses the SLoMO label/filter (see docstring).
region_label_config = {
    'Cb': (10, (1100, 1500)),
    'OT': (9, None),
    'IO': (4, None),
}

out = {}

####################################
### long-trial dynamics (Figure_4A windowing, probe-aligned, fish 0 only)
####################################

row0 = df.iloc[0]
save_root0 = row0['save_dir']+'/'
cells_center0 = np.load(save_root0+'cell_center_registered.npy')
cell_in_brain0 = np.load(save_root0+'cell_in_brain.npy')
cells_center0 = cells_center0[cell_in_brain0]
cell_idx0 = np.load(save_root0 + 'cell_state_pulse_filtered.npy')
label_sub0 = np.load(save_root0 + 'fsensory_cluster_label.npy')
dFF_all0 = np.load(save_root0+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain0]

_ = np.load(save_root0 + 'KA_ephys.npz', allow_pickle=True)
probe_amp = _['probe_amp']
pulse_frame_full = _['pulse_frame']
epoch_frame_full = _['epoch_frame']
lswim_frame_full = _['lswim_frame']
rswim_frame_full = _['rswim_frame']
rl0, rr0 = row0['rl'], row0['rr']
swim_frame_full = lswim_frame_full if rl0 >= rr0 else rswim_frame_full

for loc in regions:
    trial_post = 185
    dFF_epochs = []
    CL_trial = []

    label_, z_filter = region_label_config[loc]
    idx_ = label_sub0[:, 0] == label_
    if z_filter is not None:
        zmin, zmax = z_filter
        idx_ = idx_ & (cells_center0[cell_idx0, 2] > zmin) & (cells_center0[cell_idx0, 2] < zmax)
    dFF_ave = dFF_all0[cell_idx0][idx_].mean(axis=0)

    len_dff = dFF_ave.shape[0]
    epoch_frame = epoch_frame_full[:len_dff]
    pulse_frame = pulse_frame_full[:len_dff]
    swim_frame_ = swim_frame_full[:len_dff]
    pulse_frame_ = pulse_frame.copy()
    pulse_frame_[epoch_frame%5<3] = 0
    recovery_swim_thres = np.percentile(swim_frame_[swim_frame_>0], 0)

    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    len_ = len(epoch_on)-1

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

        if (not CL_trial_) & (swm_[epoch_%5==2].sum()>0):
            continue

        if swm_[epoch_%5==2].sum()==0:
            last_swm = np.where(swm_[epoch_%5<2])[0][-1]
            last_swm = last_swm - (epoch_%5<2).sum()
        else:
            last_swm = 0
        if CL_trial_ & (last_swm<-3):
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

        CL_trial.append(CL_trial_)
        dFF_ave_ = dFF_ave[on_:off_]
        dFF_tmp_ = dFF_ave_[probe_on_-5:probe_on_+trial_post].copy()
        _recovery_swim = swm_[epoch_%5==3]>recovery_swim_thres
        if _recovery_swim.sum()>0:
            _t_ = np.where(_recovery_swim)[0][0]
            _t_ = _t_ + (epoch_%5<=2).sum()
            dFF_tmp_[_t_-5:] = np.nan
        dFF_epochs.append(dFF_tmp_)

    CL_trial = np.array(CL_trial)
    len_dFF_epochs = np.array([len(_) for _ in dFF_epochs]).max()
    dFF_epochs_ = np.zeros((len(dFF_epochs), len_dFF_epochs))
    dFF_epochs_[:] = np.nan
    for n in range(len(dFF_epochs)):
        len_ = len(dFF_epochs[n])
        dFF_epochs_[n, :len_] = dFF_epochs[n]

    out[f'{loc}_long_CL_dynamics'] = dFF_epochs_[CL_trial]*100
    out[f'{loc}_long_OL_dynamics'] = dFF_epochs_[~CL_trial]*100

for loc in regions:
    ####################################
    ### pulse-response statistics (Figure_4DE windowing, swim-on aligned)
    ####################################
    dFF_epochs_swim = []
    CL_trial_swim = []

    for n_fish in fish_list:
        row = df.iloc[n_fish]
        save_root = row['save_dir']+'/'
        dFF_ave = np.load(dat_save_folder + f'fish_{n_fish}_{loc}.npy')

        _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
        len_dff = dFF_ave.shape[0]
        epoch_frame = _['epoch_frame'][:len_dff]
        pulse_frame = _['pulse_frame'][:len_dff]
        lswim_frame = _['lswim_frame'][:len_dff]
        rswim_frame = _['rswim_frame'][:len_dff]
        rl, rr = row['rl'], row['rr']
        swim_frame_ = lswim_frame if rl >= rr else rswim_frame
        pulse_frame_ = pulse_frame.copy()
        pulse_frame_[epoch_frame%5<3] = 0
        recovery_swim_thres = np.percentile(swim_frame_[swim_frame_>0], 0)

        epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
        len_ = len(epoch_on)-1

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

            if (not CL_trial_) & (swm_[epoch_%5==2].sum()>0):
                continue

            if (not CL_trial_) & ((epoch_%5==2).sum()>10):
                continue

            catch_trial_ = pulse_.sum()==0
            if catch_trial_:
                continue
            else:
                probe_on_ = np.where(pulse_>0)[0][0]
                probe_off_ = np.where(pulse_>0)[0][-1]

            _recovery_swim = swm_[epoch_%5==3]>recovery_swim_thres
            if _recovery_swim.sum()>0:
                dFF_ave_ = dFF_ave[on_:off_]
                _t_ = np.where(_recovery_swim)[0][0]
                _t_ = _t_ + (epoch_%5<=2).sum()
                probe_on_switch = np.where((pulse_[:-1]==0) & (pulse_[1:]>0))[0]
                if (probe_on_switch<_t_).sum()==0:
                    continue
                last_epoch_on = probe_on_switch[probe_on_switch<_t_].max()
                if last_epoch_on<36:
                    continue
                CL_trial_swim.append(CL_trial_)
                dFF_epochs_swim.append(dFF_ave_[last_epoch_on-36:_t_+1])

    CL_trial_swim = np.array(CL_trial_swim)
    len_dFF_epochs_swim = np.array([len(_) for _ in dFF_epochs_swim]).max()
    dFF_epochs_swim_ = np.zeros((len(dFF_epochs_swim), len_dFF_epochs_swim))
    dFF_epochs_swim_[:] = np.nan
    for n in range(len(dFF_epochs_swim)):
        len_ = len(dFF_epochs_swim[n])
        dFF_epochs_swim_[n, :len_] = dFF_epochs_swim[n]

    dFF_epochs_pulse = np.zeros((len(dFF_epochs_swim_), 7))
    for n in range(7):
        dFF_epochs_pulse[:, n] = dFF_epochs_swim_[:, n*6:n*6+6].sum(axis=1)-dFF_epochs_swim_[:, n*6+1]*6

    out[f'{loc}_stats_CL'] = dFF_epochs_pulse[CL_trial_swim]*100
    out[f'{loc}_stats_OL'] = dFF_epochs_pulse[~CL_trial_swim]*100


np.savez('../processed_data/Fig_3b_example_cluster_dynamics.npz', **out)
