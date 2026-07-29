'''
Generate the processed data for Figure_3b_example_cluster_dynamics.ipynb: for
each of three example brain regions -- Cb, OT, IO (IO used in place of PT,
whose per-region dFF cache was never generated -- anatomically/functionally
exchangeable for this purpose per the user's call) -- produce (1) the
long-trial CL/OL population-average dynamics trace (probe-aligned, extending
through the full recovery period) and (2) the pulse-response statistics
(short window aligned to the last pulse before the recovery swim), both
aggregated across fish 0/1/2.

Both panels reuse the same per-region cached array fish_{n}_{loc}.npy in
/nrs/ahrens/Ziqiang/Jing_Glia_project/processed_af_data/cluster_dynamics/,
which makes region selection fully code-reproducible (no manual per-fish
DBSCAN-cluster lookup, unlike the original single-fish, single-region
version of the long-trial panel).

Ported from (kept as two separate, faithfully-copied trial-selection loops,
since the two source scripts differ subtly in which filters are applied):
- Notebooks/old_figure_panels/Figure_4A_ex_brain_cluster_long_trial.py
  (long-trial windowing: probe-aligned, trial_post=185)
- Notebooks/old_figure_panels/Figure_4DE_af_brain_cluster_at_swim_on.py
  (pulse-response statistics: short window aligned to last pulse before the
  recovery swim, per-region cached dFF array, loc parameter)

Run from this src/ folder: `python Fig_3b_generate_processed_data.py`.
'''

import numpy as np
import pandas as pd

df = pd.read_csv('../../Datalists/data_list_in_analysis_neuron_v8.csv')
dat_save_folder = '/nrs/ahrens/Ziqiang/Jing_Glia_project/processed_af_data/cluster_dynamics/'

regions = ['Cb', 'OT', 'IO']
fish_list = [0, 1, 2]

out = {}

for loc in regions:

    ####################################
    ### long-trial dynamics (Figure_4A windowing, probe-aligned)
    ####################################
    trial_post = 185
    dFF_epochs = []
    CL_trial = []

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
