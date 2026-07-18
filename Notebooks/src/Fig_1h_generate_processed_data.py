'''
Precompute dFF_epochs, CL_trial, cells_center, and the per-epoch
CL-vs-OL Mann-Whitney p_mat/sign_mat for the example fish used by
Figure_1h_average_diff_dynamics_epochs.ipynb, so the notebook itself can
skip the raw-data trial-parsing loop and the p-value computation, and
just load everything from
processed_data/Fig_1h_ex_fish_average_diff_dynamics_epochs.npz.

Run from this src/ folder: `python Fig_1h_generate_processed_data.py`.
Rerun whenever the underlying fish data or trial-selection logic changes.
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import mannwhitneyu

df = pd.read_csv('../../Datalists/data_list_in_analysis_neuron_v8.csv')
row = df.iloc[0]
save_root = row['save_dir']+'/'
cells_center = np.load(save_root+'cell_center_registered.npy')
cell_in_brain = np.load(save_root+'cell_in_brain.npy')
cells_center = cells_center[cell_in_brain]

dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain]

_ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
len_dff = dFF_.shape[1]
epoch_frame=_['epoch_frame'][:len_dff]
pulse_frame=_['pulse_frame'][:len_dff]
lswim_frame=_['lswim_frame'][:len_dff]
rswim_frame=_['rswim_frame'][:len_dff]
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

dFF_epochs = []
CL_trial = []
trial_pre = 3
trial_post = 9
_tmp_ = np.zeros(dFF_.shape[0])
_tmp_[:] = np.nan # this is used to marked the trial where recovery swim not happen

for n_ in range(len_):
    dFF_epochs_ = []
    on_ = epoch_on[n_]
    off_ = epoch_on[n_+1]-1
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
    if CL_trial_ & (last_swm<-3):
        continue

    # remove the long pause trial
    if (not CL_trial_) & ((epoch_%5==2).sum()>10):
        continue
    if (CL_trial_) & ((epoch_%5==2).sum()<20):
        continue
    if ((epoch_%5==2).sum()>100):
        continue

    # set probe on time; skip catch (pause) trials
    catch_trial_ = pulse_.sum()==0
    if catch_trial_:
        continue
    else:
        probe_on_ = np.where(pulse_>0)[0][0]

    CL_trial.append(CL_trial_)
    dFF_ave_ = dFF_[:, on_:off_]
    evoke_on_ = (epoch_%5==0).sum()
    dFF_epochs_.append(dFF_ave_[:, evoke_on_+trial_post-3:evoke_on_+trial_post+3].mean(axis=1)) # dynamics evoke onset
    last_swm = np.where(swim_frame_[on_:off_][epoch_%5<2])[0][-1]
    dFF_epochs_.append(dFF_ave_[:, last_swm-1:last_swm+6].mean(axis=1)) # dynamics evoke offset
    dFF_epochs_.append(dFF_ave_[:, probe_on_-3:probe_on_+0].mean(axis=1)) # dynamics probe on
    dFF_epochs_.append(dFF_ave_[:, probe_on_+6:probe_on_+9].mean(axis=1)) # dynamics probe on
    _recovery_swim = swm_[epoch_%5==3]>recovery_swim_thres
    if _recovery_swim.sum()>0:
        _t_ = np.where(_recovery_swim)[0][0]
        _t_ = _t_ + (epoch_%5<=2).sum()
        dFF_epochs_.append(dFF_ave_[:, _t_-6:_t_-3].mean(axis=1))
    else:
        dFF_epochs_.append(_tmp_)
    dFF_epochs.append(dFF_epochs_)

CL_trial = np.array(CL_trial)
dFF_epochs = np.array(dFF_epochs)

print('dFF_epochs shape:', dFF_epochs.shape)
print('CL_trial shape:', CL_trial.shape)
print('cells_center shape:', cells_center.shape)


####################################
### CL vs OL significance per epoch
####################################

def diff_p_value(mat, CL_trial):
    num_cells = mat.shape[0]
    p_mat = np.zeros(num_cells)
    for n_cell in range(num_cells):
        x = mat[n_cell, ~CL_trial]
        y = mat[n_cell, CL_trial]
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        _, p_mat[n_cell] = mannwhitneyu(x, y)
    return p_mat,


def unpacking_apply_func(list_params):
    func1d, arr, args, kwargs = list_params
    return func1d(arr, *args, **kwargs)


def parallel_to_chunks(func1d, arr, *args, **kwargs):
    mp_count = min(mp.cpu_count(), arr.shape[0])
    print(f'Number of processes to parallel: {mp_count}')
    chunks = [(func1d, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, mp_count)]
    pool = mp.Pool(processes=mp_count)
    individual_results = pool.map(unpacking_apply_func, chunks)
    pool.close()
    pool.join()

    results = ()
    for i_tuple in range(len(individual_results[0])):
        results = results + (np.concatenate([_[i_tuple] for _ in individual_results]), )
    return results


num_cells = cells_center.shape[0]
p_mat = np.zeros((5, num_cells))
for n_epoch in range(5):
    p_mat_epoch, = parallel_to_chunks(diff_p_value, dFF_epochs[:, n_epoch, :].T, CL_trial)
    p_mat[n_epoch] = p_mat_epoch

sign_mat = np.zeros((5, num_cells))
for n_epoch in range(5):
    sign_mat[n_epoch] = np.nanmean(dFF_epochs[~CL_trial], axis=0)[n_epoch] - np.nanmean(dFF_epochs[CL_trial], axis=0)[n_epoch]>0

np.savez('../processed_data/Fig_1h_ex_fish_average_diff_dynamics_epochs.npz',
         dFF_epochs=dFF_epochs, CL_trial=CL_trial, cells_center=cells_center,
         p_mat=p_mat, sign_mat=sign_mat)
