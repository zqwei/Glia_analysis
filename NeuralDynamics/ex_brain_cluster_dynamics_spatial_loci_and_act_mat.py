import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.stats import zscore
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.5, style='ticks')
atlas = np.load('/nrs/ahrens/Ziqiang/Atlas/atlas.npy')

df = pd.read_csv('Datalists/data_list_in_analysis_neuron_v8.csv')
row = df.iloc[2]
save_root = row['save_dir']+'/'
cells_center = np.load(save_root+'cell_center_registered.npy')
cell_in_brain = np.load(save_root+'cell_in_brain.npy')
cells_center = cells_center[cell_in_brain]

cell_idx = np.load(save_root + 'cell_state_pulse_filtered.npy')
label_sub = np.load(save_root + 'fsensory_cluster_label.npy')
# 4, IO; 9, OT; 10, SLoMO; 8, PT; 
label_ = 16
# cell_idx = np.load(save_root + 'cell_state_motor_filtered.npy')
# label_sub = np.load(save_root + 'fmotor_cluster_label.npy')
# label_ = 12 #4, 6, 12
idx_ = label_sub[:,0]==label_ # (label_sub[:,0]==4) | (label_sub[:,0]==8) # & (cells_center[cell_idx, 2]<2000) # (cells_center[cell_idx, 2]>1100) & (cells_center[cell_idx, 2]<1300)

dFF_ave = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain][cell_idx][idx_]
loc_sort = cells_center[:, 2][cell_idx][idx_]

_ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
probe_amp=_['probe_amp']
swim_t_frame=_['swim_t_frame']
len_dff = dFF_ave.shape[1]
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

dFF_epochs = []

# 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
len_ = len(epoch_on)-1
CL_trial = []
swim_evoke = []
trial_pre = 3
_tmp_ = np.zeros((dFF_ave.shape[0], 40)) # this is used to marked the trial where recovery swim not happen
_tmp_[:] = np.nan
recovery_swim = []
recovery_swim_thres = np.percentile(swim_frame_[swim_frame_>0], 10)

for n_ in range(len_):
    dFF_epochs_ = []
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
    if CL_trial_ & (last_swm<-3):
        continue
    
    # length of pause -- it seems like some of them are extremely long
    # remove the long pause trial
    if (not CL_trial_) & ((epoch_%5==2).sum()>10):
        continue
    if (CL_trial_) & ((epoch_%5==2).sum()<20):
        continue
    if ((epoch_%5==2).sum()>100):
        continue

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
    CL_trial.append(CL_trial_)
    dFF_ave_ = dFF_ave[:, on_:off_]
    evoke_on_ = (epoch_%5==0).sum()
    # dFF_off_ = dFF_ave_[evoke_on_-4:evoke_on_+0].mean(axis=0)
    last_swm = np.where(swim_frame_[on_:off_][epoch_%5<2])[0][-1]
    _recovery_swim = swm_[epoch_%5==3]>recovery_swim_thres
    if _recovery_swim.sum()>0:
        _t_ = np.where(_recovery_swim)[0][0]
        recovery_swim.append(_t_)
        _t_ = _t_ + (epoch_%5<=2).sum()
        dFF_epochs_.append(dFF_ave_[:, probe_on_-5:probe_on_+35])
        dFF_epochs_.append(dFF_ave_[:, _t_-35:_t_+5])
    else:
        recovery_swim.append(np.nan)
        dFF_epochs_.append(dFF_ave_[:, probe_on_-5:probe_on_+35])
        dFF_epochs_.append(_tmp_)
    dFF_epochs.append(np.array(dFF_epochs_))

CL_trial = np.array(CL_trial)
dFF_epochs = np.array(dFF_epochs)

# labels_ = (cells_center[:, 2][cell_idx][idx_]>650).astype('int') +(cells_center[:, 2][cell_idx][idx_]>1100).astype('int') + (cells_center[:, 2][cell_idx][idx_]>1800).astype('int')
# labels_ = (cells_center[:, 2][cell_idx][idx_]>1100).astype('int') + (cells_center[:, 2][cell_idx][idx_]>1300).astype('int') + (cells_center[:, 2][cell_idx][idx_]>1800).astype('int')
# labels_[labels_>1] = labels_[labels_>1] +1
# cidx_ = (cells_center[:, 2][cell_idx][idx_]>1100) & (cells_center[:, 2][cell_idx][idx_]<1300) & (cells_center[:, 1][cell_idx][idx_]>400) & (cells_center[:, 1][cell_idx][idx_]<800) 
# labels_[cidx_] = 2
labels_ = (cells_center[:, 2][cell_idx][idx_]>700).astype('int') + (cells_center[:, 2][cell_idx][idx_]>1100).astype('int') + (cells_center[:, 2][cell_idx][idx_]>1800).astype('int')
labels_[labels_>1] = labels_[labels_>1] +1
cidx_ = (cells_center[:, 2][cell_idx][idx_]<700) & (cells_center[:, 0][cell_idx][idx_]>110)
labels_[cidx_] = 2

mat_CL = np.nanmean(dFF_epochs[CL_trial], axis=0)
mat_OL = np.nanmean(dFF_epochs[~CL_trial], axis=0)
mat_CL_ = mat_CL.copy()
# mat_CL_[0] = mat_CL[0] - mat_CL[0][:, :5].mean(axis=1, keepdims=True)
mat_OL_ = mat_OL.copy()
# mat_OL_[0] = mat_OL[0] - mat_OL[0][:, :5].mean(axis=1, keepdims=True)
mat_ = np.concatenate([mat_CL_, mat_OL_], axis=2)
mat_ = zscore(mat_, axis=2)
# mat_ = mat_[:, np.argsort(loc_sort), :]
mat_ = mat_[:, np.argsort(labels_), :]

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ax = ax.flatten()
num_cells = mat_.shape[1]
print(num_cells)

ax[0].imshow(mat_[0, :, :40], vmax=2, vmin=-1, aspect='auto', origin='lower')
ax[0].vlines(5, 0, num_cells, linestyle='--', color='w')
ax[0].set_axis_off()
ax[1].imshow(mat_[0, :, 40:], vmax=2, vmin=-1, aspect='auto', origin='lower')
ax[1].vlines(5, 0, num_cells, linestyle='--', color='w')
ax[1].set_axis_off()
ax[2].imshow(mat_[1, :, :40], vmax=2, vmin=-1, aspect='auto', origin='lower')
ax[2].vlines(35, 0, num_cells, linestyle='--', color='w')
ax[2].set_axis_off()
ax[3].imshow(mat_[1, :, 40:], vmax=2, vmin=-1, aspect='auto', origin='lower')
ax[3].vlines(35, 0, num_cells, linestyle='--', color='w')
ax[3].set_axis_off()
# plt.savefig(f'act_matrix_{label_}_epoch.svg')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(atlas.max(0), cmap = plt.cm.gray, origin='lower')
plt.scatter(cells_center[:, 2][cell_idx][idx_], cells_center[:, 1][cell_idx][idx_], s=1, c=labels_, cmap=plt.cm.tab10)
plt.axis('off')
plt.savefig('brain_cluster.svg')

plt.figure(figsize=(8, 3))
plt.imshow(atlas.max(1), cmap = plt.cm.gray, origin='lower', aspect='auto')
plt.scatter(cells_center[:, 2][cell_idx][idx_], cells_center[:, 0][cell_idx][idx_], s=1, c=labels_, cmap=plt.cm.tab10)
plt.axis('off')
plt.savefig('brain_cluster_side.svg')

