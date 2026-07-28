import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
sns.set(font_scale=1.5, style='ticks')
atlas = np.load('/nrs/ahrens/Ziqiang/Atlas/atlas.npy')

def plot_shade_err(x, y, axis=-1, plt=plt, linespec='-k', shadespec='k'):
    from scipy.stats import sem
    mean_ = np.nanmean(y, axis=axis)
    error = sem(y, axis=axis, nan_policy='omit')
    # error = np.nanstd(y, axis=axis)
    plt.plot(x, mean_, linespec)
    plt.fill_between(x, mean_-error, mean_+error, edgecolor=None, linewidth=0.0, facecolor=shadespec, alpha=0.8)


df = pd.read_csv('Datalists/data_list_in_analysis_neuron_v8.csv')
row = df.iloc[4]

save_root = row['save_dir']+'/'
cells_center = np.load(save_root+'cell_center_registered.npy')
cell_in_brain = np.load(save_root+'cell_in_brain.npy')
cells_center = cells_center[cell_in_brain]

cell_idx = np.load(save_root + 'cell_state_pulse_filtered.npy')
label_sub = np.load(save_root + 'fsensory_cluster_label.npy')
#2: OT
label_ = 0

# 6: vTel; 4: PO; 12: NEMO
# cell_idx = np.load(save_root + 'cell_state_motor_filtered.npy')
# label_sub = np.load(save_root + 'fmotor_cluster_label.npy')
# label_ = 12 #4, 6, 12
idx_ = (label_sub[:,0]==label_) #& (cells_center[cell_idx, 2]>2000) & (cells_center[cell_idx, 2]<2600)  #& (cells_center[cell_idx, 0]>100)

# labels_ = (cells_center[:, 2][cell_idx][idx_]>1300).astype('int') + (cells_center[:, 2][cell_idx][idx_]>1800).astype('int')
# plt.figure(figsize=(8, 6))
# plt.imshow(atlas.max(0), cmap = plt.cm.gray, origin='lower')
# plt.scatter(cells_center[:, 2][cell_idx][idx_], cells_center[:, 1][cell_idx][idx_], s=1, c=labels_, cmap=plt.cm.tab10)
# plt.axis('off')
# plt.show()

# labels_ = (cells_center[:, 2][cell_idx][idx_]>700).astype('int') + (cells_center[:, 2][cell_idx][idx_]>1800).astype('int')
# plt.figure(figsize=(8, 3))
# plt.imshow(atlas.max(1), cmap = plt.cm.gray, origin='lower', aspect='auto')
# plt.scatter(cells_center[:, 2][cell_idx][idx_], cells_center[:, 0][cell_idx][idx_], s=1, c=labels_, cmap=plt.cm.tab10)
# plt.axis('off')
# plt.show()

dFF_ave = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain][cell_idx][idx_].mean(axis=0)
_ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
probe_amp=_['probe_amp']
swim_t_frame=_['swim_t_frame']
len_dff = dFF_ave.shape[0]
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
_tmp_ = np.array([np.nan]*40) # this is used to marked the trial where recovery swim not happen
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
    # if CL_trial_ & (last_swm<-3):
    #     continue
    
    # length of pause -- it seems like some of them are extremely long
    # remove the long pause trial
    if (not CL_trial_) & ((epoch_%5==2).sum()>10):
        continue
    # if (CL_trial_) & ((epoch_%5==2).sum()<20):
    #     continue
    # if ((epoch_%5==2).sum()>100):
    #     continue

    # set probe on time
    catch_trial_ = pulse_.sum()==0
    # pause trial
    if catch_trial_:
        probe_on_ = (epoch_%5<3).sum()
        # continue
    else:
        probe_on_ = np.where(pulse_>0)[0][0]
        probe_off_ = np.where(pulse_>0)[0][-1]
        continue

    # probe_on_ = probe_on_+on_
    # print(probe_on_)

    # swm_ = np.cumsum(swim_frame_[probe_on_-trial_pre:probe_on_+trial_post])
    # if (swm_>0).sum()>swim_thres:
    #     continue

    # catch_trial.append(catch_trial_)
    # print(swim_frame_[on_:off_][epoch_%5<2].sum())
    CL_trial.append(CL_trial_)
    dFF_ave_ = dFF_ave[on_:off_]
    evoke_on_ = (epoch_%5==0).sum()
    # dFF_off_ = dFF_ave_[evoke_on_-4:evoke_on_+0].mean(axis=0)
    dFF_off_ = 0
    last_swm = np.where(swim_frame_[on_:off_][epoch_%5<2])[0][-1]
    dFF_epochs_.append(dFF_ave_[evoke_on_-3:last_swm+1]-dFF_off_) # evoke epoch
    dFF_epochs_.append(dFF_ave_[last_swm:probe_on_+1]-dFF_off_) # pause
    _recovery_swim = swm_[epoch_%5==3]>recovery_swim_thres
    if _recovery_swim.sum()>0:
        _t_ = np.where(_recovery_swim)[0][0]
        recovery_swim.append(_t_)
        _t_ = _t_ + (epoch_%5<=2).sum()
        dFF_epochs_.append(dFF_ave_[probe_on_-5:probe_on_+35] - dFF_off_)
        dFF_epochs_.append(dFF_ave_[_t_-35:_t_+5] - dFF_off_)
        if (probe_on_+35)<(_t_-35):
            dFF_epochs_.append(dFF_ave_[probe_on_+34:_t_-34] - dFF_off_)
        else:
            dFF_epochs_.append(np.array([dFF_ave_[probe_on_+34], np.nan]))
    else:
        recovery_swim.append(np.nan)
        dFF_epochs_.append(dFF_ave_[probe_on_-5:probe_on_+35] - dFF_off_)
        # dFF_epochs_.append(dFF_ave_[probe_on_:probe_on_+150] - dFF_off_)
        # dFF_epochs_.append(_tmp_)
        dFF_epochs_.append(_tmp_)
        dFF_epochs_.append(dFF_ave_[probe_on_+34:probe_on_+np.random.randint(55, high=130)] - dFF_off_)
    dFF_epochs.append(dFF_epochs_)

CL_trial = np.array(CL_trial)
dFF_epochs = np.array(dFF_epochs)

len_epoch = np.zeros(dFF_epochs.shape[1]).astype('int')
for n_epoch in range(dFF_epochs.shape[1]):
    _ = dFF_epochs[:, n_epoch]
    len_epoch[n_epoch] = np.max([len(__) for __ in _])
cs_len_epoch = np.cumsum(np.r_[0, len_epoch])

intep1d_dFF_evoke = []
num_trials = dFF_epochs.shape[0]

for n_epoch in range(2):
    intep1d_dFF_n_epoch = np.zeros((num_trials, len_epoch[n_epoch]))
    x = np.arange(len_epoch[n_epoch])
    for n_trial in range(num_trials):
        yp = dFF_epochs[n_trial, n_epoch]
        xp = np.linspace(0, len_epoch[n_epoch], len(yp))
        intep1d_dFF_n_epoch[n_trial] = np.interp(x, xp, yp)
    intep1d_dFF_evoke.append(intep1d_dFF_n_epoch)
intep1d_dFF_evoke = np.hstack(intep1d_dFF_evoke)

num_trials = dFF_epochs.shape[0]

n_epoch=4
intep1d_dFF_n_epoch = np.zeros((num_trials, len_epoch[n_epoch]))
x = np.arange(len_epoch[n_epoch])
for n_trial in range(num_trials):
    yp = dFF_epochs[n_trial, n_epoch]
    xp = np.linspace(0, len_epoch[n_epoch], len(yp))
    intep1d_dFF_n_epoch[n_trial] = np.interp(x, xp, yp)
intep1d_dFF_pulse = intep1d_dFF_n_epoch
intep1d_dFF_pulse_smooth = gaussian_filter1d(intep1d_dFF_pulse, sigma=3, axis=-1, mode='nearest')

intep1d_dFF_pulse_on = np.array([_ for _ in dFF_epochs[:, 2]])
intep1d_dFF_swim_on = np.array([_ for _ in dFF_epochs[:, 3]])

fig, ax = plt.subplots(1, 4, figsize=(8, 3))
ax = ax.flatten()
fig.subplots_adjust(wspace=0)

ax[0].plot(intep1d_dFF_evoke[~CL_trial].T*100, '-r', alpha=0.2, lw=0.5)
ax[0].plot(intep1d_dFF_evoke[CL_trial].T*100, '-k', alpha=0.2, lw=0.5)
plot_shade_err(np.arange(0, cs_len_epoch[2]), intep1d_dFF_evoke[CL_trial]*100, axis=0, \
               plt=ax[0], linespec='-k', shadespec='k')
plot_shade_err(np.arange(0, cs_len_epoch[2]), intep1d_dFF_evoke[~CL_trial]*100, axis=0, \
               plt=ax[0], linespec='-r', shadespec='r')
ax[0].vlines(cs_len_epoch[1], 0, 10, linestyles='--', colors='k')
ax[0].set_ylim([-1, 15])
ax[0].set_xlim([100, cs_len_epoch[2]])

ymin = -1
ymax = 15
ax[1].plot(np.arange(-5, 35)/3,intep1d_dFF_pulse_on[~CL_trial].T*100, '-r', alpha=0.2, lw=0.5)
ax[1].plot(np.arange(-5, 35)/3,intep1d_dFF_pulse_on[CL_trial].T*100, '-k', alpha=0.2, lw=0.5)
plot_shade_err(np.arange(-5, 35)/3, intep1d_dFF_pulse_on[CL_trial]*100, axis=0, \
               plt=ax[1], linespec='-k', shadespec='k')
plot_shade_err(np.arange(-5, 35)/3, intep1d_dFF_pulse_on[~CL_trial]*100, axis=0, \
               plt=ax[1], linespec='-r', shadespec='r')
ax[1].vlines(0, -1, 10, linestyles='--', colors='k')
ax[1].set_xlim([-5/3, 34/3])
ax[1].set_ylim([ymin, ymax])

ax[3].plot(np.arange(-35, 5)/3,intep1d_dFF_swim_on[~CL_trial].T*100, '-r', alpha=0.2, lw=0.5)
ax[3].plot(np.arange(-35, 5)/3,intep1d_dFF_swim_on[CL_trial].T*100, '-k', alpha=0.2, lw=0.5)
plot_shade_err(np.arange(-35, 5)/3, intep1d_dFF_swim_on[CL_trial]*100, axis=0, \
               plt=ax[3], linespec='-k', shadespec='k')
plot_shade_err(np.arange(-35, 5)/3, intep1d_dFF_swim_on[~CL_trial]*100, axis=0, \
               plt=ax[3], linespec='-r', shadespec='r')
ax[3].vlines(0, -1, 10, linestyles='--', colors='k')
ax[3].set_xlim([-35/3, 4/3])
ax[3].set_ylim([ymin, ymax])
ax[3].set_yticks([])

ax[2].plot(intep1d_dFF_pulse[~CL_trial].T*100, '-r', alpha=0.1, lw=0.5)
ax[2].plot(intep1d_dFF_pulse[CL_trial].T*100, '-k', alpha=0.1, lw=0.5)
plot_shade_err(np.arange(0, len_epoch[-1]), intep1d_dFF_pulse[CL_trial]*100, axis=0, \
               plt=ax[2], linespec='-k', shadespec='k')
plot_shade_err(np.arange(0, len_epoch[-1]), intep1d_dFF_pulse[~CL_trial]*100, axis=0, \
               plt=ax[2], linespec='-r', shadespec='r')
ax[2].set_ylim([ymin, ymax])
ax[2].set_yticks([])
ax[2].set_xlim([0, len_epoch[-1]-1])
sns.despine()
plt.show()


