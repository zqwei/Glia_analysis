'''
example neural dynamics (single fish) at long trial
'''

from utils import *
from scipy.stats import zscore
from scipy.stats import mannwhitneyu
from scipy.ndimage import gaussian_filter

df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v8.csv')
row = df.iloc[0]
save_root = row['save_dir']+'/'
cells_center = np.load(save_root+'cell_center_registered.npy')
cell_in_brain = np.load(save_root+'cell_in_brain.npy')
cells_center = cells_center[cell_in_brain]

cell_idx = np.load(save_root + 'cell_state_pulse_filtered.npy')
label_sub = np.load(save_root + 'fsensory_cluster_label.npy')
# 9: OT; 10: SLoMO; 4, 8: PT, IO; 
label_ = 10
idx_ = (label_sub[:,0]==label_) & (cells_center[cell_idx, 2]>1100) & (cells_center[cell_idx, 2]<1500)
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
trial_post = 185
_tmp_ = np.array([np.nan]*40) # this is used to marked the trial where recovery swim not happen
recovery_swim = []
recovery_swim_thres = np.percentile(swim_frame_[swim_frame_>0], 0)

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
        probe_on_ = (epoch_%5<3).sum()
        continue
    else:
        probe_on_ = np.where(pulse_>0)[0][0]
        probe_off_ = np.where(pulse_>0)[0][-1]
        # continue

    # probe_on_ = probe_on_+on_
    # print(probe_on_)

    # catch_trial.append(catch_trial_)
    # print(swim_frame_[on_:off_][epoch_%5<2].sum())
    CL_trial.append(CL_trial_)
    dFF_ave_ = dFF_ave[on_:off_]
    dFF_tmp_ = dFF_ave_[probe_on_-5:probe_on_+trial_post]
    _recovery_swim = swm_[epoch_%5==3]>recovery_swim_thres
    if _recovery_swim.sum()>0:
        _t_ = np.where(_recovery_swim)[0][0]
        recovery_swim.append(_t_)
        _t_ = _t_ + (epoch_%5<=2).sum()
        dFF_tmp_[_t_-5:] = np.nan
    dFF_epochs.append(dFF_tmp_)

CL_trial = np.array(CL_trial)
dFF_epochs = np.array(dFF_epochs)

### plot for long trial
plt.figure(figsize=(4, 3))
ymin = 0
ymax = 6
plot_shade_err(np.arange(-5, 185)/3, dFF_epochs[CL_trial]*100, axis=0, \
               plt=plt, linespec='-k', shadespec='k')
plot_shade_err(np.arange(-5, 185)/3, dFF_epochs[~CL_trial]*100, axis=0, \
               plt=plt, linespec='-r', shadespec='r')
plt.vlines(0, -1, 10, linestyles='--', colors='k')
plt.xlim([-5/3, 184/3])
plt.ylim([ymin, ymax])
sns.despine()
plt.show()


### Statistics for modulation period
p_list = np.zeros(trial_post//6)
for n in range(trial_post//6):
    _, p = mannwhitneyu(dFF_epochs[CL_trial, trial_pre+n*6:trial_pre+(n+1)*6].mean(axis=1), \
                        dFF_epochs[~CL_trial, trial_pre+n*6:trial_pre+(n+1)*6].mean(axis=1), nan_policy='omit')
    p_list[n] = p

plt.figure(figsize=(4, 3))
smooth_p_ = gaussian_filter((p_list<0.05).astype('float'), 1)
plt.plot(smooth_p_)
print(np.where(smooth_p_[:-5]>0.5)[0].max())