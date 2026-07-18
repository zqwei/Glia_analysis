'''
Neural dynamics aligned to last pulse before swim onset
'''

from utils import *

df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v8.csv')
dat_save_folder = 'depreciated/_dat_brain_clusters_dynamcis/cluster_dynamics/'
loc = 'IPN'

dFF_epochs = []
CL_trial = []
animal_idx = []

for n_fish in [0, 1, 2]:
    row = df.iloc[n_fish]
    save_root = row['save_dir']+'/'
    dFF_ave = np.load(dat_save_folder + f'fish_{n_fish}_{loc}.npy')

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

    # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    len_ = len(epoch_on)-1
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
        
        evoke_on_ = (epoch_%5==0).sum()
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
            CL_trial.append(CL_trial_)
            dFF_epochs.append(dFF_ave_[last_epoch_on-36:_t_+1])
            animal_idx.append(n_fish)

CL_trial = np.array(CL_trial)
animal_idx = np.array(animal_idx)

len_dFF_epochs = np.array([len(_) for _ in dFF_epochs]).max()
dFF_epochs_ = np.zeros((len(dFF_epochs), len_dFF_epochs))
dFF_epochs_[:] = np.nan
for n in range(len(dFF_epochs)):
    len_ = len(dFF_epochs[n])
    dFF_epochs_[n, :len_] = dFF_epochs[n]


### plot of dynamics
plt.figure(figsize=(4, 3))
plot_shade_err(np.arange(-36, len_dFF_epochs-36)/3, dFF_epochs_[CL_trial]*100, axis=0, \
               plt=plt, linespec='-k', shadespec='k')
plot_shade_err(np.arange(-36, len_dFF_epochs-36)/3, dFF_epochs_[~CL_trial]*100, axis=0, \
               plt=plt, linespec='-r', shadespec='r')
plt.vlines(0, -1, 10, linestyles='--', colors='k')
plt.show()

### pulse response statistics
dFF_epochs_pulse = np.zeros((len(dFF_epochs_), 7))
for n in range(7):
    dFF_epochs_pulse[:, n] = dFF_epochs_[:, n*6:n*6+6].sum(axis=1)-dFF_epochs_[:, n*6+1]*6

plt.figure(figsize=(3, 2))
plot_shade_err(np.arange(7)-6, dFF_epochs_pulse[CL_trial]*100, axis=0, \
               plt=plt, linespec='-k', shadespec='k')
plot_shade_err(np.arange(7)-6, dFF_epochs_pulse[~CL_trial]*100, axis=0, \
               plt=plt, linespec='-r', shadespec='r')
plt.plot(np.arange(7)-6,np.nanmean(dFF_epochs_pulse[CL_trial]*100, axis=0), '-ok')
plt.plot(np.arange(7)-6,np.nanmean(dFF_epochs_pulse[~CL_trial]*100, axis=0), '-or')
plt.vlines(-0.5, 1, 5, linestyles='--', colors='k')
plt.xticks(np.arange(-6, 1))
plt.margins(x=0)
sns.despine()
plt.savefig(f'pulse_int_{loc}.svg')