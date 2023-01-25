from utils import *
import seaborn as sns
from scipy.stats import mannwhitneyu, sem
df = pd.read_csv('../../Datalists/Behavioral_ephys.csv')

def recoveryTimeCompute(row):
    save_root = row['save_dir']+'/'
    _=np.load(save_root+'KA_raw_swim.npz')
    probe_amp=_['probe_amp']
    swim_t_frame=_['swim_t_frame']
    epoch_frame=_['epoch_frame']
    lswim_frame=_['lswim_frame']
    rswim_frame=_['rswim_frame']
    pulse_frame=_['pulse_frame']
    pulse_frame_ = pulse_frame.copy()
    pulse_frame_[epoch_frame%5<=2]=0
    visu_frame=_['visu_frame']
    visu_frame_=_['visu_frame_']

    trial_ = epoch_frame%5 == 0
    trial_start = np.where((~trial_[:-1]) & (trial_[1:]))[0]
    trial_end = np.r_[trial_start, len(trial_)]
    trial_start = np.r_[0, trial_start+1]
    trial_len = len(epoch_frame)
    num_trial = min((trial_start<trial_len).sum(), (trial_end<trial_len).sum())
    trial_end = trial_end[:num_trial]
    trial_start = trial_start[:num_trial]
    swim_frame = lswim_frame+rswim_frame
    swim_index = np.zeros(len(swim_frame)).astype('bool')
    
    # remove too small swims (detected)
    swim_power = []
    for start_, end_ in swim_t_frame.T:
        swim_power.append(swim_frame[start_:end_+1].mean())
    thres_swim_noise = np.percentile(swim_power, 1)
    
    swim_power_mean = []
    swim_power_sum = []
    swim_length = []
    swim_epoch = []
    valid_swim = []
    for start_, end_ in swim_t_frame.T:
        if swim_frame[start_:end_+1].mean()<=thres_swim_noise:
            valid_swim.append(False)
            continue
        swim_index[start_:end_+1] = 1
        swim_power_mean.append(swim_frame[start_:end_+1].mean())  
        swim_power_sum.append(swim_frame[start_:end_+1].sum())  
        swim_length.append((end_+1-start_)/6000)
        swim_epoch.append(np.median(epoch_frame[start_:end_+1]))
        valid_swim.append(True)

    swim_epoch = np.array(swim_epoch).astype('int')
    swim_power_mean = np.array(swim_power_mean)
    swim_power_sum = np.array(swim_power_sum)
    swim_length = np.array(swim_length)
    
    # remove the fish without `struggling` behaviors
    if ((swim_epoch==6).sum()==0) or ((swim_epoch==1).sum()==0):
        return None
    _, p_mean = mannwhitneyu(swim_power_mean[swim_epoch==1], swim_power_mean[swim_epoch==6])
    diff_mean = swim_power_mean[swim_epoch==1].mean()-swim_power_mean[swim_epoch==6].mean()
    _, p_sum = mannwhitneyu(swim_power_sum[swim_epoch==1], swim_power_sum[swim_epoch==6])
    diff_sum = swim_power_sum[swim_epoch==1].mean()-swim_power_sum[swim_epoch==6].mean()
    
    if ((diff_mean>0) and (p_mean<0.05)) and ((diff_sum>0) and (p_sum<0.05)):
        # print(diff_mean, p_mean, diff_sum, p_sum)
        return None
    
    trial_type_ = []
    last_swim_to_evoke_off = []
    rt_to_last_swim = []
    rt_to_pulse = []
    swim_t_frame_valid = swim_t_frame[:, valid_swim]
    resp_to_pulse = []
    epoch_index = []
    for n in range(len(trial_start)):
        # select the trial with pulse
        on_ = trial_start[n]
        off_ = trial_end[n]
        # epoch index in trial
        epoch_ = epoch_frame[on_:off_]
        # swim in trial
        swim_ = swim_index[on_:off_].astype('int')
        # pulse in trial
        pulse_ = pulse_frame_[on_:off_]
        if swim_.sum()==0:
            continue
        if swim_[epoch_%5==1].sum()==0:
            continue
        if swim_[epoch_%5==0].sum()==0:
            continue
        if pulse_.sum()<6000:
            continue

        evoke_off = (epoch_%5<=1).sum()
        pulse_on = np.where(pulse_>0)[0][0]

        # select the trial no swim between evoke off and pulse on
        swim_events = (swim_t_frame_valid[0]>on_+evoke_off) & (swim_t_frame_valid[0]<on_+pulse_on)
        if (swim_events.sum()>0) and (epoch_[evoke_off]>5):
            continue #skip the trial with swimming in pause period after OL

        # find last swim during evoke
        last_swim_evoke_ind = (swim_t_frame_valid[0]<=on_+evoke_off).sum()-1
        trial_type_.append(epoch_[evoke_off]//5)
        last_swim_to_evoke_off.append((swim_t_frame_valid[1, last_swim_evoke_ind]-(on_+evoke_off))/6000)
        first_swim_probe_ind = (swim_t_frame_valid[0]>on_+pulse_on) & (swim_t_frame_valid[0]<off_)

        if first_swim_probe_ind.sum()>0:
            first_swim_probe_time = swim_t_frame_valid[0][first_swim_probe_ind][0]
            rt_to_last_swim.append((first_swim_probe_time-swim_t_frame_valid[1, last_swim_evoke_ind])/6000)
            rt_to_pulse.append((first_swim_probe_time-on_-pulse_on)/6000)
            resp_to_pulse.append(True)
            epoch_index.append(n)
        else:
            first_swim_probe_time = off_
            rt_to_last_swim.append((off_-swim_t_frame_valid[1, last_swim_evoke_ind])/6000)
            rt_to_pulse.append((off_-on_-pulse_on)/6000)
            resp_to_pulse.append(False)
            epoch_index.append(n)
    
    # remove CL trial with passive state
    last_swim_to_evoke_off = np.array(last_swim_to_evoke_off)
    trial_type_ = np.array(trial_type_)
    rt_to_pulse = np.array(rt_to_pulse)
    rt_to_last_swim = np.array(rt_to_last_swim)
    
    invalid_trial = (trial_type_==0) & (last_swim_to_evoke_off<-3)    
    resp_time = rt_to_pulse[~invalid_trial]
    trial_type = trial_type_[~invalid_trial]
    resp_to_pulse = np.array(resp_to_pulse)[~invalid_trial]
    
    return resp_time, trial_type, resp_to_pulse



## Example fish
row = df.iloc[1]
resp_time, trial_type, resp_to_pulse = recoveryTimeCompute(row)
epoch_index = np.arange(len(resp_time))
plt.figure(figsize=(4, 3))
plt.plot(epoch_index[trial_type==1], resp_time[trial_type==1], 'ok')
plt.plot(epoch_index[trial_type==0], resp_time[trial_type==0], 'or')
plt.yticks(np.arange(0, 80, 10))
sns.despine()
plt.savefig('raw_reaction_time_epoch_index.pdf')
plt.show()


## Across fish
replay_exp = []
gain_level = []
rt_corr = []
rt_p=[]
ephys_file = 'KA_raw_swim.npz'
for ind, row in df.iterrows():
    save_root = row['save_dir']
    if not type(save_root)==str:
        continue
    if not os.path.exists(save_root+ephys_file):
        continue
    if (not 'G_vs_NGGU' in row['Expt type']) and (not 'replay' in row['Expt type']):
        continue
    if 'multivel' in save_root:
        continue
    res = recoveryTimeCompute(row)
    if res is None:
        continue
    resp_time, trial_type, resp_to_pulse = res
    trial_thres = 5
    if (((trial_type==0)&(resp_time>0)).sum()<trial_thres) or (((trial_type==1)&(resp_time>0)).sum()<trial_thres):
        continue
    if 'replay' in row['Expt type']:
        replay_exp.append(True)
    else:
        replay_exp.append(False)
    if 'MG' in row['Expt type']:
        gain_level.append(True)
    else:
        gain_level.append(False)
    
    epoch_index = np.arange(len(resp_time))
    corr_, p_ = spearmanr(resp_time[8:], epoch_index[8:])
    rt_corr.append(corr_)
    rt_p.append(p_)
    # print(ind, corr_)
    
    
rt_corr = np.array(rt_corr)
rt_p = np.array(rt_p)
replay_exp = np.array(replay_exp)

plt.figure(figsize=(2, 3))
sns.violinplot(y=rt_corr, x=replay_exp)
sns.swarmplot(y=rt_corr[rt_p<0.01], x=replay_exp[rt_p<0.01], color='k', edgecolor='k', linewidth=1)
sns.swarmplot(y=rt_corr[rt_p>0.01], x=replay_exp[rt_p>0.01], color='w', edgecolor='k', linewidth=1)
plt.ylim([-1, 1])
sns.despine()
plt.savefig('reaction_time_history_af.pdf')
