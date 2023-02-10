from utils import *
import pymc as pm
import arviz as az
from scipy.stats import gamma
df = pd.read_csv('../../Datalists/Behavioral_ephys.csv')

def reaction_time(row):
    save_root = row['save_dir']+'/'
    _=np.load(save_root+'KA_raw_swim.npz')
    probe_amp=_['probe_amp']
    # probe_gain=_['probe_gain']
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

    swim_power = []
    for start_, end_ in swim_t_frame.T:
        swim_power.append(swim_frame[start_:end_+1].mean())
    thres_swim_noise = np.percentile(swim_power, 1)
    
    swim_power = []
    swim_length = []
    swim_epoch = []
    valid_swim = []
    inter_swim_interval = []
    num_swim = swim_t_frame.shape[1] - 1

    for n in range(num_swim):
        start_, end_ = swim_t_frame[:, n]
        start_next = swim_t_frame[0, n+1]
        if swim_frame[start_:end_+1].mean()<=thres_swim_noise:
            valid_swim.append(False)
            continue
        swim_index[start_:end_+1] = 1
        swim_power.append(swim_frame[start_:end_+1].mean())   
        swim_length.append((end_+1-start_)/6000)
        swim_epoch.append(np.median(epoch_frame[start_:end_+1]))
        valid_swim.append(True)
        inter_swim_interval.append((start_next+1-end_)/6000)

    swim_epoch = np.array(swim_epoch).astype('int')
    swim_power = np.array(swim_power)
    swim_length = np.array(swim_length)
    inter_swim_interval = np.array(inter_swim_interval)

    trial_type_ = []
    last_swim_to_evoke_off = []
    rt_to_last_swim = []
    rt_to_pulse = []
    swim_t_frame_valid = swim_t_frame[:, :-1][:, valid_swim]
    swim_power = []

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
        if swim_events.sum()>0:
            continue #skip the trial with swimming in pause period

        # find last swim during
        last_swim_evoke_ind = (swim_t_frame_valid[0]<=on_+evoke_off).sum()-1
        if swim_t_frame_valid[1, last_swim_evoke_ind]<on_+evoke_off-6*6000:
            if epoch_[evoke_off]<5:
                continue
        trial_type_.append(epoch_[evoke_off]//5)
        last_swim_to_evoke_off.append((swim_t_frame_valid[1, last_swim_evoke_ind]-(on_+evoke_off))/6000)
        first_swim_probe_ind = (swim_t_frame_valid[0]>on_+pulse_on) & (swim_t_frame_valid[0]<off_)

        if first_swim_probe_ind.sum()>0:
            first_swim_probe_time = swim_t_frame_valid[0][first_swim_probe_ind][0]
            rt_to_last_swim.append((first_swim_probe_time-swim_t_frame_valid[1, last_swim_evoke_ind])/6000)
            rt_to_pulse.append((first_swim_probe_time-on_-pulse_on)/6000)
        else:
            first_swim_probe_time = off_
            rt_to_last_swim.append((off_-swim_t_frame_valid[1, last_swim_evoke_ind])/6000)
            rt_to_pulse.append((off_-on_-pulse_on)/6000)

    return np.array(rt_to_pulse), np.array(trial_type_)


def model_fit_censored_gamma(bw):
    uthres_ = np.floor(bw.max())
    lthres_ = 1
    bw[bw>uthres_]=uthres_
    bw[bw<lthres_] = lthres_
    #### using scipy estimate of alpha beta as a inital guess
    mu = bw.mean()
    sigma = bw.std()
    a = (mu/sigma)**2
    b = mu/(sigma**2)
    with pm.Model() as model:
        alpha = pm.Exponential('alpha', a)
        beta = pm.Exponential('beta', b)
        gamma_dist = pm.Gamma.dist(alpha=alpha, beta=beta)
        gamma_dist = pm.Censored("gamma_dist", gamma_dist, \
                                      lower=lthres_, upper=uthres_, \
                                      observed=bw)
        trace = pm.sample(1000)
    with model:
        trace.extend(pm.sample_posterior_predictive(trace))
    return trace.posterior.alpha.values.flatten(), trace.posterior.beta.values.flatten()


num_trials = 10
ephys_file = 'KA_raw_swim.npz'
for ind, row in df.iterrows():
    if ind<17:
        continue
    print(f'{ind}/{len(df)}')
    save_root = row['save_dir']
    if not type(save_root)==str:
        continue
    if not os.path.exists(save_root+ephys_file):
        continue
    if (not 'G_vs_NGGU' in row['Expt type']) and (not 'replay' in row['Expt type']):
        continue

    # rt_to_pulse, trial_type_ = reaction_time(row)
    # np.savez(save_root + 'behavioral_rt.npz', rt_to_pulse=rt_to_pulse, trial_type_=trial_type_)
    _ = np.load(save_root + 'behavioral_rt.npz')
    rt_to_pulse=_['rt_to_pulse']
    trial_type_=_['trial_type_']
    valid_ = ((trial_type_==0).sum()>num_trials) & ((trial_type_==1).sum()>num_trials) 
    if len(rt_to_pulse)>0:
        valid_ = valid_ & (rt_to_pulse.max()>5)

    if valid_:
        try:
            bw = rt_to_pulse[trial_type_==0]
            alpha0, beta0 = model_fit_censored_gamma(bw)
        except:
            print('failed fitting')
            continue
        try:
            bw = rt_to_pulse[trial_type_==1]
            alpha1, beta1 = model_fit_censored_gamma(bw)
        except:
            print('failed fitting')
            continue
            
        np.savez(save_root + 'behavioral_rt_model.npz', alpha0=alpha0, beta0=beta0, \
                 alpha1=alpha1, beta1=beta1)



## plots
num_trials = 12
recovery_time = []
valid_ind = []
replay_exp = []
gain_level = []
ephys_file = 'KA_raw_swim.npz'
valid_fish = []
for ind, row in df.iterrows():
    save_root = row['save_dir']
    if not type(save_root)==str:
        continue
    if not os.path.exists(save_root+ephys_file):
        continue
    if (not 'G_vs_NGGU' in row['Expt type']) and (not 'replay' in row['Expt type']):
        continue
    valid_ind.append(ind)
    if 'replay' in row['Expt type']:
        replay_exp.append(True)
    else:
        replay_exp.append(False)
    if 'MG' in row['Expt type']:
        gain_level.append(True)
    else:
        gain_level.append(False)
    
    _ = np.load(save_root + 'behavioral_rt.npz')
    rt_to_pulse=_['rt_to_pulse']
    trial_type_=_['trial_type_']
    valid_ = ((trial_type_==0).sum()>num_trials) & ((trial_type_==1).sum()>num_trials)
    if len(rt_to_pulse)>0:
        rt_max = np.floor(rt_to_pulse.max())
        valid_ = valid_ & (rt_to_pulse.max()>5)
    if not valid_:
        valid_fish.append(valid_)
    else:
        if not os.path.exists(save_root + 'behavioral_rt_model.npz'):
            valid_fish.append(False)
            continue
        valid_fish.append(True)
        _=np.load(save_root + 'behavioral_rt_model.npz')
        alpha0=_['alpha0']
        beta0=_['beta0']
        alpha1=_['alpha1']
        beta1=_['beta1']
        p = ((alpha0*beta0)[:1500]<(alpha1*beta1)[:1500]).mean()
        p = min(1-p, p)/2
        print(ind, (alpha0*beta0).mean(), (alpha1*beta1).mean(), p)
        recovery_time.append([(alpha0*beta0).mean(), (alpha1*beta1).mean(), p])

valid_fish = np.array(valid_fish)
recovery_time = np.array(recovery_time)
replay_exp = np.array(replay_exp)[valid_fish]
gain_level = np.array(gain_level)[valid_fish]

recovery_time_ = recovery_time[:, :2] - recovery_time[:, 0][:, None]
p = recovery_time[:, -1]
# p[recovery_time_[:, -1]>5] = 0.049

plt.figure(figsize=(2, 3))
plt.plot(['active', 'passive'], recovery_time[~replay_exp & (p<0.05), :2].T, '-ok')
plt.plot(['active', 'passive'], recovery_time[replay_exp & (p<0.05), :2].T, '-og')
plt.plot(['active', 'passive'], recovery_time[~replay_exp & (p>0.05), :2].T, '-ok', markerfacecolor='none')
plt.plot(['active', 'passive'], recovery_time[replay_exp & (p>0.05), :2].T, '-og', markerfacecolor='none')
plt.ylabel('Time to swim (s)')
plt.yticks([0, 50, 100])
sns.despine()
plt.savefig('response_time_est.pdf')
plt.show()

plt.figure(figsize=(2, 3))
plt.plot(['active', 'passive'], recovery_time_[~replay_exp & (p<0.05)].T, '-ok')
plt.plot(['active', 'passive'], recovery_time_[replay_exp & (p<0.05)].T, '-og')
plt.plot(['active', 'passive'], recovery_time_[~replay_exp & (p>0.05)].T, '-ok', markerfacecolor='none')
plt.plot(['active', 'passive'], recovery_time_[replay_exp & (p>0.05)].T, '-og', markerfacecolor='none')
plt.ylabel('Time to swim (s)')
plt.yticks([0, 20, 40, 60])
sns.despine()
plt.savefig('response_time_diff.pdf')
plt.show()

plt.figure(figsize=(2, 3))
sns.boxplot(x=replay_exp, y=recovery_time_[:,1])
sns.swarmplot(x=replay_exp, y=recovery_time_[:,1], color='k')
sns.despine()
plt.yticks([0, 20, 40, 60])
plt.savefig('response_time_diff_hist.pdf')
plt.show()