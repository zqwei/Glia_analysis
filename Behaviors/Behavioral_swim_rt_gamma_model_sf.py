from utils import *
import pymc as pm
import arviz as az
from scipy.stats import gamma

df = pd.read_csv('../../Datalists/Behavioral_ephys.csv')
row = df.iloc[1]
save_root = row['save_dir']+'/'
print(save_root)

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

#     swim_ons = np.where(np.diff(swim_)==1)[0]+1
#     swim_offs = np.where(np.diff(swim_)==-1)[0]
#     if len(swim_offs[swim_offs<pulse_on])==0: # no swim during evoke epoch
#         continue
#     last_swim_evoke_on = swim_offs[swim_offs<pulse_on][-1]
#     first_swim_evoke_off = swim_ons[(swim_ons>last_swim_evoke_on)]
#     if len(first_swim_evoke_off)>0:
#         first_swim_evoke_off = swim_ons[(swim_ons>last_swim_evoke_on)][0]
#     else:
#         first_swim_evoke_off = len(epoch_)
        
    swim_power.append(swim_frame[on_+pulse_on-6000*20:first_swim_probe_time+6000])
    # swim_power.append(pulse_frame_[trial_start[n]:trial_end[n]]/probe_amp*0.1)
#     pulse_time.append((np.arange(len(epoch_)) - pulse_on)/6000)
#     evoke_off_time.append(evoke_off)
#     swim_end.append(first_swim_evoke_off)
#     last_swim_evoke_on_.append(last_swim_evoke_on)
#     pulse_on_.append(pulse_on)
#     trial_type_.append(epoch_[20]//5)


## Reaction time distribution of RT
plt.figure(figsize=(4, 3))
sns.swarmplot(y=rt_to_pulse, x=trial_type_)
sns.violinplot(y=rt_to_pulse, x=trial_type_, cut=0, inner='quartile')
sns.despine()
plt.xlabel('Evoke type')
plt.ylabel('Reaction time')
# plt.savefig('Reaction time after pulse.pdf')

## Reaction time distribution of RT using Censored Gamma distribution
bw = np.array(rt_to_pulse)[np.array(trial_type_)==1]
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
trace1 = trace.copy()

fig, ax = plt.subplots(figsize = (4, 3))
ax2 = ax.twinx()
ax.hist(bw, density=True, edgecolor='k', facecolor='w', alpha=0.75, bins=np.arange(0.5, 82.5, 4))
az.plot_ppc(trace1, num_pp_samples=100, observed=False, ax=ax2, legend=False, kind='cumulative')
az.plot_ecdf(bw, ax=ax2)
ax.set_yticks(np.arange(0, 0.121, 0.06))
ax2.set_yticks([0, 0.5, 1.0])
ax2.set_ylim([0, 1.01])
ax2.set_xticks(np.arange(0, 100, 25))
ax2.set_xlim([0, 75])
# sns.despine()
# plt.savefig('Dist_fit_reaction_time_2.pdf')
plt.show()




bw = np.array(rt_to_pulse)[np.array(trial_type_)==0]
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
trace0 = trace.copy()

fig, ax = plt.subplots(figsize = (4, 3))
ax2 = ax.twinx()
ax.hist(bw, density=True, edgecolor='k', facecolor='w', alpha=0.75, bins=np.arange(0.5, 82.5, 4))
az.plot_ppc(trace0, num_pp_samples=100, observed=False, ax=ax2, legend=False, kind='cumulative')
az.plot_ecdf(bw, ax=ax2)
ax.set_yticks(np.arange(0, 0.061, 0.03))
ax2.set_yticks([0, 0.5, 1.0])
ax2.set_ylim([0, 1.01])
ax2.set_xticks(np.arange(0, 100, 25))
ax2.set_xlim([0, 75])
# sns.despine()
# plt.savefig('Dist_fit_reaction_time_1.pdf')
plt.show()


alpha = trace1.posterior.alpha.values.flatten()
beta = trace1.posterior.beta.values.flatten()
m1 = alpha * beta
mean1 = alpha.mean()*beta.mean()

alpha = trace0.posterior.alpha.values.flatten()
beta = trace0.posterior.beta.values.flatten()
m2 = alpha * beta
mean2 = alpha.mean()*beta.mean()

mu = np.concatenate([m1, m2])
label_ = np.concatenate([np.ones(m1.shape), np.zeros(m2.shape)])

plt.hist(m1, density=True, bins=100, color='r', edgecolor='none')
plt.hist(m2, density=True, bins=100, color='k', edgecolor='none')
plt.plot([mean1, mean1], [0, 0.1], '--r')
plt.plot([mean2, mean2], [0, 0.1], '--k')
sns.despine()
plt.savefig('Dist_fit_reaction_time_mean.pdf')
plt.show()



## Example swim traces
idx = np.argsort(rt_to_pulse)

plt.figure(figsize=(10, 10))
m = 0
for n in idx:
    if trial_type_[n] == 1:
        continue
    m = m+0.2
    # time_ = pulse_time[n][:swim_end[n]+6000]
    # swim_power_ = swim_power[n][:swim_end[n]+6000]*1000
    # plt.plot(time_, swim_power_+m, '-k')
    time_ = np.arange(len(swim_power[n]))/6000-21
    plt.plot(time_, swim_power[n]*100+m, '-k')
#     plt.plot(time_[evoke_off_time[n]], swim_power_[evoke_off_time[n]]+m, 'or')
#     plt.plot(time_[last_swim_evoke_on_[n]], swim_power_[last_swim_evoke_on_[n]]+m, 'og')
plt.vlines([0], 0, 5, lw=3)
plt.xlim([-20, 85])
# plt.ylim([0, 5])
sns.despine()
# plt.savefig('swim_power_CL.pdf')

plt.figure(figsize=(10, 10))
m = 0
for n in idx:
    if trial_type_[n] == 0:
        continue
    m = m+0.2
#     time_ = pulse_time[n][:swim_end[n]+6000]
#     swim_power_ = swim_power[n][:swim_end[n]+6000]*100
#     plt.plot(time_, swim_power_+m, '-k')
    time_ = np.arange(len(swim_power[n]))/6000-20
    plt.plot(time_, swim_power[n]*100+m, '-k')
#     plt.plot(time_[evoke_off_time[n]], swim_power_[evoke_off_time[n]]+m, 'or')
#     plt.plot(time_[last_swim_evoke_on_[n]], swim_power_[last_swim_evoke_on_[n]]+m, 'og')
plt.vlines([0], 0, 5, lw=3)
plt.xlim([-20, 85])
# plt.ylim([0, 5])
sns.despine()
# plt.savefig('swim_power_OL.pdf')



