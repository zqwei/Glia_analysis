from utils import *
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

plt.figure(figsize=(4, 3))
sns.ecdfplot(swim_power[swim_epoch==1]*20000, log_scale=True)
sns.ecdfplot(swim_power[swim_epoch==6]*20000, log_scale=True)
sns.despine()
plt.xlabel('swim power per bout')
plt.ylabel('Cumulative Prob.')
# plt.savefig('swim_power_duration_evoke.pdf')
plt.show()
# plt.close('all')

print(np.median(swim_power[swim_epoch==1]), np.median(swim_power[swim_epoch==6]))
print(mannwhitneyu(swim_power[swim_epoch==1], swim_power[swim_epoch==6]))

plt.figure(figsize=(4, 3))
sns.ecdfplot(inter_swim_interval[:-1][(swim_epoch[:-1]==1) & (swim_epoch[1:]==1)])
sns.ecdfplot(inter_swim_interval[:-1][(swim_epoch[:-1]==6) & (swim_epoch[1:]==6)])
sns.despine()
plt.xlim([0, 3])
plt.xlabel('swim length per bout (s)')
plt.ylabel('Cumulative Prob.')
plt.savefig('swim_interval_evoke.pdf')
plt.show()
plt.close('all')

print(inter_swim_interval[:-1][(swim_epoch[:-1]==1) & (swim_epoch[1:]==1)].mean(), inter_swim_interval[:-1][(swim_epoch[:-1]==6) & (swim_epoch[1:]==6)].mean())

print(mannwhitneyu(inter_swim_interval[:-1][(swim_epoch[:-1]==1) & (swim_epoch[1:]==1)], inter_swim_interval[:-1][(swim_epoch[:-1]==6) & (swim_epoch[1:]==6)]))