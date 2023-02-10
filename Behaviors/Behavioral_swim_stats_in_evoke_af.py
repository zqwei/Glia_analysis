from utils import *
df = pd.read_csv('../../Datalists/Behavioral_ephys.csv')

def swim_states_evoke(row):
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
    if ((swim_epoch==1).sum()<50) or ((swim_epoch==6).sum()<50):
        return None, None, None, None
    try:
        swim_power_ = [np.median(swim_power[swim_epoch==1]), np.median(swim_power[swim_epoch==6])]
        _, swim_power_stats_ = mannwhitneyu(swim_power[swim_epoch==1], swim_power[swim_epoch==6])
            
        swim_interval_ = [np.median(inter_swim_interval[:-1][(swim_epoch[:-1]==1) & (swim_epoch[1:]==1)]), \
                          np.median(inter_swim_interval[:-1][(swim_epoch[:-1]==6) & (swim_epoch[1:]==6)])]
        _, swim_interval_stats_ = mannwhitneyu(inter_swim_interval[:-1][(swim_epoch[:-1]==1) & (swim_epoch[1:]==1)], \
                                               inter_swim_interval[:-1][(swim_epoch[:-1]==6) & (swim_epoch[1:]==6)])        
        return swim_power_, swim_power_stats_, swim_interval_, swim_interval_stats_
    except:
        return None, None, None, None
    

swim_power_list = []
swim_power_stats_list = []
swim_interval_list = []
swim_interval_stats_list = []
replay_exp = []
gain_level = []
valid_ind = []
ephys_file = 'KA_raw_swim.npz'
for ind, row in df.iterrows():
    save_root = row['save_dir']
    if not type(save_root)==str:
        continue
    if not os.path.exists(save_root+ephys_file):
        continue
    if (not 'G_vs_NGGU' in row['Expt type']) and (not 'replay' in row['Expt type']):
        continue
    swim_power_, swim_power_stats_, swim_interval_, swim_interval_stats_ = swim_states_evoke(row)
    if swim_power_ is None:
        continue
    print(ind, swim_power_[1]/swim_power_[0])
    if 'replay' in row['Expt type']:
        replay_exp.append(True)
    else:
        replay_exp.append(False)
    if 'MG' in row['Expt type']:
        gain_level.append(True)
    else:
        gain_level.append(False)
    swim_power_list.append(swim_power_)
    swim_power_stats_list.append(swim_power_stats_)
    swim_interval_list.append(swim_interval_)
    swim_interval_stats_list.append(swim_interval_stats_)

swim_power_list_ = np.array(swim_power_list)
swim_power_list_ = swim_power_list_/swim_power_list_[:, 0][:, None]
plt.figure(figsize=(2, 5))
sig_ = np.array(swim_power_stats_list)<0.05
sig_ = sig_ & (swim_power_list_[:,1]>0.85)
plt.semilogy(['active', 'passive'], swim_power_list_[sig_].T, '-ok')
sig_ = np.array(swim_power_stats_list)>0.05
sig_ = sig_ & (swim_power_list_[:,1]>0.85)
plt.semilogy(['active', 'passive'], swim_power_list_[sig_].T, '-o', c=[0.5, 0.5, 0.5])
plt.ylabel('Swim power (s)')
plt.ylim([0.5, 100])
sns.despine()
plt.savefig('swim_power_across_fish.pdf')
plt.show()

print(wilcoxon(swim_power_list_[:, 1]-1))
print(np.exp(np.log(swim_power_list_[:, 1]).mean()))


swim_interval_list_ = np.array(swim_interval_list)
# swim_interval_list = swim_interval_list/swim_interval_list[:, 0][:, None]
plt.figure(figsize=(2, 5))
_idx_ = (swim_power_list_[:,1]>0.85) & (swim_interval_list_[:, 0]<3)
sig_ = np.array(swim_interval_stats_list)<0.05
sig_ = sig_ & _idx_
plt.plot(['active', 'passive'], swim_interval_list_[sig_].T, '-ok')
sig_ = np.array(swim_interval_stats_list)>0.05
sig_ = sig_ & _idx_
plt.plot(['active', 'passive'], swim_interval_list_[sig_].T, '-o', c=[0.5, 0.5, 0.5])
plt.ylabel('Time to swim (s)')
sns.despine()
plt.ylim([0, 2.2])
plt.savefig('swim_interval_across_fish.pdf')
plt.show()

print(_idx_.sum())
print(wilcoxon(swim_interval_list_[_idx_, 0], swim_interval_list_[_idx_, 1]))
print((swim_interval_list_[_idx_, 0]-swim_interval_list_[_idx_, 1]).mean())
print((swim_interval_list_[_idx_, 0]-swim_interval_list_[_idx_, 1]).std()/np.sqrt(_idx_.sum()))