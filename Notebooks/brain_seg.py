import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from utils import *


def motor_sensory_ind(row, t_range=(0, np.inf)):
    save_root = row['save_dir']+'/'
    print(row['dat_dir'])
    processed_dir = row['dat_dir'] + 'processed/'
    CL_path = processed_dir + 'CL_trials.pkl'
    OL_path = processed_dir + 'OL_trials.pkl'
    t_min, t_max = t_range
    
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    A = _['A']
    A_loc = _['A_loc']
    dFF = _['dFF'].astype('float')
    _ = None
    
    CL_trials = pd.read_pickle(CL_path)
    OL_trials = pd.read_pickle(OL_path)
    trial_list = [CL_trials, OL_trials]
    epoch_list = []
    trial_type = []
    time_list = []
    for n, _ in enumerate(trial_list):
        for ntrial in range(len(_)):
            epoch_ = _.iloc[ntrial]['epoch_im2ep']
            indx = ep2frame(_.iloc[ntrial]['camtrig_im2ep'], thres=3.8)
            trial_type.append(n)
            epoch_list.append(epoch_[indx])
            time_list.append(_.iloc[ntrial]['ds_trial_inds'])
    sensory_time = []
    motor_time = []
    for n_, _ in enumerate(time_list):
        n = trial_type[n_]
        valid_ = np.where((epoch_list[n_]>=(n*5)) & (epoch_list[n_]<=(n*5+1)))[0]
        motor_time.append(_[valid_.min():valid_.max()+10])
        valid_ = np.where(epoch_list[n_]==(n*5+3))[0]
        sensory_time.append(_[valid_.min()-10:valid_.max()+10])
    
    motor_time = np.concatenate(motor_time)
    sensory_time = np.concatenate(sensory_time)
    motor_time=motor_time[(motor_time>t_min)&(motor_time<t_max)]
    sensory_time=sensory_time[(sensory_time>t_min)&(sensory_time<t_max)]
    
    sensory_valid = WNtest(dFF[:, sensory_time], lags=30)
    sensory_corr = layer_corr(dFF[:, sensory_time], A_loc, corr_thres=0.2, corr_size=10)

    motor_valid = WNtest(dFF[:, motor_time], lags=30)
    motor_corr = layer_corr(dFF[:, motor_time], A_loc, corr_thres=0.2, corr_size=10)
    
    np.savez(save_root+'motor_sensory_ind.npz', \
             motor_valid=motor_valid, \
             motor_corr=motor_corr, \
             sensory_valid=sensory_valid, \
             sensory_corr=sensory_corr)
    
    
    

    