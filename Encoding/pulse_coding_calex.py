import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm


df = pd.read_csv('../Datalists/data_list_in_analysis_calex_neuron_v1.csv')
for ind, row in df.iterrows():
    if ind<9:
        continue
    save_root = row['save_dir']+'/'
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF']
    isvalid = np.isnan(dFF_).sum(axis=1)==0
    isvalid = isvalid & (np.isinf(dFF_).sum(axis=1)==0)
    dFF_ = dFF_[isvalid]
    
    _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
    probe_amp=_['probe_amp']
    swim_t_frame=_['swim_t_frame']
    len_dff = min(dFF_.shape[1], len(_['epoch_frame']))
    epoch_frame=_['epoch_frame'][:len_dff]
    pulse_frame=_['pulse_frame'][:len_dff]
    visu_frame=_['visu_frame'][:len_dff]
    lswim_frame=_['lswim_frame'][:len_dff]
    rswim_frame=_['rswim_frame'][:len_dff]
    visu_frame_=_['visu_frame_'][:len_dff]
    dFF_ = dFF_[:, :len_dff]
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
    
    pulse_stim = pulse_frame[epoch_frame==3]
    calcium_kernel = np.exp(-np.arange(0, 6)/0.8)
    pulse_stim = np.convolve(pulse_stim, calcium_kernel)[:-len(calcium_kernel)+1]
    
    splits_ = np.array_split(np.arange(dFF_.shape[0]).astype('int'), dFF_.shape[0]//1000)
    pulse_r, pulse_p = np.zeros(dFF_.shape[0]), np.zeros(dFF_.shape[0])
    t_idx = pulse_stim>5
    for n_split in tqdm(splits_):
        r, p = spearmanr(dFF_[n_split][:, epoch_frame==3][:, t_idx], pulse_stim[None, t_idx], axis=1)
        pulse_r[n_split] = r[-1, :-1]
        pulse_p[n_split] = p[-1, :-1]


    print(ind, ((pulse_r>0.01) & (pulse_p<0.01)).sum())
    
    np.savez(save_root+'pulse_spearmenr.npz', pulse_r=pulse_r, pulse_p=pulse_p)