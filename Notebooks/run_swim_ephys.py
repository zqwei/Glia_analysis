import numpy as np
import os, sys
from glob import glob
from swim_ephys import *
from utils import *
df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')

for ind, row in df.iterrows():
    # check ephys data
    dat_dir = row['dat_dir'].replace('/im/', '/')
    dat_dir = dat_dir.replace('/im_CM0/', '/')
    dat_dir = dat_dir.replace('/im_CM1/', '/')
    p_dir = dat_dir + 'processed/'
    ephys_dir = dat_dir + '/ephys/'
    save_root = row['save_dir']+'/'
    
    if os.path.exists(save_root+'ephys.npz'):
        continue
    try:
        _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    except:
        print('No data file: '+save_root+'cell_dff.npz')
        continue
    num_dff = _['dFF'].shape[-1]
    _ = None
    
    ###################################
    ## Downsample sensory and motor input to frames
    ###################################
    try:
        ephys_dat = glob(ephys_dir+'/*.10chFlt')[0]
    except:
        print('check ephys file existence in :'+ephys_dir)
        continue
    fileContent_ = load(ephys_dat)
    l_power = windowed_variance(fileContent_[0])[0]
    r_power = windowed_variance(fileContent_[1])[0]
    camtrig = fileContent_[2]
    try:
        expt_meta = glob(dat_dir+'ephys/*end*.xml')[0]
    except:
        print('check ephys-xml file existence in :'+ephys_dir)
        continue
    expt_paradigm = open_ephys_metadata(expt_meta)

    probe_amp = (expt_paradigm.loc['LG probe']['velocity']*100).astype('int')
    probe_gain = expt_paradigm.loc['LG probe']['gain']

    indx = ep2frame(camtrig, thres=3.8)
    frame_ = np.zeros(len(camtrig))
    frame_[indx]=1
    frame_ = frame_.cumsum()
    
    l_power_th = estimate_bot_threshold(l_power, window=180000, lower_percentile=0.01)
    r_power_th = estimate_bot_threshold(r_power, window=180000, lower_percentile=0.01)


    l_power_norm = np.clip(l_power-l_power_th, 0, np.inf)
    r_power_norm = np.clip(r_power-r_power_th, 0, np.inf)
    _power = l_power_norm + r_power_norm
    swim_threshold = 2.0
    
    if os.path.exists(save_root+'swim_thres.npz'):
        swim_json = np.load(save_root+'swim_thres.npz')
        swim_channel = swim_json['swim_channel']
        swim_threshold = swim_json['swim_threshold']
        if 'left' in swim_channel:
            _power = l_power_norm
        if 'right' in swim_channel:
            _power = r_power_norm

    starts, stops, thr = estimate_swims(_power, fs=6000, scaling=swim_threshold)
    starts_ = np.where(starts==1)[0]
    stops_ = np.where(stops==1)[0]

    starts_frame = ind2frame(starts_, indx)
    stops_frame = ind2frame(stops_, indx)
    swim_t_frame = np.vstack([starts_frame,stops_frame])

    frame_len = np.min(np.unique(indx[1:]-indx[:-1]))
    epoch_frame = np.median(wrap_data(fileContent_[5], indx, frame_len), axis=0).astype('int')[:num_dff]
    lswim_frame = np.mean(wrap_data(l_power_norm*10000, indx, frame_len), axis=0)[:num_dff]
    rswim_frame = np.mean(wrap_data(r_power_norm*10000, indx, frame_len), axis=0)[:num_dff]
    pulse_frame = np.rint(np.median(wrap_data(fileContent_[8], indx, frame_len), axis=0)*100).astype('int')[:num_dff]
    visu_frame = np.mean(wrap_data(fileContent_[3], indx, frame_len), axis=0)[:num_dff]
    visu_frame_ = visu_frame.copy()
    visu_frame_[visu_frame_<0]=0
    
    np.savez(save_root+'ephys.npz', \
             probe_amp=probe_amp, \
             probe_gain=probe_gain, \
             swim_t_frame=swim_t_frame, \
             epoch_frame=epoch_frame, \
             lswim_frame=lswim_frame, \
             rswim_frame=rswim_frame, \
             pulse_frame=pulse_frame, \
             visu_frame=visu_frame, \
             visu_frame_=visu_frame_)
#     print('finished: '+save_root)