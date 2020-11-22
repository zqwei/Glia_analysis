from sensory_motor_scc import *
from utils import *
from swim_ephys import *
import dask.array as da
from fish_proc.utils import dask_ as fdask


def sensory_motor_bar_code(row):
    save_root = row['save_dir']+'/'
    # check ephys data
    dat_dir = row['dat_dir'].replace('/im/', '/')
    dat_dir = dat_dir.replace('/im_CM0/', '/')
    dat_dir = dat_dir.replace('/im_CM1/', '/')
    p_dir = dat_dir + 'processed/'
    ephys_dir = dat_dir + 'ephys/'
    if not os.path.exists(ephys_dir):
        print('Missing directory')
        print(ephys_dir)
        return None
    
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    A = _['A']
    A_loc = _['A_loc']
    num_dff = _['dFF'].shape[-1]
    # dFF = _['dFF'].astype('float')
    _ = None
    
    
    numCore = 450
    cluster, client = fdask.setup_workers(numCore=numCore,is_local=False)
    fdask.print_client_links(client)
    print(client.dashboard_link)
    dFF_ = da.from_zarr(save_root+'cell_dff.zarr')

    ###################################
    ## Downsample sensory and motor input to frames
    ###################################
    ephys_dat = glob(ephys_dir+'/*.10chFlt')[0]
    fileContent_ = load(ephys_dat)
    l_power = windowed_variance(fileContent_[0])[0]
    r_power = windowed_variance(fileContent_[1])[0]
    camtrig = fileContent_[2]

    expt_meta = glob(dat_dir+'ephys/*end*.xml')[0]
    expt_paradigm = open_ephys_metadata(expt_meta)
    probe_amp = (expt_paradigm.loc['LG probe']['velocity']*100).astype('int')
    probe_gain = expt_paradigm.loc['LG probe']['gain']

    indx = ep2frame(camtrig, thres=3.8)
    frame_ = np.zeros(len(camtrig))
    frame_[indx]=1
    frame_ = frame_.cumsum()

    slide_win = 180000
    r_power_baseline = rolling_perc(r_power, window=slide_win, perc=0.1)
    l_power_baseline = rolling_perc(l_power, window=slide_win, perc=0.1)

    l_power_ = np.clip(l_power-l_power_baseline, 0, None)*10000
    r_power_ = np.clip(r_power-r_power_baseline, 0, None)*10000

    frame_len = np.min(np.unique(indx[1:]-indx[:-1]))

    epoch_frame = np.median(wrap_data(fileContent_[5], indx, frame_len), axis=0).astype('int')[:num_dff]
    swim_frame = np.mean(wrap_data(l_power_, indx, frame_len), axis=0)[:num_dff]
    pulse_frame = np.rint(np.median(wrap_data(fileContent_[8], indx, frame_len), axis=0)*100).astype('int')[:num_dff]
    visu_frame = np.mean(wrap_data(fileContent_[3], indx, frame_len), axis=0)[:num_dff]
    visu_frame_ = visu_frame.copy()
    visu_frame_[visu_frame_<0]=0
    
    ###################################
    ## Swim power threshold for swim detection
    ###################################
    swim_thres = max(np.percentile(swim_frame, 85), 0.2)
    pulse_amp = probe_amp #np.unique(pulse_frame)[1]
    
    ###################################
    ## Sensory cells for multiple pulses
    ###################################
    pulse_epoch_on = np.where((epoch_frame[:-1]%5==2) & (epoch_frame[1:]%5==3))[0]
    pulse_epoch_off = np.where((epoch_frame[:-1]%5==3) & (epoch_frame[1:]%5!=3))[0]
    mlt_pulse_on = []
    mlt_pulse_trial = []
    mlt_pulse_type = []
    mlt_nopulse_on = []
    mlt_nopulse_trial = []
    mlt_nopulse_type = []

    for n_pen, n_pef in zip(pulse_epoch_on, pulse_epoch_off):
        _ = np.where(pulse_frame[n_pen:n_pef]==pulse_amp)[0]
        if len(_)>0:
            mlt_pulse_on.append(n_pen+_[0])
        else:
            mlt_nopulse_on.append(n_pen+1)

    t_post = 50
    t_pre = 5
    for n, trial in enumerate(mlt_pulse_on):
        swim_ = np.clip(swim_frame[trial-t_pre:trial+t_post]-swim_thres, 0, np.inf)
        if swim_.sum()==0: # remove the trial mixed pulse and motor
            mlt_pulse_trial.append(trial)
            mlt_pulse_type.append(epoch_frame[trial]//5)

    for n, trial in enumerate(mlt_nopulse_on):
        swim_ = np.clip(swim_frame[trial-t_pre:trial+t_post]-swim_thres, 0, np.inf)
        if swim_.sum()==0: # remove the trial mixed pulse and motor
            mlt_nopulse_trial.append(trial)
            mlt_nopulse_type.append(epoch_frame[trial]//5)
    
    cell_msensory_stats = dFF_.map_blocks(multi_pulse_stats_chunks, pulse_trial=mlt_pulse_trial, nopulse_trial=mlt_nopulse_trial, t_pre=t_pre, t_post=t_post, dtype='O').compute() 
    np.savez(save_root+'cell_type_stats_msensory', cell_msensory_stats=cell_msensory_stats)
        
    fdask.terminate_workers(cluster, client)
    return None

