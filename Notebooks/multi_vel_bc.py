from multi_vel_scc import *
from utils import *
from swim_ephys import *
import dask.array as da
from fish_proc.utils import dask_ as fdask


def multi_vel_bar_code(row):
    save_root = row['save_dir']+'/'
    # check ephys data
    dat_dir = row['dat_dir'].replace('/im/', '/')
    dat_dir = dat_dir.replace('/im_CM0/', '/')
    dat_dir = dat_dir.replace('/im_CM1/', '/')
    p_dir = dat_dir + 'processed/'
    ephys_dir = dat_dir + 'ephys/'
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    A = _['A']
    A_loc = _['A_loc']
    num_dff = _['dFF'].shape[-1]
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
    LG_probe_amp = (expt_paradigm.loc['LG probe']['velocity']*100).astype('int')
    LG_probe_gain = expt_paradigm.loc['LG probe']['gain']
    NG_probe_amp = (expt_paradigm.loc['LG probe']['velocity']*100).astype('int')
    NG_probe_gain = expt_paradigm.loc['LG probe']['gain']

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
    
    swim_thres = max(np.percentile(swim_frame, 85), 0.2)
    
    pulse_epoch_on = np.where((epoch_frame[:-1]%5==2) & (epoch_frame[1:]%5==3))[0]
    pulse_epoch_off = np.where((epoch_frame[:-1]%5==3) & (epoch_frame[1:]%5!=3))[0]
    mlt_pulse_trial = []
    mlt_pulse_type = []
    mlt_pulse_vel = []
    mlt_nopulse_trial = []
    mlt_nopulse_type = []
    
    t_post = 50
    t_pre = 5
    
    for n_pen, n_pef in zip(pulse_epoch_on, pulse_epoch_off):
        amp_ = np.percentile(pulse_frame[n_pen:n_pef], 90).astype('int')
        # trial start: n_pen
        # trial end: n_pef
        swim_ = np.clip(swim_frame[n_pen-t_pre:n_pen+t_post]-swim_thres, 0, np.inf)
        if swim_.sum()>0:
            continue
        if amp_==0:
            trial = n_pen+1
            mlt_nopulse_trial.append(trial)
            mlt_nopulse_type.append(epoch_frame[trial]//5)
        else:
            _ = np.where(pulse_frame[n_pen:n_pef]==amp_)[0]
            trial = n_pen+_[0]
            mlt_pulse_trial.append(trial)
            mlt_pulse_type.append(epoch_frame[trial]//5)
            mlt_pulse_vel.append(amp_)
    
    mlt_pulse_vel = np.array(mlt_pulse_vel)
    mlt_pulse_trial = np.array(mlt_pulse_trial)
    mlt_pulse_type = np.array(mlt_pulse_type)
    
    ## OL trials
    OL_mlt_pulse_vel = mlt_pulse_vel[mlt_pulse_type==1]
    OL_mlt_pulse_trial = mlt_pulse_trial[mlt_pulse_type==1]
    
    cell_mvel_stats = dFF_.map_blocks(comp_pulse_stats_chunks, trials=OL_mlt_pulse_trial, conds=OL_mlt_pulse_vel, pre=t_pre, post=t_post, dtype='O').compute() 
    np.savez(save_root+'cell_type_stats_mvel', cell_mvel_stats=cell_mvel_stats, OL_mlt_pulse_vel=OL_mlt_pulse_vel)
    
    fdask.terminate_workers(cluster, client)
    return None