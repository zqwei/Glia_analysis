from brain_state_sccr import *
from utils import *
from swim_ephys import *
import dask.array as da
from fish_proc.utils import dask_ as fdask


def brain_state_pvi_bar_code_raw(row):
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
    _ = None

    
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
    # frame_len = np.min(np.unique(indx[1:]-indx[:-1]))
    frame_len = np.round(np.median(indx[1:]-indx[:-1])).astype('int')
    epoch_frame = np.median(wrap_data(fileContent_[5], indx, frame_len), axis=0).astype('int')[:num_dff]
    pulse_frame = np.rint(np.median(wrap_data(fileContent_[8], indx, frame_len), axis=0)*100).astype('int')[:num_dff]
    visu_frame = np.mean(wrap_data(fileContent_[3], indx, frame_len), axis=0)[:num_dff]
    visu_frame_ = visu_frame.copy()
    visu_frame_[visu_frame_<0]=0
        
    l_power_th = estimate_bot_threshold(l_power, window=180000, lower_percentile=0.01)
    r_power_th = estimate_bot_threshold(r_power, window=180000, lower_percentile=0.01)
    l_power_norm = np.clip(l_power-l_power_th, 0, np.inf)
    r_power_norm = np.clip(r_power-r_power_th, 0, np.inf)
    _power = l_power_norm + r_power_norm
    starts, stops, thr = estimate_swims(_power, fs=6000, scaling=2.0)
    starts_ = np.where(starts==1)[0]
    stops_ = np.where(stops==1)[0]
    starts_frame = ind2frame(starts_, indx)
    stops_frame = ind2frame(stops_, indx)+1
    swim_t_frame = np.vstack([starts_frame,stops_frame])
    # get frame with swim
    swim_frame_binary = np.zeros(len(epoch_frame)).astype('bool')
    for n_start, n_stop in zip(starts_frame, stops_frame):
        swim_frame_binary[n_start:n_stop]=True

    
    ###################################
    ## passive cells
    ###################################
    pulse_amp = probe_amp
    active_pulse_trial = []
    passive_pulse_trial = []
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    epoch_on = np.r_[0, epoch_on, len(epoch_frame)]
    t_pre = 5
    t_post = 120
    no_pulse_trial = []
    pulse_trial = []
    no_pulse_swim_trial = []
    pulse_swim_trial = []
    for n_ in range(len(epoch_on)-1):
        on_ = epoch_on[n_]
        off_ = epoch_on[n_+1]
        swim_ = swim_frame_binary[on_:off_]
        epoch_ = epoch_frame[on_:off_]
        pulse_ = pulse_frame[on_:off_]
        vis_ = visu_frame_[on_:off_]
        ## remove the trial without epoch #3
        if (epoch_%5==3).sum()<120:
            continue
        ## remove the trial with swim during the trial
        if swim_.sum()>0:
            continue
        trial_type_ = epoch_[0]//5
        # for active trial -- check if the swim continues until the end of the evoke epoch
        evoke_epoch_end = np.where(epoch_%5==1)[0][-1]
        if (trial_type_==0) and (vis_[epoch_%5==1].sum()==0):
            continue
        if trial_type_==1 and (vis_[epoch_%5==1].sum()>0):
            continue
        
        pulse_on_ = np.where(pulse_==pulse_amp)[0]
        if len(pulse_on_)==0: # catch trials -- no pulse case
            pulse_on_ = np.where(epoch_%5==3)[0][0]
            no_pulse_trial.append([trial_type_, pulse_on_+on_, -1])
        else: ### pulse case
            pulse_on_ = pulse_on_[0]
            pulse_trial.append([trial_type_, pulse_on_+on_, -1])

    numCore = 450
    cluster, client = fdask.setup_workers(numCore=numCore,is_local=False)
    fdask.print_client_links(client)
    print(client.dashboard_link)
    dFF_ = da.from_zarr(save_root+'cell_dff.zarr')
    
    if len(no_pulse_trial)>0:
        trial_ = np.array(no_pulse_trial)
        trial_type_ = trial_[:,0]
        if ((trial_type_==0).sum()>1) and ((trial_type_==1).sum()>1):
            active_ = trial_[trial_type_==0, 1:]
            passive_ = trial_[trial_type_==1, 1:]
            cell_no_pulse_ap_stats = dFF_.map_blocks(comp_pulse_stats_chunks, cond_trial=active_, comp_trial=passive_, pre=t_pre, post=t_post, dtype='O').compute()
            np.savez(save_root+'cell_no_pulse_ap_stats_raw', cell_no_pulse_ap_stats=cell_no_pulse_ap_stats)
        
    if len(pulse_trial)>0:
        if len(no_pulse_trial)>0:
            trial_ = np.array(no_pulse_trial)
            trial_type_ = trial_[:,0]
            if (trial_type_==0).sum()>=1:
                active_ = trial_[trial_type_==0, 1:]
            else:
                active_ = None
            if (trial_type_==1).sum()>=1:
                passive_ = trial_[trial_type_==1, 1:]
            else:
                passive_ = None
        else:
            active_ = None
            passive_ = None
        
        trial_ = np.array(pulse_trial)
        trial_type_ = trial_[:,0]
        if ((trial_type_==0).sum()>1) and ((trial_type_==1).sum()>1):
            active_p = trial_[trial_type_==0, 1:]
            passive_p = trial_[trial_type_==1, 1:]
            cell_pulse_ap_stats = dFF_.map_blocks(comp_pulse_stats_ref_chunks, cond_trial=active_p, comp_trial=passive_p, cond_trial_ref=active_, comp_trial_ref=passive_, pre=t_pre, post=t_post, dtype='O').compute()
            np.savez(save_root+'cell_pulse_ap_stats_raw', cell_pulse_ap_stats=cell_pulse_ap_stats)
    
    fdask.terminate_workers(cluster, client)
    return None