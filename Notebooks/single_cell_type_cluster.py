from single_cell_class import *
from utils import *
from brain_seg import brain_seg_factor
from factor import thres_factor_
from brain_segment_vis import *
from swim_ephys import *
from kernel_fit import *
from scipy.stats import spearmanr
import dask.array as da
import zarr
from fish_proc.utils.memory import clear_variables
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
    dFF = _['dFF'].astype('float')
    _ = None

    brain_map = np.load(save_root+'Y_ave.npy').astype('float').squeeze()
    A_center = np.load(save_root+'cell_center.npy')
    A_center_grid = np.round(A_center).astype('int')
    cells_in_mask = []

    for n_layer in range(brain_map.shape[0]):
        layer_ = A_center[:, 0]==n_layer
        cell_ids = np.where(layer_)[0]
        mask_ = brain_map[n_layer]>2
        y = A_center_grid[cell_ids, 2]
        x = A_center_grid[cell_ids, 1]
        x_max, y_max = mask_.shape
        num_cells = len(cell_ids)
        in_mask_ = np.zeros(num_cells).astype('bool')
        for n in range(num_cells):
            if (x[n]<x_max) and (y[n]<y_max):
                in_mask_[n] = mask_[x[n], y[n]]
        cells_in_mask.append(cell_ids[in_mask_])
    cells_in_mask = np.concatenate(cells_in_mask)
    A_center = A_center[cells_in_mask]
    dFF = dFF[cells_in_mask]
    num_dff = dFF.shape[-1]

    numCore = 450
    cluster, client = fdask.setup_workers(numCore=numCore,is_local=False)
    fdask.print_client_links(client)
    print(client.dashboard_link)
    if not os.path.exists(save_root+'cell_dff.zarr'):
        dFF_ = zarr.array(dFF, chunks=(dFF.shape[0]//(numCore-2), dFF.shape[1]))
        zarr.save(save_root+'cell_dff.zarr', dFF_)
    dFF_ = da.from_zarr(save_root+'cell_dff.zarr')
    clear_variables(dFF)

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
    
    
    ###################################
    ## Sensory cells for single pulse
    ###################################   
    pulse_trial = []
    pulse_type = []
    pulse_motor_trial = []
    pulse_motor_type = []
    nopulse_trial = []
    nopulse_type = []

    pulse_on = np.where((pulse_frame[:-1]==0) & (pulse_frame[1:]==pulse_amp))[0]+1
    pulse_on = pulse_on[pulse_on<num_dff-10]

    for n, trial in enumerate(pulse_on):
        swim_ = np.clip(swim_frame[trial-2:trial+5]-swim_thres, 0, np.inf)
        if swim_.sum()==0: # remove the trial mixed pulse and motor
            pulse_trial.append(trial)
            pulse_type.append(epoch_frame[trial]//5)
        else:
            pulse_motor_trial.append(trial)
            pulse_motor_type.append(epoch_frame[trial]//5)

    trial_len_ = 7
    for n, trial in enumerate(mlt_nopulse_on):
        for m in range(8):
            swim_ = np.clip(swim_frame[trial+m*trial_len_-2:trial+m*trial_len_+5]-swim_thres, 0, np.inf)
            if swim_.sum()==0:
                nopulse_trial.append(trial+m*trial_len_-2)
                nopulse_type.append(epoch_frame[trial+m*trial_len_]//5)

    if len(pulse_trial)>0:
        cell_sensory_stats = dFF_.map_blocks(pulse_stats_chunks, pulse_trial=pulse_trial, nopulse_trial=nopulse_trial, dtype='O').compute() 
        np.savez(save_root+'cell_type_stats_sensory', cell_sensory_stats=cell_sensory_stats)

    if (len(pulse_motor_trial)>0) and (probe_gain==0):
        cell_pulse_motor_stats = dFF_.map_blocks(pulse_stats_chunks, pulse_trial=pulse_motor_trial, nopulse_trial=nopulse_trial, dtype='O').compute()
        np.savez(save_root+'cell_type_stats_pulse_motor', cell_pulse_motor_stats=cell_pulse_motor_stats)

    if (len(pulse_trial)>0) and (len(pulse_motor_trial)>0) and (probe_gain==0):
        cell_comp_pulse_motor_stats = dFF_.map_blocks(comp_stats_chunks, cond_trial=pulse_trial, comp_trial=pulse_motor_trial, pre=2, post=5, dtype='O').compute()  
        np.savez(save_root+'cell_comp_pulse_motor_stats_', cell_comp_pulse_motor_stats=cell_comp_pulse_motor_stats)

    ###################################
    ## motor cells
    ###################################
    swim_thres = max(np.percentile(swim_frame, 85), 0.2)
    drop_frame = 100
    num_dff = dFF.shape[-1]

    swim_trial = []
    swim_type = []

    noswim_trial = []
    noswim_type = []

    swim_smooth = smooth(swim_frame, gaussKernel(sigma=1.5))
    swim_smooth = swim_smooth>swim_thres
    swim_on = np.where((swim_smooth[:-1]==0) & (swim_smooth[1:]==1))[0]+1
    swim_off = np.where((swim_smooth[:-1]==1) & (swim_smooth[1:]==0))[0]
    swim_on = swim_on[swim_on>drop_frame]
    swim_on = swim_on[swim_on<num_dff-10]
    swim_off = swim_off[swim_off>drop_frame]
    swim_off = swim_off[swim_off<num_dff-10]
    if swim_off[0]<swim_on[0]:
        swim_off = swim_off[1:]
    if swim_on[-1]>swim_off[-1]:
        swim_on = swim_on[:-1]

    if swim_on.shape!=swim_off.shape:
        print('Error in swim matches')

    swim_lens = (swim_on[1:] - swim_on[:-1])
    swim_lens = np.r_[swim_lens, np.inf]
    swim_on = swim_on[swim_lens>=10]
    swim_off = swim_off[swim_lens>=10]
    swim_len = (swim_on[1:] - swim_on[:-1]).min()
    pre_len = 2

    for n, on_ in enumerate(swim_on):
        epoch = epoch_frame[on_-pre_len:on_+swim_len]
        type_ = np.unique(epoch)
        if (len(type_)==1) and ((type_%5<=1).sum()>0):
            swim_trial.append(on_)
            swim_type.append(type_[0])

    off_set = 9
    for n, off_ in enumerate(swim_off[:-1]):
        if swim_on[n+1]-off_<(off_set*2+pre_len+swim_len):
            continue
        epoch = epoch_frame[(off_+off_set):(off_+pre_len+swim_len+off_set*2)]
        type_, cout_ = np.unique(epoch, return_counts=True)
        if (cout_>(swim_len+pre_len)).sum()>0:
            type_ = type_[np.argmax(cout_).astype('int')]
            if type_%5>1:
                continue
            on_ = np.where(epoch==type_)[0][0]
            if on_+off_+off_set>num_dff:
                continue
            noswim_type.append(type_)
            noswim_trial.append(on_+off_+off_set)

    swim_trial_ = np.array(swim_trial)[np.array(swim_type)!=6]
    if len(swim_trial_)>0:
        cell_sm_stats = dFF_.map_blocks(motor_stats_chunks, swim_trial=swim_trial_, noswim_trial=noswim_trial, swim_len=swim_len, pre_len=pre_len, dtype='O').compute()
        np.savez(save_root+'cell_type_stats_sm', cell_sm_stats=cell_sm_stats)

    swim_trial_ = np.array(swim_trial)[np.array(swim_type)==6]
    if len(swim_trial_)>0:
        cell_motor_stats = dFF_.map_blocks(motor_stats_chunks, swim_trial=swim_trial_, noswim_trial=noswim_trial, swim_len=swim_len, pre_len=pre_len, dtype='O').compute() 
        np.savez(save_root+'cell_type_stats_motor', cell_motor_stats=cell_motor_stats)
        
    fdask.terminate_workers(cluster, client)
    return None




    ###################################
    ## passive cells
    ###################################
    swim_thres = max(np.percentile(swim_frame, 85), 0.2)
    active_pulse_trial = []
    passive_pulse_trial = []
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    epoch_on = np.r_[0, epoch_on, len(epoch_frame)]
    t_pre = 5
    t_post = 50
    passive_trial = []
    for n_ in range(len(epoch_on)-1):
        on_ = epoch_on[n_]
        off_ = epoch_on[n_+1]
        swim_ = np.clip(swim_frame[on_:off_]-swim_thres, 0, np.inf)
        epoch_ = epoch_frame[on_:off_]
        pulse_ = pulse_frame[on_:off_]
        if (epoch_%5==3).sum()<120:
            continue
        if swim_[epoch_%5==0].sum()==0:
            continue
        if swim_[epoch_%5==1].sum()==0:
            continue
        swim_len_ = np.where(swim_[epoch_%5==1]>0)[0]
        swim_len_ = swim_len_.max() - swim_len_.min()
        if swim_len_<10:
            continue
        if swim_[epoch_%5==2][5:].sum()>0:
            plt.plot(swim_[epoch_%5==2])
            plt.show()
            continue
        trial_type_ = epoch_[0]//5
        pulse_on_ = np.where(pulse_==pulse_amp)[0]
        if len(pulse_on_)==0: # skip catch trials
            continue
        pulse_on_ = pulse_on_[0]
        if swim_[pulse_on_-t_pre:pulse_on_+t_post].sum()>0:
            continue
        passive_trial.append([trial_type_, pulse_on_+on_])
    passive_trial=np.array(passive_trial).astype('int')

    active_pulse_trial = passive_trial[passive_trial[:,0]==0, 1]
    passive_pulse_trial = passive_trial[passive_trial[:,0]==1, 1]

    if (len(active_pulse_trial)>0) and (len(passive_pulse_trial)>0):
        cell_active_pulse_stats = dFF_.map_blocks(comp_stats_chunks, cond_trial=active_pulse_trial, comp_trial=passive_pulse_trial, pre=t_pre, post=t_post, dtype='O').compute()  
        np.savez(save_root+'cell_active_pulse_stats', cell_active_pulse_stats=cell_active_pulse_stats)

    fdask.terminate_workers(cluster, client)