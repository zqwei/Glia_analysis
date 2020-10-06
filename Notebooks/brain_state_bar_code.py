from brain_state_single_cell_class import *
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
from sensory_motor_single_cell_class import open_ephys_metadata


def brain_state_bar_code(row):
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
    num_dff = dFF.shape[-1]
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
    ## passive cells
    ###################################
    swim_thres = max(np.percentile(swim_frame, 85), 0.2)
    pulse_amp = probe_amp
    active_pulse_trial = []
    passive_pulse_trial = []
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    epoch_on = np.r_[0, epoch_on, len(epoch_frame)]
    t_pre = 5
    t_post = 50
    no_pulse_trial = []
    pulse_trial = []
    no_pulse_swim_trial = []
    pulse_swim_trial = []
    for n_ in range(len(epoch_on)-1):
        on_ = epoch_on[n_]
        off_ = epoch_on[n_+1]
        swim_ = np.clip(swim_frame[on_:off_]-swim_thres, 0, np.inf)
        epoch_ = epoch_frame[on_:off_]
        pulse_ = pulse_frame[on_:off_]
        ## remove the trial without epoch #3
        if (epoch_%5==3).sum()<120:
            continue
        ## remove the trial without swim during rest epoch
        if swim_[epoch_%5==0].sum()==0:
            continue
        ## remove the trial without swim during evoke epoch
        if swim_[epoch_%5==1].sum()==0:
            continue
        trial_type_ = epoch_[0]//5
        # for active trial -- check if the swim continues until the end of the evoke epoch
        evoke_epoch_end = np.where(epoch_%5==1)[0][-1]
        if trial_type_==0:
            swim_len_ = np.where(swim_[epoch_%5<=2]>0)[0]
            if (evoke_epoch_end-swim_len_[-1])>6:
                continue
        
        if trial_type_==1:
            swim_len_ = np.where(swim_[epoch_%5<=2]>0)[0]
            if (evoke_epoch_end-swim_len_[-1])<15:
                continue
        pre_swim = swim_len_[-1]
        if (swim_[pre_swim+1:]>0).sum():
            next_swim = np.where(swim_[pre_swim+1:]>0)[0][0]+pre_swim+1
            next_swim_end = np.where(swim_[next_swim:]==0)[0][0]+next_swim
        else:
            next_swim = np.nan
            next_swim_end = np.nan


        ### no pulse case
        pulse_on_ = np.where(pulse_==pulse_amp)[0]
        if len(pulse_on_)==0: # catch trials -- no pulse case
            pulse_on_ = np.where(epoch_%5==3)[0][0]
            if (next_swim-pulse_on_)<=t_post:
                no_pulse_trial.append([trial_type_, pulse_on_+on_, next_swim+on_])
            else:
                no_pulse_trial.append([trial_type_, pulse_on_+on_, -1])
            if not np.isnan(next_swim):
                no_pulse_swim_trial.append([trial_type_, next_swim+on_, swim_[next_swim:next_swim_end]])
        else: ### pulse case
            pulse_on_ = pulse_on_[0]
            if (next_swim-pulse_on_)<=t_post:
                pulse_trial.append([trial_type_, pulse_on_+on_, next_swim+on_])
            else:
                pulse_trial.append([trial_type_, pulse_on_+on_, -1])
            if not np.isnan(next_swim):
                pulse_swim_trial.append([trial_type_, next_swim+on_, swim_[next_swim:next_swim_end]])

    if (not os.path.exists(save_root+'cell_no_pulse_ap_stats.npz')) and (len(no_pulse_trial)>0):
        trial_ = np.array(no_pulse_trial)
        trial_type_ = trial_[:,0]
        if ((trial_type_==0).sum()>1) and ((trial_type_==1).sum()>1):
            active_ = trial_[trial_type_==0, 1:]
            passive_ = trial_[trial_type_==1, 1:]
            cell_no_pulse_ap_stats = dFF_.map_blocks(comp_pulse_stats_chunks, cond_trial=active_, comp_trial=passive_, pre=t_pre, post=t_post, dtype='O').compute()
            np.savez(save_root+'cell_no_pulse_ap_stats', cell_no_pulse_ap_stats=cell_no_pulse_ap_stats)
        
    if (not os.path.exists(save_root+'cell_pulse_ap_stats.npz')) and (len(pulse_trial)>0):
        trial_ = np.array(pulse_trial)
        trial_type_ = trial_[:,0]
        if ((trial_type_==0).sum()>1) and ((trial_type_==1).sum()>1):
            active_ = trial_[trial_type_==0, 1:]
            passive_ = trial_[trial_type_==1, 1:]
            cell_pulse_ap_stats = dFF_.map_blocks(comp_pulse_stats_chunks, cond_trial=active_, comp_trial=passive_, pre=t_pre, post=t_post, dtype='O').compute()
            np.savez(save_root+'cell_pulse_ap_stats', cell_pulse_ap_stats=cell_pulse_ap_stats)
    
    t_pre_ = 5
    t_post_ = 10
    if (not os.path.exists(save_root+'cell_no_swim_ap_stats.npz')) and (len(no_pulse_swim_trial)>0):
        trial_ = np.array(no_pulse_swim_trial)
        swim_list_thres = np.percentile(np.concatenate(trial_[:,2]), 60)
        valid_ = np.array([(_<=swim_list_thres).mean()==1 for _ in trial_[:,2]])
        trial_ = trial_[valid_]
        trial_type_ = trial_[:,0]
        trial_[:, 2] = -1
        if ((trial_type_==0).sum()>1) and ((trial_type_==1).sum()>1):
            active_ = trial_[trial_type_==0, 1:]
            passive_ = trial_[trial_type_==1, 1:]
            cell_no_swim_ap_stats = dFF_.map_blocks(comp_pulse_stats_chunks, cond_trial=active_, comp_trial=passive_, pre=t_pre_, post=t_post_, dtype='O').compute()
            np.savez(save_root+'cell_no_swim_ap_stats', cell_no_swim_ap_stats=cell_no_swim_ap_stats)
        
    
    if (not os.path.exists(save_root+'cell_pulse_swim_ap_stats.npz')) and (len(pulse_swim_trial)>0):
        trial_ = np.array(pulse_swim_trial)
        swim_list_thres = np.percentile(np.concatenate(trial_[:,2]), 60)
        valid_ = np.array([(_<=swim_list_thres).mean()==1 for _ in trial_[:,2]])
        trial_ = trial_[valid_]
        trial_type_ = trial_[:,0]
        trial_[:, 2] = -1
        if ((trial_type_==0).sum()>1) and ((trial_type_==1).sum()>1):
            active_ = trial_[trial_type_==0, 1:]
            passive_ = trial_[trial_type_==1, 1:]
            cell_pulse_swim_ap_stats = dFF_.map_blocks(comp_pulse_stats_chunks, cond_trial=active_, comp_trial=passive_, pre=t_pre_, post=t_post_, dtype='O').compute()
            np.savez(save_root+'cell_pulse_swim_ap_stats', cell_pulse_swim_ap_stats=cell_pulse_swim_ap_stats)
    
    fdask.terminate_workers(cluster, client)
    return None