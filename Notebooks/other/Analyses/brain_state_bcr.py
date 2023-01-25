from brain_state_sccr import *
from utils import *
from swim_ephys import *
import dask.array as da


def brain_state_bar_code_raw(row):
    save_root = row['save_dir']+'/'
    dFF_ = da.from_zarr(save_root+'cell_dff.zarr')
    num_dff = dFF_.shape[-1]
    num_blocks = len(dFF_.chunks[0])
    num_cells = dFF_.shape[0]
    
    _ = np.load(save_root+'ephys.npz', allow_pickle=True)
    probe_amp   = _['probe_amp']
    probe_gain  = _['probe_gain']
    swim_t_frame= _['swim_t_frame']
    epoch_frame = _['epoch_frame']
    lswim_frame = _['lswim_frame']
    rswim_frame = _['rswim_frame']
    pulse_frame = _['pulse_frame']
    visu_frame  = _['visu_frame']
    visu_frame_ = _['visu_frame_']
    swim_channel = np.array2string(_['swim_channel']).replace("'", "")
    swim_threshold = _['swim_threshold']
    pulse_amp = probe_amp
    
#     swim_frame_power  = lswim_frame + rswim_frame
#     swim_frame_med = medfilt(swim_frame_power, 201)
#     swim_frame_power = swim_frame_power - swim_frame_med
#     swim_ons, swim_offs = swim_t_frame
#     swim_frame_ = np.zeros(len(swim_frame_power)).astype('bool')
#     for swim_on, swim_off in zip(swim_ons, swim_offs):
#         swim_frame_[swim_on:swim_off]=1
#     offset_swim_ = np.abs(swim_frame_power[swim_frame_])
#     offset_swim_thres = np.percentile(offset_swim_, 30) #offset_swim_[offset_swim_<np.percentile(offset_swim_, 99.5)].mean()
#     swim_thres = max(np.percentile(swim_frame_power, 90), offset_swim_thres)
#     swim_frame = swim_frame_ | (swim_frame_power>swim_thres)
    swim_frame_power  = lswim_frame + rswim_frame
    if 'left' in swim_channel:
        swim_frame_power  = lswim_frame 
    if 'right' in swim_channel:
        swim_frame_power  = rswim_frame 
    swim_frame_med = medfilt(swim_frame_power, 201)
    swim_frame_power = swim_frame_power - swim_frame_med

    swim_ons, swim_offs = swim_t_frame
    swim_frame_ = np.zeros(len(swim_frame_power)).astype('bool')
    for swim_on, swim_off in zip(swim_ons, swim_offs):
        swim_frame_[swim_on:swim_off]=1
#     offset_swim_ = np.abs(swim_frame_power[swim_frame_])
#     offset_swim_thres = np.percentile(offset_swim_, 30) #offset_swim_[offset_swim_<np.percentile(offset_swim_, 99.5)].mean()
#     swim_thres = max(np.percentile(swim_frame_power, 90), offset_swim_thres)
    swim_frame = swim_frame_.copy()
    

    
    ###################################
    ## passive cells
    ###################################
    active_pulse_trial = []
    passive_pulse_trial = []
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    epoch_on = np.r_[0, epoch_on, len(epoch_frame)]
    t_pre = 5
    t_post = 30
    no_pulse_trial = []
    pulse_trial = []
    no_pulse_swim_trial = []
    pulse_swim_trial = []
    for n_ in range(len(epoch_on)-1):
        on_ = epoch_on[n_]
        off_ = epoch_on[n_+1]
        # swim_ = np.clip(swim_frame[on_:off_]-swim_thres, 0, np.inf)
        swim_ = swim_frame[on_:off_]
        epoch_ = epoch_frame[on_:off_]
        pulse_ = pulse_frame[on_:off_]
        pulse_[epoch_%5<3] = 0 # reset the forward grating to zero before pulse epoch
        ## remove the trial without epoch #3
        if (epoch_%5==3).sum()<t_post:
            continue
        ## remove the trial without swim during evoke epoch
        if swim_[epoch_%5==1].sum()==0:
            continue
        trial_type_ = epoch_[0]//5
        # for active trial -- check if the swim continues until the end of the evoke epoch
        evoke_epoch_end = np.where(epoch_%5==1)[0][-1]
        if trial_type_==0:
            swim_len_ = np.where(swim_[epoch_%5<=2]>0)[0]
            pulse_on_ = np.where(pulse_==pulse_amp)[0]
            if len(pulse_on_)==0:
                if (evoke_epoch_end-swim_len_[-1])>20:
                    continue
            else:
                if (evoke_epoch_end-swim_len_[-1])>12:
                    continue
        
        if trial_type_==1:
            swim_len_ = np.where(swim_[epoch_%5<=2]>0)[0]
            if (evoke_epoch_end-swim_len_[-1])<12: #change it from 14 to 12
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

    if len(no_pulse_trial)>0:
        trial_ = np.array(no_pulse_trial)
        trial_type_ = trial_[:,0]
        if ((trial_type_==0).sum()>1) and ((trial_type_==1).sum()>1):
            active_ = trial_[trial_type_==0, 1:]
            passive_ = trial_[trial_type_==1, 1:]
            cell_no_pulse_ap_stats = []
            for n in tqdm(range(num_blocks)):
                dff = dFF_.blocks[n].compute()
                cell_stats = comp_pulse_stats_chunks(dff, cond_trial=active_, comp_trial=passive_, pre=t_pre, post=t_post)
                cell_no_pulse_ap_stats.append(cell_stats)
                dff = None
            np.savez(save_root+'cell_no_pulse_ap_stats_raw_v2', cell_no_pulse_ap_stats=np.concatenate(cell_no_pulse_ap_stats))
        
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
            cell_pulse_ap_stats = []
            for n in tqdm(range(num_blocks)):
                dff = dFF_.blocks[n].compute()
                cell_stats = comp_pulse_stats_ref_chunks(dff, cond_trial=active_p, comp_trial=passive_p, cond_trial_ref=active_, comp_trial_ref=passive_, pre=t_pre, post=t_post)
                cell_pulse_ap_stats.append(cell_stats)
                dff = None
            np.savez(save_root+'cell_pulse_ap_stats_raw_v2', cell_pulse_ap_stats=np.concatenate(cell_pulse_ap_stats))
    
    return None