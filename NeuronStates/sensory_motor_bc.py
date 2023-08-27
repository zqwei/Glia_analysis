from sensory_motor_scc import *
from utils import *
from swim_ephys import *
import dask.array as da
from fish_proc.utils import dask_ as fdask


def sensory_motor_bar_code(row):
    # check the existence of files in run file
    save_root = row['save_dir']+'/' 
    dFF_ = da.from_zarr(save_root+'cell_dff.zarr')
    num_dff = dFF_.shape[-1]
    num_blocks = len(dFF_.chunks[0])
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

    t_post = 30
    t_pre = 5
    for n, trial in enumerate(mlt_pulse_on):
        # swim_ = np.clip(swim_frame[trial-t_pre:trial+t_post]-swim_thres, 0, np.inf)
        swim_ = swim_frame[trial-t_pre:trial+t_post]
        if swim_.sum()==0: # remove the trial mixed pulse and motor
            mlt_pulse_trial.append(trial)
            mlt_pulse_type.append(epoch_frame[trial]//5)

    for n, trial in enumerate(mlt_nopulse_on):
        # swim_ = np.clip(swim_frame[trial-t_pre:trial+t_post]-swim_thres, 0, np.inf)
        swim_ = swim_frame[trial-t_pre:trial+t_post]
        if swim_.sum()==0: # remove the trial mixed pulse and motor
            mlt_nopulse_trial.append(trial)
            mlt_nopulse_type.append(epoch_frame[trial]//5)
    
    cell_stats_list = []
    for n in tqdm(range(num_blocks)):
        dff = dFF_.blocks[n].compute()
        cell_stats = multi_pulse_stats_chunks(dff, pulse_trial=mlt_pulse_trial, nopulse_trial=mlt_nopulse_trial, t_pre=t_pre, t_post=t_post)
        cell_stats_list.append(cell_stats)
        dff = None
    cell_stats_list_ = np.concatenate(cell_stats_list)
    np.savez(save_root+'cell_type_stats_msensory_v2', cell_msensory_stats=cell_stats_list_)
    cell_stats_list_ = None
    cell_stats_list = []
    
    ###################################
    ## motor cells
    ###################################
    swim_trial = []
    swim_type = []
    noswim_trial = []
    noswim_type = []
    swim_on,  swim_off = swim_t_frame
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
        cell_sm_stats_list = []
        for n in tqdm(range(num_blocks)): #
            dff = dFF_.blocks[n].compute()
            cell_stats = motor_stats_chunks(dff, swim_trial=swim_trial_, noswim_trial=noswim_trial, swim_len=swim_len, pre_len=pre_len)
            cell_sm_stats_list.append(cell_stats)
            dff = None
        cell_sm_stats_list_ = np.concatenate(cell_sm_stats_list)
        np.savez(save_root+'cell_type_stats_sm_v2', cell_sm_stats=cell_sm_stats_list_)

    swim_trial_ = np.array(swim_trial)[np.array(swim_type)==6]
    if len(swim_trial_)>0:
        cell_motor_stats_list = []
        for n in tqdm(range(num_blocks)): #
            dff = dFF_.blocks[n].compute()
            cell_stats = motor_stats_chunks(dff, swim_trial=swim_trial_, noswim_trial=noswim_trial, swim_len=swim_len, pre_len=pre_len)
            cell_motor_stats_list.append(cell_stats)
            dff = None
        cell_motor_stats_list_ = np.concatenate(cell_motor_stats_list)
        np.savez(save_root+'cell_type_stats_motor_v2', cell_motor_stats=cell_motor_stats_list_)
    return None

