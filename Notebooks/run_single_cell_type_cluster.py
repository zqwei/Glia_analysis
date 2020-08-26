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


df = pd.read_csv('../Processing/data_list.csv')
row = df.iloc[5]

def bar_code(row):
    save_root = row['save_dir']+'/'
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
        num_cells = len(cell_ids)
        in_mask_ = np.zeros(num_cells).astype('bool')
        for n in range(num_cells):
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
    dat_dir = row['dat_dir'].replace('/im/', '/')
    p_dir = dat_dir + 'processed/'
    ephys_dir = dat_dir + 'ephys/'
    ephys_dat = glob(ephys_dir+'/*.10chFlt')[0]
    fileContent_ = load(ephys_dat)
    l_power = windowed_variance(fileContent_[0])[0]
    r_power = windowed_variance(fileContent_[1])[0]
    camtrig = fileContent_[2]

    expt_meta = glob(dat_dir+'ephys/*end.xml')[0]
    expt_paradigm = open_ephys_metadata(expt_meta)
    probe_amp = expt_paradigm.loc['LG probe']['velocity']

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

    epoch_frame = np.median(wrap_data(fileContent_[5], indx, frame_len), axis=0).astype('int')
    swim_frame = np.mean(wrap_data(l_power_, indx, frame_len), axis=0)
    pulse_frame = np.median(wrap_data(fileContent_[8], indx, frame_len), axis=0).astype('int')
    visu_frame = np.mean(wrap_data(fileContent_[3], indx, frame_len), axis=0)
    visu_frame_ = visu_frame.copy()
    visu_frame_[visu_frame_<0]=0

    ###################################
    ## Sensory cells
    ###################################
    swim_thres = np.percentile(swim_frame, 85)
    pulse_trial = []
    pulse_type = []

    pulse_motor_trial = []
    pulse_motor_type = []

    nopulse_trial = []
    nopulse_type = []

    pulse_amp = probe_amp #np.unique(pulse_frame)[1]
    print(np.unique(pulse_frame)[1], probe_amp)
    pulse_on = np.where((pulse_frame[:-1]==0) & (pulse_frame[1:]==pulse_amp))[0]+1
    num_dff = dFF.shape[-1]
    pulse_on = pulse_on[pulse_on<num_dff-10]

    for n, trial in enumerate(pulse_on):
        swim_ = np.clip(swim_frame[trial-2:trial+5]-swim_thres, 0, np.inf)
        if swim_.sum()==0: # remove the trial mixed pulse and motor
            pulse_trial.append(trial)
            pulse_type.append(epoch_frame[trial]//5)
        else:
            pulse_motor_trial.append(trial)
            pulse_motor_type.append(epoch_frame[trial]//5)

    nopulse_on = epoch_frame%5==4
    nopulse_on = np.where((~nopulse_on[:-1]) & nopulse_on[1:])[0]+1
    nopulse_on = nopulse_on[nopulse_on<num_dff-10]

    for n, trial in enumerate(nopulse_on):
        swim_ = np.clip(swim_frame[trial+3:trial+10]-swim_thres, 0, np.inf)
        if swim_.sum()==0:
            nopulse_trial.append(trial)
            nopulse_type.append(epoch_frame[trial]//5)

    if len(pulse_trial)>0:
        cell_sensory_stats = dFF_.map_blocks(pulse_stats_chunks, pulse_trial=pulse_trial, nopulse_trial=nopulse_trial, dtype='O').compute()  
        np.savez(save_root+'cell_type_stats_sensory', cell_sensory_stats=cell_sensory_stats)

    if len(pulse_motor_trial)>0:
        cell_pulse_motor_stats = dFF_.map_blocks(pulse_stats_chunks, pulse_trial=pulse_motor_trial, nopulse_trial=nopulse_trial, dtype='O').compute()
        np.savez(save_root+'cell_type_stats_pulse_motor', cell_pulse_motor_stats=cell_pulse_motor_stats)

    if (len(pulse_trial)>0) & (len(pulse_motor_trial)>0):
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
    swim_on = swim_on[swim_lens>=7]
    swim_off = swim_off[swim_lens>=7]
    swim_len = (swim_on[1:] - swim_on[:-1]).min()
    pre_len = 2

    for n, on_ in enumerate(swim_on):
        epoch = epoch_frame[on_-pre_len:on_+swim_len]
        type_ = np.unique(epoch)
        if (len(type_)==1) and ((type_%5<=1).sum()>0):
            swim_trial.append(on_)
            swim_type.append(type_[0])

    off_set = 5
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

    ###################################
    ## passive cells
    ###################################

    swim_thres = max(np.percentile(swim_frame, 85), 0.2)
    active_pulse_trial = []
    passive_pulse_trial = []

    active_on = np.where((epoch_frame[1:]==3) & (epoch_frame[:-1]==2))[0]+1
    passive_on = np.where((epoch_frame[1:]==8) & (epoch_frame[:-1]==7))[0]+1
    post_ = 50
    pre_ = 2

    for n, trial in enumerate(active_on):
        swim_ = np.clip(swim_frame[trial-pre_:trial+post_]-swim_thres, 0, np.inf)
        if swim_.sum()==0:
            active_pulse_trial.append(trial)

    for n, trial in enumerate(passive_on):
        swim_ = np.clip(swim_frame[trial-pre_:trial+post_]-swim_thres, 0, np.inf)
        if swim_.sum()==0:
            passive_pulse_trial.append(trial)

    if (len(active_pulse_trial)>0) & (len(passive_pulse_trial)>0):
        cell_active_pulse_stats = dFF_.map_blocks(comp_stats_chunks, cond_trial=active_pulse_trial, comp_trial=passive_pulse_trial, pre=pre_, post=post_, dtype='O').compute()  
        np.savez(save_root+'cell_active_pulse_stats', cell_active_pulse_stats=cell_active_pulse_stats)

    fdask.terminate_workers(cluster, client)