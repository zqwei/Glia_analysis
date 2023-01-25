from utils import *
from h5py import File
from scipy.optimize import minimize
from scipy.optimize import Bounds
from fish_proc.utils import dask_ as fdask
from os.path import exists

def filter_resp(swim_input, w_swim, t_pre=0):
    from scipy.signal import convolve
    t_post = len(w_swim)-t_pre-1
    return convolve(swim_input, w_swim)[t_pre:-t_post]
def mse_filter_resp(w_swim, dff, swim_input, lw = 0, t_pre=0, mask_out=None):
    dff_ = filter_resp(swim_input, w_swim, t_pre)
    if mask_out is None:
        return ((dff - dff_)**2).sum() + lw*(w_swim**2).sum()
    else:
        return ((dff[mask_out] - dff_[mask_out])**2).sum() + lw*(w_swim**2).sum()
# def defilter_resp(dFF, swim_input, t_pre, t_post):
#     from scipy.signal import deconvolve
#     return deconvolve(np.r_[np.zeros(t_pre), dFF, np.zeros(t_post)], swim_input)[0]

df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')
cluster, client = fdask.setup_workers(is_local=True)
fdask.print_client_links(client)
print(client.dashboard_link)

ind = 1
row = df.iloc[ind]
save_root = row['save_dir']+'/'
sarr_ = np.load(save_root+'cell_type_stats_msensory_mcluster.npz', allow_pickle=True)['arr_']
marr_ = np.load(save_root+'cell_type_stats_motor_cluster.npz', allow_pickle=True)['arr_']
arr_ = sarr_ | marr_

dFF_ = da.from_zarr(save_root+'cell_dff.zarr')
model_fits = np.zeros(dFF_.shape[0]).astype('O')
cell_dFFs = dFF_[arr_]


_ = np.load(save_root+'ephys.npz', allow_pickle=True)
probe_amp   = _['probe_amp']
probe_gain  = _['probe_gain']
swim_t_frame= _['swim_t_frame']
epoch_frame = _['epoch_frame']
lswim_frame = _['lswim_frame']
rswim_frame = _['rswim_frame']
pulse_frame = _['pulse_frame']
visu_frame  = _['visu_frame']
# visu_frame_ = _['visu_frame_']
swim_channel = np.array2string(_['swim_channel']).replace("'", "")
swim_threshold = _['swim_threshold']
pulse_amp = probe_amp
pulse_frame_ = (pulse_frame==pulse_amp) & (epoch_frame%5==3)
visu_frame_ = -visu_frame.copy()
visu_frame_[epoch_frame%5>=3] = 0
# visu_frame_ = epoch_frame%5<3

swim_frame_power  = lswim_frame + rswim_frame
if 'left' in swim_channel:
    swim_frame_power  = lswim_frame 
if 'right' in swim_channel:
    swim_frame_power  = rswim_frame 

swim_frame_power = swim_frame_power - np.percentile(swim_frame_power, 5)
swim_frame_power[swim_frame_power<0] = 0

visu_frame_fit = -visu_frame.copy()
visu_frame_fit[epoch_frame%5>=3] = 0


t_max = dFF_.shape[1]//3

# fw grating
t_pre_grt = 2
t_post_grt = 40
lw_fw = 10
visu_fit = visu_frame_[:t_max]
mask_out_fw = (epoch_frame%5<3)[:t_max]

# pulse
t_pre_pulse = 2
t_post_pulse = 7
lw_pulse = 100
pulse_frame_fit = pulse_frame_[:t_max]
mask_out_pulse = (epoch_frame%5==3)[:t_max]

# swim
t_pre_swim = 6
t_post_swim = 10
# w_motor = defilter_resp(act_fit, motor_frame_fit, t_pre=t_pre_swim, t_post=t_post_swim)
lw_swim = 10
motor_frame_fit = swim_frame_power[:t_max]
mask_out_swim = motor_frame_fit>np.percentile(motor_frame_fit, 10)

def fit_kernels(cell_dFF_):
    cell_dFF_ = cell_dFF_.squeeze() 
    w_fw = np.r_[np.zeros(t_pre_grt), 1, np.zeros(t_post_grt)]
    w_pulse = np.r_[np.zeros(t_pre_pulse), 1, np.zeros(t_post_pulse)]
    w_motor = np.r_[np.zeros(t_pre_swim), 1, np.zeros(t_post_swim)]
    act_fit = cell_dFF_[:t_max]
    const_ = np.percentile(act_fit, 20)

    # neuronal repsonse == visual + pulse + motor
    for n in range(10):
        # swim filter
        # cell activity - pulse - fw
        dFF_fw = filter_resp(visu_fit, w_fw, t_pre_grt)
        dFF_motor = filter_resp(motor_frame_fit, w_motor, t_pre_swim)
        dFF_pulse = filter_resp(pulse_frame_fit, w_pulse, t_pre_pulse)
        motor_resp_res_ = act_fit - dFF_fw - dFF_pulse - const_
        res_motor = minimize(mse_filter_resp, w_motor, args=(motor_resp_res_, motor_frame_fit, lw_swim, t_pre_swim, mask_out_swim))
        w_motor = res_motor.x
        w_motor[0] = 0
        w_motor[-1] = 0

        # pulse filter
        # cell activity - motor - fw
        dFF_fw = filter_resp(visu_fit, w_fw, t_pre_grt)
        dFF_motor = filter_resp(motor_frame_fit, w_motor, t_pre_swim)
        dFF_pulse = filter_resp(pulse_frame_fit, w_pulse, t_pre_pulse)
        pulse_resp_res_ = act_fit - dFF_motor - dFF_fw - const_
        res_pulse = minimize(mse_filter_resp, w_pulse, args=(pulse_resp_res_, pulse_frame_fit, lw_pulse, t_pre_pulse))
        w_pulse = res_pulse.x
        w_pulse[:t_pre_pulse] = 0
        w_pulse[-1] = 0

        # visual filter -- looking for neural response after explained by other twos
        # cell activity - motor - pulse
        dFF_fw = filter_resp(visu_fit, w_fw, t_pre_grt)
        dFF_motor = filter_resp(motor_frame_fit, w_motor, t_pre_swim)
        dFF_pulse = filter_resp(pulse_frame_fit, w_pulse, t_pre_pulse)
        fw_resp_res_ = act_fit - dFF_motor - dFF_pulse - const_
        res_grt = minimize(mse_filter_resp, w_fw, args=(fw_resp_res_, visu_fit, lw_fw, t_pre_grt))
        w_fw = res_grt.x
        w_fw[:t_pre_grt] = 0


        dFF_fw = filter_resp(visu_fit, w_fw, t_pre_grt)
        dFF_motor = filter_resp(motor_frame_fit, w_motor, t_pre_swim)
        dFF_pulse = filter_resp(pulse_frame_fit, w_pulse, t_pre_pulse)
        const_ = (act_fit - dFF_motor - dFF_pulse - dFF_fw).mean()
        
    dFF_fw = filter_resp(visu_frame_, w_fw)
    dFF_motor = filter_resp(swim_frame_power, w_motor, t_pre_swim)
    dFF_pulse = filter_resp(pulse_frame_, w_pulse)
    cell_est_ = dFF_motor + dFF_pulse + dFF_fw + const_
    ev_ = 1-((cell_est_-cell_dFF_)**2).mean()/cell_dFF_.var()
    returns_ = [w_motor, w_pulse, w_fw, const_, ev_]
    
    return np.array(returns_)[None, :]

num_cpu = 100
chunks_ = cell_dFFs.shape[0]//num_cpu+1
n_split = np.array_split(np.arange(cell_dFFs.shape[0]), chunks_)

for nc in range(chunks_):
    print(f'{nc}/{chunks_}')
    if not exists(f'/scratch/weiz/encoding_tmp/n_block_result_{nc}.npy'):
        n_cell_dFFs = cell_dFFs[n_split[nc]].compute()
        n_cell_dFFs = da.from_array(n_cell_dFFs, chunks=(1, -1))
        n_res_ = n_cell_dFFs.map_blocks(fit_kernels, dtype='O').compute()
        np.save(f'/scratch/weiz/encoding_tmp/n_block_result_{nc}', n_res_)