import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma
from scipy.stats import zscore
from scipy.stats import spearmanr
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import curve_fit

sns.set(font_scale=1.5, style='ticks')
df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v8.csv')
cluster_folder = '/nrs/ahrens/Ziqiang/Jing_Glia_project/cell_spatial_clusters_af/'
atlas = np.load('/nrs/ahrens/Ziqiang/Atlas/atlas.npy')


def func(x, a, b, c):
    return a * np.exp(-b * x) + c



cell_locs = []
CL_dFF_list = []
OL_dFF_list = []
CL_dFF_list_ = []
OL_dFF_list_ = []
animal_indx = []

for ind, row in df.iterrows():
    print(ind)
    save_root = row['save_dir']+'/'
    cells_center = np.load(save_root+'cell_center_registered.npy')
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    cell_idx = np.load(save_root + 'cell_state_pulse_filtered.npy')
    cells_center = cells_center[cell_in_brain][cell_idx]
    cell_locs.append(cells_center)
    
    trial_post = 35
    trial_pre = 5
    pre_ext = 0
    swim_thres = 30
    _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
    probe_amp=_['probe_amp']
    swim_t_frame=_['swim_t_frame']
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain][cell_idx]
    len_dff = dFF_.shape[1]
    epoch_frame=_['epoch_frame'][:len_dff]
    pulse_frame=_['pulse_frame'][:len_dff]
    visu_frame=_['visu_frame'][:len_dff]
    lswim_frame=_['lswim_frame'][:len_dff]
    rswim_frame=_['rswim_frame'][:len_dff]
    visu_frame_=_['visu_frame_'][:len_dff]
    
    CL_idx = epoch_frame<=1
    rl = row['rl']
    rr = row['rr']
    if rl >= rr:
        swim_frame_ = lswim_frame
    else:
        swim_frame_ = rswim_frame
    pulse_frame_=pulse_frame.copy()
    pulse_frame_[epoch_frame%5<3]=0
    
    # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    len_ = len(epoch_on)-1
    dFF_trial = []
    swim_mask_trial = []
    CL_trial = []

    for n_ in range(len_):
        on_ = epoch_on[n_]
        off_ = epoch_on[n_+1]-1
        # check if this is a complete trial
        # skip incomplete trials
        if len(np.unique(epoch_frame[on_:off_]))<5:
            continue
        epoch_ = epoch_frame[on_:off_]
        # remove the trials without swim during evoke
        if swim_frame_[on_:off_][epoch_%5==1].sum()==0:
            continue
        swm_ = swim_frame_[on_:off_]
        pulse_ = pulse_frame_[on_:off_]
        CL_trial_ = epoch_[10]//5==0

        # remove OL active trials -- fish swim during pause
        if (not CL_trial_) & (swim_frame_[on_:off_][epoch_%5==2].sum()>0):
            continue

        # remove CL passive trials
        if swim_frame_[on_:off_][epoch_%5==2].sum()==0:
            last_swm = np.where(swim_frame_[on_:off_][epoch_%5<2])[0][-1]
            last_swm = last_swm - (epoch_%5<2).sum()
        else:
            last_swm = 0
        if CL_trial_ & (last_swm<-10):
            continue

        # length of pause -- it seems like some of them are extremely long
        # remove the long pause trial
        if (not CL_trial_) & ((epoch_%5==2).sum()>10):
            continue
        if (CL_trial_) & ((epoch_%5==2).sum()<20):
            continue
        # print((epoch_%5==2).sum())
        if ((epoch_%5==2).sum()>100):
            continue

        # set probe on time
        catch_trial_ = pulse_.sum()==0
        # pause trial
        if catch_trial_:
            # probe_on_ = (epoch_<3).sum()
            continue
        else:
            probe_on_ = np.where(pulse_>0)[0][0]

        probe_on_ = probe_on_+on_

        swm_ = np.cumsum(swim_frame_[probe_on_-trial_pre:probe_on_+trial_post])
        if (swm_>0).sum()>swim_thres:
            continue

        # catch_trial.append(catch_trial_)
        CL_trial.append(CL_trial_)
        dFF_trial.append(dFF_[:, probe_on_-trial_pre:probe_on_+trial_post])
        swim_mask_trial.append(swm_<=0)
        
    dFF_trial = np.array(dFF_trial)
    dFF_trial = dFF_trial.transpose([1, 2, 0])
    dFF_trial_ = dFF_trial - dFF_trial[:, :trial_pre, :].mean(axis=1, keepdims=True)
    swim_mask_trial = np.array(swim_mask_trial).T
    # catch_trial = np.array(catch_trial)
    CL_trial = np.array(CL_trial)
    
    dat_ = dFF_trial[:, :, CL_trial]
    swim_mask_ = swim_mask_trial[:, CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    CL_dFF_list.append(mask_arr.mean(axis=-1))

    dat_ = dFF_trial[:, :, ~CL_trial]
    swim_mask_ = swim_mask_trial[:, ~CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    OL_dFF_list.append(mask_arr.mean(axis=-1))
    
    
    dat_ = dFF_trial_[:, :, CL_trial]
    swim_mask_ = swim_mask_trial[:, CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    CL_dFF_list_.append(mask_arr.mean(axis=-1))

    dat_ = dFF_trial_[:, :, ~CL_trial]
    swim_mask_ = swim_mask_trial[:, ~CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    OL_dFF_list_.append(mask_arr.mean(axis=-1))
    
    animal_indx.append([ind]*dFF_trial.shape[0])

CL_dFF_list = np.concatenate(CL_dFF_list, axis=0)
OL_dFF_list = np.concatenate(OL_dFF_list, axis=0)
CL_dFF_list_ = np.concatenate(CL_dFF_list_, axis=0)
OL_dFF_list_ = np.concatenate(OL_dFF_list_, axis=0)
cell_locs = np.concatenate(cell_locs)
animal_indx_ = np.concatenate(animal_indx)


num_cells = zdat_.shape[0]
int_t = np.zeros(num_cells)

for n in tqdm(range(num_cells)):
    x = pd.Series(zdat_[n]).rolling(window=10, min_periods=1, center=True).quantile(0.1).values
    max_ = x[:trial_pre].mean()
    min_ = x[trial_pre:].min()
    try:
        popt, pcov = curve_fit(func, np.arange(trial_post)/3, x[trial_pre:], p0=(max_-min_, 1/5, min_))
        if popt[1]>0:
            int_t[n] = 1/popt[1]
    except:
        pass

### time constant distribution
plt.hist(int_t[int_t>0], bins=np.arange(100))
plt.show()


rz, ry, rx = 5, 5, 5
result_ = np.zeros(atlas.shape)
n_list =  np.concatenate([cell_locs, int_t[:, None]], axis=1)[int_t>0]
ind_loc = (n_list[:, 1]<atlas.shape[1]-1) & (n_list[:, 1]>0)
ind_loc = ind_loc & (n_list[:, 0]<atlas.shape[0]-1) & (n_list[:, 0]>0)
ind_loc = ind_loc & (n_list[:, 2]<atlas.shape[2]-1) & (n_list[:, 2]>0)
n_list = n_list[ind_loc]
num_cells = ind_loc.sum()
map_corr = np.zeros(atlas.shape)
map_weight = np.zeros(atlas.shape)

for n in range(num_cells):
    z, y, x, corr_ = n_list[n]
    if corr_==0:
        continue
    if corr_>64:
        continue
    z, x, y = np.array([z, x, y]).astype('int')
    map_corr[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += np.log2(corr_)
    map_weight[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += 1

result_tmp = map_corr/map_weight
# result_tmp = np.exp(result_tmp)
result_tmp[map_weight<3] = 0

plt.figure(figsize=(8, 6))
plt.imshow(atlas.max(0), cmap='gray')
plt.imshow(result_tmp.max(0), alpha=0.8, vmax=5, vmin=0)
plt.axis('off')
plt.savefig('int_cell_neg_xy.tiff')

plt.figure(figsize=(8, 3))
plt.imshow(atlas.max(1), cmap='gray', aspect='auto',origin='lower')
plt.imshow(result_tmp.max(1), alpha=0.8, aspect='auto',origin='lower', vmax=5, vmin=0)
plt.axis('off')
plt.savefig('int_cell_neg_xz.tiff')