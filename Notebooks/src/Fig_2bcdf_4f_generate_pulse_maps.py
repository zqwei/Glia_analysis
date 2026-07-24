'''
Generate the whole-brain (183, 1325, 2509) pulse-position correlation maps
af_pulse_pos_cells.npy / af_pulse_neg_cells.npy consumed by
Fig_2bcdf_4f_generate_processed_data.py, from the raw per-fish cell data.

Ported from Clusters/depreciated/Figure_3_brain_maps/brain_clusters_af_pulse.ipynb.
NOT executed/verified here -- each output is ~4.9GB and this loops over
every pulse-responsive cell across all 5 fish, so it's expensive to run.

One deliberate deviation from that source notebook: its own cells for the
positive map reload result_tmp from the already-saved af_pulse_pos_cells.npy
right after computing it, then just re-save the reload (the freshly
computed correlation map is never used) -- almost certainly leftover
interactive-session checkpoint behavior rather than intentional. This
script skips that reload and saves the freshly computed map directly.
Review before running for the first time.

Run from this src/ folder: `python Fig_2bcdf_4f_generate_pulse_maps.py`.
'''

import numpy as np
import pandas as pd
from scipy.stats import zscore, spearmanr

df = pd.read_csv('../../Datalists/data_list_in_analysis_neuron_v8.csv')
brain_map_dir = '/nrs/ahrens/Ziqiang/Jing_Glia_project/brain_maps/'
atlas = np.load('/nrs/ahrens/Ziqiang/Atlas/atlas.npy')


####################################
### per-fish trial-averaged dFF and cell locations
####################################

cell_locs = []
CL_dFF_list = []
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
    swim_thres = 30
    _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain][cell_idx]
    len_dff = dFF_.shape[1]
    epoch_frame=_['epoch_frame'][:len_dff]
    pulse_frame=_['pulse_frame'][:len_dff]
    lswim_frame=_['lswim_frame'][:len_dff]
    rswim_frame=_['rswim_frame'][:len_dff]

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

        # remove the long pause trial
        if (not CL_trial_) & ((epoch_%5==2).sum()>10):
            continue
        if (CL_trial_) & ((epoch_%5==2).sum()<20):
            continue
        if ((epoch_%5==2).sum()>100):
            continue

        # set probe on time; skip catch (pause) trials
        catch_trial_ = pulse_.sum()==0
        if catch_trial_:
            continue
        else:
            probe_on_ = np.where(pulse_>0)[0][0]

        probe_on_ = probe_on_+on_

        swm_ = np.cumsum(swim_frame_[probe_on_-trial_pre:probe_on_+trial_post])
        if (swm_>0).sum()>swim_thres:
            continue

        CL_trial.append(CL_trial_)
        dFF_trial.append(dFF_[:, probe_on_-trial_pre:probe_on_+trial_post])
        swim_mask_trial.append(swm_<=0)

    dFF_trial = np.array(dFF_trial).transpose([1, 2, 0])
    swim_mask_trial = np.array(swim_mask_trial).T
    CL_trial = np.array(CL_trial)

    import numpy.ma as ma
    dat_ = dFF_trial[:, :, CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_trial[:, CL_trial], dat_.shape)
    mask_arr = ma.masked_array(dat_, mask=~swim_mask_)
    CL_dFF_list.append(mask_arr.mean(axis=-1))

    animal_indx.append([ind]*dFF_trial.shape[0])

CL_dFF_list = np.concatenate(CL_dFF_list, axis=0)
cell_locs = np.concatenate(cell_locs)
animal_indx_ = np.concatenate(animal_indx)


####################################
### correlate each cell's dFF against the pulse-stim regressor
### (regressor timing is taken from the last fish processed above,
### matching the source notebook's own behavior)
####################################

zdat_ = zscore(CL_dFF_list, axis=-1)

pulse_stim = pulse_frame_[probe_on_-trial_pre:probe_on_+trial_post]
calcium_kernel = np.exp(-np.arange(0, 6)/0.8)
pulse_stim = np.convolve(pulse_stim, calcium_kernel)[:-len(calcium_kernel)+1]

splits_ = np.array_split(np.arange(zdat_.shape[0]).astype('int'), zdat_.shape[0]//1000)
pulse_r, pulse_p = np.zeros(zdat_.shape[0]), np.zeros(zdat_.shape[0])
for n_split in splits_:
    r, p = spearmanr(zdat_[n_split], pulse_stim[None, :], axis=1)
    pulse_r[n_split] = r[-1, :-1]
    pulse_p[n_split] = p[-1, :-1]


####################################
### positive map: spatially-weighted average of positive pulse_r
####################################

rz, ry, rx = 5, 5, 5
n_list = np.concatenate([cell_locs, pulse_r[:, None]], axis=1)[pulse_r>0]
ind_loc = (n_list[:, 1]<atlas.shape[1]-1) & (n_list[:, 1]>0)
ind_loc = ind_loc & (n_list[:, 0]<atlas.shape[0]-1) & (n_list[:, 0]>0)
ind_loc = ind_loc & (n_list[:, 2]<atlas.shape[2]-1) & (n_list[:, 2]>0)
n_list = n_list[ind_loc]
num_cells = ind_loc.sum()
map_corr = np.zeros(atlas.shape)
map_weight = np.zeros(atlas.shape)

for n in range(num_cells):
    z, y, x, corr_ = n_list[n]
    if corr_<0:
        continue
    z, x, y = np.array([z, x, y]).astype('int')
    map_corr[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += corr_
    map_weight[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += 1

result_tmp = map_corr/map_weight
result_tmp[map_weight<3] = 0

np.save(brain_map_dir + 'af_pulse_pos_cells.npy', result_tmp)


####################################
### negative map: spatially-weighted average of |negative pulse_r|
####################################

rz, ry, rx = 5, 10, 10
n_list = np.concatenate([cell_locs, pulse_r[:, None]], axis=1)[pulse_r<0]
ind_loc = (n_list[:, 1]<atlas.shape[1]-1) & (n_list[:, 1]>0)
ind_loc = ind_loc & (n_list[:, 0]<atlas.shape[0]-1) & (n_list[:, 0]>0)
ind_loc = ind_loc & (n_list[:, 2]<atlas.shape[2]-1) & (n_list[:, 2]>0)
n_list = n_list[ind_loc]
num_cells = ind_loc.sum()
map_corr = np.zeros(atlas.shape)
map_weight = np.zeros(atlas.shape)

for n in range(num_cells):
    z, y, x, corr_ = n_list[n]
    if corr_>0:
        continue
    z, x, y = np.array([z, x, y]).astype('int')
    map_corr[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += -corr_
    map_weight[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += 1

result_tmp = map_corr/map_weight
result_tmp[map_weight<4] = 0

np.save(brain_map_dir + 'af_pulse_neg_cells.npy', result_tmp)
