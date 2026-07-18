'''
Assign each state/pulse-responsive cell (across fish) to a zbrain atlas
region: cells landing outside the labeled brain mask are matched to the
nearest labeled voxel (in parallel over CPU cores), then everything is
saved to pulse_motor_zbrain_labels_refine_dat.npz.
'''

import numpy as np
import multiprocessing as mp
import os, sys
import pandas as pd
from scipy.ndimage import gaussian_filter, uniform_filter
df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v6.csv')
atlas = np.load('/nrs/ahrens/Ziqiang/Atlas/atlas.npy')
cluster_folder = '/nrs/ahrens/Ziqiang/Jing_Glia_project/cell_spatial_clusters_af/'

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def parallel_to_chunks(func1d, arr, *args, **kwargs):
    if mp.cpu_count() == 1:
        raise ValueError('Multiprocessing is not running on single core cpu machines, and consider to change code.')

    mp_count = min(mp.cpu_count(), arr.shape[0]) # fix the error if arr is shorter than cpu counts
    print(f'Number of processes to parallel: {mp_count}')
    chunks = [(func1d, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, mp_count)]
    pool = mp.Pool(processes=mp_count)
    individual_results = pool.map(unpacking_apply_func, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    results = ()
    # print(len(individual_results[0]))
    for i_tuple in range(len(individual_results[0])):
        results = results + (np.concatenate([_[i_tuple] for _ in individual_results]), )
    return results


def unpacking_apply_func(list_params):
    func1d, arr, args, kwargs = list_params
    return func1d(arr, *args, **kwargs)


cell_loc = []
cell_state = []
animal_indx = []
for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    cells_center = np.load(save_root+'cell_center_registered.npy')
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    cells_center = cells_center[cell_in_brain]
    
    # state-dependent cells
    res = np.load(save_root+'cell_state_pulse.npy')
    sig_time = (res<0.05).sum(axis=1)
    cell_idx_ = sig_time>5
    res = np.load(save_root+'cell_states_no_CT.npz')['pulse_condition_res_no_CT']
    sig_time = (res<0.05).sum(axis=1)
    cell_idx_ = cell_idx_ | (sig_time>5)
    
    pvec = np.load(save_root+'cell_states_mean_baseline_pvec.npy')
    baseline_ = np.load(save_root+'cell_states_baseline.npz', allow_pickle=True)['baseline_']    
    res = baseline_
    pvec_ = pvec<0.01
    pvec_ = moving_average(pvec_, 5)
    if (pvec_>0.4).sum()>0:
        ind_ = np.where(pvec_>0.4)[0][-1]+3
    else:
        ind_ = 3
    sig_time = (res[:, ind_:]<0.05).sum(axis=1)
    cell_idx = sig_time>6

    cell_loc.append(cells_center)
    cell_state.append(cell_idx_|cell_idx)
    animal_indx.append(np.ones(len(cells_center))*ind)

cell_state = np.concatenate(cell_state)
cell_loc = np.concatenate(cell_loc)
animal_indx = np.concatenate(animal_indx)

_ = np.load(cluster_folder + 'pulse_motor_zbrain_labels.npz', allow_pickle=True)
labeled_brain = _['labeled_brain']
ind_label = _['ind_label']
mask_labels = _['mask_labels']


z, y, x = cell_loc.astype('int').T
num_cells = len(z)
cell_label = np.zeros(num_cells).astype('int')-1
labeled_brain_ref = np.array(np.where(labeled_brain>=0)).T
labels = labeled_brain[labeled_brain>=0]

for n in range(num_cells):
    if (z[n]>=170) or (z[n]<=0):
        cell_label[n] = -2
        continue
    cell_label[n] = labeled_brain[z[n], y[n], x[n]]
    
ref_locs = labeled_brain_ref[::300]
ref_labels = labels[::300]
def dist_label(arr, ref_locs=ref_locs, ref_labels=ref_labels):
    n_cells = arr.shape[0]
    dist_label_ref = np.zeros((n_cells, 2))
    for n in range(n_cells):
        dist_ = ((arr[n][None, :] - ref_locs)**2).sum(axis=1)
        dist_label_ref[n, 0] = dist_.min()
        dist_label_ref[n, 1] = ref_labels[np.argmin(dist_)]
    return dist_label_ref,

missing_locs = cell_loc[cell_label==-1]
res_ = parallel_to_chunks(dist_label, missing_locs)

dist_ = np.sqrt(res_[0][:, 0])
label_ = res_[0][:, 1]
label_[dist_>50] = -1

cell_label[cell_label==-1] = label_

np.savez_compressed(cluster_folder + 'pulse_motor_zbrain_labels_refine_dat.npz', \
                    cell_state=cell_state.astype('bool'), \
                    cell_loc=cell_loc.astype('float16'), \
                    animal_indx=animal_indx.astype('uint8'), \
                    cell_label=cell_label.astype('int'), \
                    ind_label=ind_label.astype('int'), \
                    mask_labels=mask_labels)