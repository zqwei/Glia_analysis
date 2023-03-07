import numpy as np
import os, sys
import pandas as pd
from scipy.ndimage import gaussian_filter, uniform_filter

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v0.csv')
task_type = []
animal = []
cell_loc = []
for ind, row in df.iterrows():
    if ind>5:
        continue
    save_root = row['save_dir']+'/'
    pvec = np.load(save_root+'cell_states_mean_baseline_pvec.npy')
    
    cells_center = np.load(save_root+'cell_center_registered.npy')
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    cells_center = cells_center[cell_in_brain]
    _ = np.load(save_root+'cell_states_baseline.npz', allow_pickle=True)
    baseline_= _['baseline_']
    
    res = baseline_
    pvec_ = pvec<0.01
    pvec_ = moving_average(pvec_, 5)

    if (pvec_>0.4).sum()>0:
        ind_ = np.where(pvec_>0.4)[0][-1]+3
    else:
        ind_ = 3
    sig_time = (res[:, ind_:]<0.05).sum(axis=1)
    cell_idx = sig_time>6
    cell_idx = cell_idx
    np.save(save_root + 'cell_state_catch_filtered', cell_idx)

    num_cells = cell_idx.sum()
    cell_loc.append(cells_center[cell_idx])
    
    if 'Replay' in save_root:
        task_ = np.zeros(num_cells).astype('bool')
    else:
        task_ = np.ones(num_cells).astype('bool')
    task_type.append(task_)
    
    animal.append(np.ones(num_cells).astype('int')*ind)


task_type = np.concatenate(task_type)
animal = np.concatenate(animal)
cell_loc = np.concatenate(cell_loc)
z, y, x = cell_loc.T

rz, ry, rx = 2.5, 5, 5
result_ = np.zeros(atlas.shape)
num_cell_loc = z.shape[0]
z_, y_, x_ = np.round([z, y, x]).astype('int')

for n_animal in range(animal.max()+1):
    ind = (animal==n_animal) & (x_<atlas.shape[2])
    if ind.sum()<=100:
        continue
    result_anm = np.zeros(atlas.shape)
    result_anm[z_[ind], y_[ind], x_[ind]]=1
    result_anm = gaussian_filter(result_anm, [rz, ry, rx], truncate=1.0)
    result_ = result_ + (result_anm>result_anm.max()*.1).astype('int')

mask_folder = '/nrs/ahrens/Ziqiang/Jing_Glia_project/cell_masks/'
np.savez_compressed(mask_folder+'baseline_state_cells.npz', result_tmp=result_.astype('uint8'))