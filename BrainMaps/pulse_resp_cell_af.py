import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, uniform_filter
import os
df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v0.csv')


task_type = []
animal = []
cell_loc = []
for ind, row in df.iterrows():
    if ind>5:
        continue
    save_root = row['save_dir']+'/'
    cells_center = np.load(save_root+'cell_center_registered.npy')
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    cells_center = cells_center[cell_in_brain]
    
    _ = np.load(save_root+'cell_active_with_mean.npz', allow_pickle=True)
    CL_res, OL_res = _['CL_res'], _['OL_res']

    res = CL_res
    sig_time = (res<0.05).sum(axis=1)
    cell_idx1 = sig_time>6
    res = OL_res
    sig_time = (res<0.05).sum(axis=1)
    cell_idx2 = sig_time>6
    cell_idx = cell_idx1 | cell_idx2
    
    res = np.load(save_root+'cell_state_pulse.npy')
    sig_time = (res<0.05).sum(axis=1)
    cell_idx_ = sig_time>5
    res = np.load(save_root+'cell_states_no_CT.npz')['pulse_condition_res_no_CT']
    sig_time = (res<0.05).sum(axis=1)
    cell_idx_ = cell_idx_ | (sig_time>5)
    cell_idx = cell_idx & cell_idx_
    np.save(save_root + 'cell_state_pulse_filtered', cell_idx)
    
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

for n_animal in range(6):
    ind = (animal==n_animal) & (x_<atlas.shape[2])
    # print(n_animal, ind.sum())
    if ind.sum()==0:
        continue
    result_anm = np.zeros(atlas.shape)
    result_anm[z_[ind], y_[ind], x_[ind]]=1
    result_anm = gaussian_filter(result_anm, [rz, ry, rx], truncate=1.0)
    result_ = result_ + (result_anm>result_anm.max()*.1).astype('int')

mask_folder = '/nrs/ahrens/Ziqiang/Jing_Glia_project/cell_masks/'
_ = np.savez(mask_folder + 'pulse_state_cell_mask_v0.npz', result_tmp=result_.astype('uint8'))