import numpy as np
import os, sys
import pandas as pd
from scipy.ndimage import gaussian_filter, uniform_filter
df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v6.csv')
atlas = np.load('/nrs/ahrens/Ziqiang/Atlas/atlas.npy')

task_type = []
animal = []
cell_loc = []
for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    cells_center = np.load(save_root+'cell_center_registered.npy')
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    cells_center = cells_center[cell_in_brain]
    
    _ = np.load(save_root+'cell_motor_corr.npz')
    p_cell=_['p_cell']
    r_cell=_['r_cell']
    cell_idx = (p_cell<0.05) & (r_cell<-0.05) # negative cells
    # cell_idx = (p_cell<0.05) & (r_cell>0.3) # positive cells
    
    # state-dependent cells
    res = np.load(save_root+'cell_state_pulse.npy')
    sig_time = (res<0.05).sum(axis=1)
    cell_idx_ = sig_time>5
    res = np.load(save_root+'cell_states_no_CT.npz')['pulse_condition_res_no_CT']
    sig_time = (res<0.05).sum(axis=1)
    cell_idx_ = cell_idx_ | (sig_time>5)
    cell_idx = cell_idx & cell_idx_
    
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
# np.savez_compressed(mask_folder+'positive_motor_cells.npz', result_tmp=result_.astype('uint8'))
# np.savez_compressed(mask_folder+'negative_motor_cells.npz', result_tmp=result_.astype('uint8'))
# np.savez_compressed(mask_folder+'positive_motor_state_cells_v6.npz', result_tmp=result_.astype('uint8'))
np.savez_compressed(mask_folder+'negative_motor_state_cells_v6.npz', result_tmp=result_.astype('uint8'))