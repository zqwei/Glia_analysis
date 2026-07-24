'''
Generate the whole-brain (183, 1325, 2509) motor correlation maps
af_motor_pos_cells.npy / af_motor_neg_cells.npy consumed by
Fig_2bcdf_4f_generate_processed_data.py, from the raw per-fish cell data.

Ported from Clusters/depreciated/Figure_3_brain_maps/brain_clusters_af_motor_v2.ipynb.
NOT executed/verified here -- each output is ~4.9GB and this loops over
every motor-correlated cell across all 5 fish, so it's expensive to run.

Run from this src/ folder: `python Fig_2bcdf_4f_generate_motor_maps.py`.
'''

import numpy as np
import pandas as pd

df = pd.read_csv('../../Datalists/data_list_in_analysis_neuron_v8.csv')
brain_map_dir = '/nrs/ahrens/Ziqiang/Jing_Glia_project/brain_maps/'
atlas = np.load('/nrs/ahrens/Ziqiang/Atlas/atlas.npy')


####################################
### per-fish motor correlation and cell locations
####################################

r_list = []
p_list = []
cell_locs = []
for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    cells_center = np.load(save_root+'cell_center_registered.npy')
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    cells_center = cells_center[cell_in_brain]
    _ = np.load(save_root+'cell_motor_corr.npz', allow_pickle=True)
    r_list.append(_['r_cell'])
    p_list.append(_['p_cell'])
    cell_locs.append(cells_center)

r_list = np.concatenate(r_list)
p_list = np.concatenate(p_list)
cell_locs = np.concatenate(cell_locs, axis=0)


####################################
### positive map: spatially-weighted average of significant positive r
####################################

rz, ry, rx = 5, 5, 5
n_list = np.concatenate([cell_locs, r_list[:, None]], axis=1)[(r_list>0) & (p_list<0.01)]
ind_loc = (n_list[:, 1]<atlas.shape[1]-1) & (n_list[:, 1]>0)
ind_loc = ind_loc & (n_list[:, 0]<atlas.shape[0]-1) & (n_list[:, 0]>0)
ind_loc = ind_loc & (n_list[:, 2]<atlas.shape[2]-1) & (n_list[:, 2]>0)
n_list = n_list[ind_loc]
num_cells = ind_loc.sum()
map_corr = np.zeros(atlas.shape)
map_weight = np.zeros(atlas.shape)

for n in range(num_cells):
    z, y, x, corr_ = n_list[n]
    z, x, y = np.array([z, x, y]).astype('int')
    map_corr[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += corr_
    map_weight[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += 1

result_tmp = map_corr/map_weight
result_tmp[map_weight<6] = 0

np.save(brain_map_dir + 'af_motor_pos_cells.npy', result_tmp)


####################################
### negative map: spatially-weighted average of |significant negative r|
####################################

rz, ry, rx = 5, 5, 5
n_list = np.concatenate([cell_locs, r_list[:, None]], axis=1)[(r_list<0) & (p_list<0.01)]
ind_loc = (n_list[:, 1]<atlas.shape[1]-1) & (n_list[:, 1]>0)
ind_loc = ind_loc & (n_list[:, 0]<atlas.shape[0]-1) & (n_list[:, 0]>0)
ind_loc = ind_loc & (n_list[:, 2]<atlas.shape[2]-1) & (n_list[:, 2]>0)
n_list = n_list[ind_loc]
num_cells = ind_loc.sum()
map_corr = np.zeros(atlas.shape)
map_weight = np.zeros(atlas.shape)

for n in range(num_cells):
    z, y, x, corr_ = n_list[n]
    z, x, y = np.array([z, x, y]).astype('int')
    map_corr[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += -corr_
    map_weight[z-rz:z+rz, y-ry:y+ry, x-rx:x+rx] += 1

result_tmp = map_corr/map_weight
result_tmp[map_weight<3] = 0

np.save(brain_map_dir + 'af_motor_neg_cells.npy', result_tmp)
