import numpy as np
import os, sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# df = pd.read_csv('../Datalists/data_list_in_analysis_slimmed_v4.csv', index_col=0)
df = pd.read_csv('../Datalists/data_list_in_analysis_NE_v2.csv', index_col=0)
# df = pd.read_csv('../Datalists/data_list_in_analysis_glia_v3.csv', index_col=0)


for ind, row in df.iterrows():
    if ind>10:
        continue
    save_root = row['save_dir']+'/'
    if os.path.exists(save_root+'cell_in_brain.npy'):
        continue
    if not os.path.exists(save_root+'Y_ave.npy'):
        continue
    brain_map = np.load(save_root+'Y_ave.npy').astype('float')
    brain_thres = np.percentile(brain_map, 60)
    # print(brain_thres)
    brain_map_ = brain_map - brain_thres
    brain_map_[brain_map_<0] =0
    bz, by, bx = brain_map.squeeze().shape
    print(ind, brain_thres)
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF']
    valid_cell = (np.isnan(dFF_).sum(axis=1)==0)
    dFF_ = None
    brain_map = np.load(save_root+'Y_ave.npy').astype('float')
    cells_center = np.load(save_root+'cell_center.npy')
    num_cells = cells_center.shape[0]
    cell_in_brain = np.zeros(num_cells).astype('bool')

    for n_cell in range(num_cells):
        z, y, x = np.floor(cells_center[n_cell]).astype('int')
        if y>=by:
            continue
        if z>=bz:
            continue
        if x>=bx:
            continue
        if (np.array([z, y, x])>0).sum()==3:
            cell_in_brain[n_cell] = (brain_map[z, y, x]>brain_thres)

    cell_in_brain = cell_in_brain & valid_cell
    np.save(save_root+'cell_in_brain.npy', cell_in_brain)
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    cells_center = cells_center[cell_in_brain]
    print(cell_in_brain.mean())