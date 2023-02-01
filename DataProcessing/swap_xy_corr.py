import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
# df = pd.read_csv('../Datalists/data_list_in_analysis_glia_v3.csv', index_col=0)
# df = pd.read_csv('../Datalists/data_list_in_analysis_NE_v2.csv', index_col=0)
df = pd.read_csv('../Datalists/data_list_in_analysis_slimmed_v4.csv', index_col=0)


for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    im_volseg = row['im_volseg']
    # if 'voluseg' not in im_volseg:
    #     continue
    if not os.path.exists(save_root+'Y_ave.npy'):
        print(ind)
        continue
    brain_map = np.load(save_root+'Y_ave.npy').astype('float')
    _, by, bx = brain_map.squeeze().shape
    cells_center_ = np.load(save_root+'cell_center.npy')
    _, cy, cx = cells_center_[~(np.isnan(cells_center_).sum(axis=-1)>0)].max(axis=0)
    if (by<bx) == (cy<cx):
        print(ind)
        continue
    print(ind, by, bx, cy, cx)
    cells_center = cells_center_.copy()
    cells_center[:, 1] = cells_center_[:, 2]
    cells_center[:, 2] = cells_center_[:, 1]    
    np.save(save_root+'cell_center_voluseg.npy', cells_center_)
    np.save(save_root+'cell_center.npy', cells_center)