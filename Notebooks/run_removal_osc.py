import numpy as np
import os, sys
import warnings
warnings.filterwarnings('ignore')
from remove_osc import remove_osc
import pandas as pd


def removal_chunk_osc(row):
    save_root = row['save_dir']+'/'
    osc_removal_dir = row['osc_removal_dir']+'/'
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    A = _['A']
    A_loc = _['A_loc']
    dFF = _['dFF'].astype('float')
    num_dff = _['dFF'].shape[-1]
    _ = None
    num_cell = dFF.shape[0]
    dFF_removal_osc = dFF.copy()
    
    for cell_id in range(num_cell):
        x = dFF[cell_id].squeeze()
        x_loc = A_loc[cell_id]
        xStrip = (A_loc[:,0]==x_loc[0]) & (A_loc[:,1]<x_loc[1]+40) & (A_loc[:,1]>x_loc[1]-40)
        strip_ = dFF[xStrip].mean(axis=0).squeeze()
        dFF_removal_osc[cell_id] = remove_osc(x, strip_)
    np.savez(osc_removal_dir+'cell_dff.npz', A=A, A_loc=A_loc, dFF=dFF_removal_osc.astype('float16'))


if __name__ == "__main__": 
    df = pd.read_csv('../Processing/data_list_in_analysis.csv')
    for ind, row in df.iterrows():
        save_dir = row['save_dir']+'/'
        osc_removal_dir = row['osc_removal_dir']
        if osc_removal_dir is np.nan:
            continue
        osc_removal_dir = osc_removal_dir+'/'
        if not os.path.exists(osc_removal_dir):
            os.mkdir(osc_removal_dir)
#         if os.path.exists(osc_removal_dir+'cell_dff.npz'):
#             continue
        print(osc_removal_dir)
        removal_chunk_osc(row)
        