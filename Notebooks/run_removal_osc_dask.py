import numpy as np
import os, sys
import warnings
warnings.filterwarnings('ignore')
from remove_osc import remove_osc
import pandas as pd
from tqdm import tqdm
from shutil import copyfile


def copyfiles(row):
    save_root = row['save_dir']+'/'
    osc_removal_dir = row['osc_removal_dir']+'/'
    copyfile(save_root+'Y_ave.npy', osc_removal_dir+'Y_ave.npy')
    copyfile(save_root+'trans_affs.npy', osc_removal_dir+'trans_affs.npy')
    copyfile(save_root+'cell_center.npy', osc_removal_dir+'cell_center.npy')


def remove_osc_cells(xs):
    strip_ = xs.mean(axis=0)
    def remove_osc_(x, dataBackground=strip_):
        return remove_osc(x, dataBackground)[None,:]
    import dask.array as da
    xs_dask = da.from_array(xs, chunks=(1,-1))
    return xs_dask.map_blocks(remove_osc_, dtype='float').compute()


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
    print('start correction....')
    z_max, y_max, x_max = A_loc.max(0)
    for z in tqdm(range(z_max+1)):
        for y in range(0, y_max, 200):
            idx = (A_loc[:,0]==z) & (A_loc[:,1]<(y+200)) & (A_loc[:,1]>=y)
            if idx.sum()==0:
                continue
            dFF_removal_osc[idx]=remove_osc_cells(dFF[idx])    
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
        if not os.path.exists(save_dir+'cell_dff.npz'):
            continue
        copyfiles(row)
        if os.path.exists(osc_removal_dir+'cell_dff.npz'):
            continue
        print(osc_removal_dir)
        removal_chunk_osc(row)
        