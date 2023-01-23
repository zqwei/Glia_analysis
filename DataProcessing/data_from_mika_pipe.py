import numpy as np
import re
from h5py import File
import matplotlib.pyplot as plt
import fish_proc.wholeBrainDask.cellProcessing_single_WS as fwc
import dask.array as da
import os
import pandas as pd
sub_str = 'im_CM._dff'
temp_str = re.compile(sub_str)

def cell_loc(cell_id):
    thres_ = 0.005
    w_ = (W[cell_id]>thres_)*W[cell_id]
    w_[np.isnan(w_)]=0
    x_loc, y_loc, z_loc = X[cell_id],Y[cell_id],Z[cell_id]
    return (z_loc.dot(w_))/w_.sum(), (x_loc.dot(w_))/w_.sum(), (y_loc.dot(w_))/w_.sum()

# df = pd.read_csv('../Datalists/data_list_in_analysis_slimmed_v3.csv')
df = pd.read_csv('../Datalists/data_list_in_analysis_NE_v1.csv')


for ind, row in df.iterrows():
    # if ind<73:
    #     continue
    print(ind)
    # check if downsample data
    dir_ = row['im_volseg']+'/'    
    if temp_str.search(dir_) is not None:
        print(dir_)
        print('downsample data skip')
        continue
    # check if there is a saving location
    save_root = row['save_dir']
    if '/' not in save_root:
        print(save_root)
        print('invalid save location')
        continue
        
    save_root = save_root +'/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print(save_root)
    if os.path.exists(save_root+'cell_center.npy'):
        continue
    # if os.path.exists(dir_+'cells_clean.hdf5'):
    cell_file = File(dir_+'cells_clean.hdf5', 'r')
    volume_file = File(dir_+'volume.hdf5', 'r')
    X=cell_file['cell_x']
    Y=cell_file['cell_y']
    Z=cell_file['cell_z']
    W=cell_file['cell_weights'][()]
    V=cell_file['volume_weight']
    F = cell_file['cell_timeseries_raw'][()]
    background = cell_file['background'][()]
    brain_map=volume_file['volume_mean'][()]

    # remove background from raw F
    F = F - (background - 10)
    F[F<0]=0
    F_dask = da.from_array(F, chunks=('auto', -1))
    win_ = 400
    baseline_ = da.map_blocks(fwc.baseline, F_dask, dtype='float', window=win_, percentile=20, downsample=10).compute()
    dFF = F/baseline_-1
    
    brain_shape = V.shape
    np.savez(save_root+'cell_dff.npz', \
             dFF=dFF.astype('float16'), \
             brain_shape=brain_shape, \
             X=X, Y=Y, Z=Z, W=W)
    np.save(save_root+'Y_ave.npy', brain_map)
    
    F = cell_file['cell_timeseries_raw']
    
    numCells = F.shape[0]
    A_center = np.zeros((numCells,3))
    for n_cell in range(numCells):
        A_center[n_cell] = cell_loc(n_cell)
    np.save(save_root+'cell_center.npy', A_center)
    