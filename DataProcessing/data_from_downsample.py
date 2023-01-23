import numpy as np
from h5py import File
import matplotlib.pyplot as plt
import dask
import dask.array as da
import os
import pandas as pd
from glob import glob
df = pd.read_csv('../Datalists/data_list_in_analysis_slimmed_v3.csv')

def imread(v, baseline=100):
    img = File(v,'r')['func'][()]
    img[np.isnan(img)] = 0
    img = img - baseline
    img[img<0] = 0
    return img

for n, row in df.iterrows():
    if n<73:
        continue
    im_dir = row['im_volseg']
    save_root = row['save_dir']
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    im_ds_files = im_dir.replace('im_CM0_dff_cis.zarr', 'processed/')+'im_CM0_dff_background_stack.h5'
    brain_map = imread(im_ds_files)
    brain_map[np.isnan(brain_map)] = 0
    brain_map[brain_map<0] = 0
    np.save(save_root+'Y_ave.npy', brain_map)


for n, row in df.iterrows():
    if n<73:
        continue
    im_dir = row['im_volseg']
    save_root = row['save_dir']
    brain_map = np.load(save_root+'Y_ave.npy').astype('float')
    dff_data = da.from_zarr(im_dir).compute().astype('float16')
    thres_ = np.percentile(brain_map, 60)
    mask = brain_map>thres_
    dFF = dff_data[:,mask].T # cells x time
    print(dFF.shape)
    np.savez(save_root+'cell_dff.npz', dFF=dFF.astype('float16'))

    z, x, y = np.where(mask)
    A_center = np.hstack([z[None, :].T, x[None, :].T, y[None, :].T])
    np.save(save_root+'cell_center.npy', A_center)

    dff_data = None
    dFF = None
    mask = None