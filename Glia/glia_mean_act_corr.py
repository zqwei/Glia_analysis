import numpy as np
from scipy.stats import zscore
from scipy.stats import spearmanr
import pandas as pd
import dask
import dask.array as da
from dask.distributed import LocalCluster, Client


def spearmanr_vec(a, b, axis=1):
    c, p = spearmanr(a, b, axis=axis)
    return c[-1][:-1]


def spearmanr_vec_mean(a, vec=np.zeros((1, 100))):
    return spearmanr_vec(a, vec, axis=1)[:, None]


cluster = LocalCluster(n_workers=48, threads_per_worker=1, dashboard_address=':8787')
client = Client(cluster)
df = pd.read_csv('../Datalists/data_list_in_analysis_glia_v3.csv')


for ind, row in df.iterrows():
    if ind%2==0:
        continue
    print(ind)
    save_root = row['save_dir']+'/'
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain]
    num_cells, num_time = dFF_.shape
    time_clip = np.ones(num_time).astype('bool')# for bad procing time overall
    time_clip[:5000] = False
    time_clip[30000:] = False
    dFF_max = dFF_.max(axis=-1)
    valid_dFF_ = (dFF_max>0.2) & (dFF_max<4)
    dFF_ = dFF_[valid_dFF_][:, time_clip]
    dFF_dask = da.from_array(dFF_, chunks=[500, -1]).astype('float')
    zdFF_ = (dFF_dask - dFF_dask.mean(axis=1, keepdims=True))/dFF_dask.std(axis=1, keepdims=True)
    dFF_mean = zdFF_.mean(axis=0).compute()[None, :]
    corr_ = da.map_blocks(spearmanr_vec_mean, dFF_dask, vec=dFF_mean, dtype='float').compute()
    np.savez(save_root+'mean_act_corr.npz', corr_=corr_.squeeze(), valid_dFF_=valid_dFF_)
