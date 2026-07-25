import sys
sys.path.insert(0, '../utils/')
from mp_funcs import *
import pandas as pd
df = pd.read_csv('../Datalists/data_list_in_analysis_NE_v3.csv')
num_splits = 200


for ind, row in df.iterrows():
    if ind<12:
        continue
    save_root = row['save_dir']+'/'
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    _ = np.load(save_root+'mean_act_corr.npz', allow_pickle=True)
    corr_ = _['corr_']
    valid_dFF_ = _['valid_dFF_']
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain]
    num_cells, num_time = dFF_.shape
    time_clip = np.ones(num_time).astype('bool')# for bad procing time overall
    time_clip[:5000] = False
    time_clip[30000:] = False
    dFF_ = dFF_[valid_dFF_][:, time_clip]
    corr_max = corr_.max()
    print((corr_>corr_max*0.5).mean())
    dFF_mean = dFF_[corr_>corr_max*0.5].mean(axis=0)

    corr_ = parallel_to_chunks(num_splits, spearmanr_vec, dFF_, vec=dFF_mean[None, :], axis=1)[0]
    np.savez(save_root+'mean_top_act_corr.npz', corr_=corr_, valid_dFF_=valid_dFF_)

    _ = np.load(save_root+'mean_top_act_corr.npz', allow_pickle=True)
    corr_ = _['corr_']
    valid_dFF_ = _['valid_dFF_']
    corr_thres_ = np.percentile(corr_, 80)
    dFF_mean = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain][valid_dFF_][corr_>corr_thres_].mean(axis=0)
    np.save(save_root+'mean_top_act_dff.npy', dFF_mean)