from mp_funcs import *
import pandas as pd
df = pd.read_csv('../Datalists/data_list_in_analysis_NE_v3.csv')

## this code is for an example fish
row = df.iloc[12]
save_root = row['save_dir']+'/'

cell_in_brain = np.load(save_root+'cell_in_brain.npy')
dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain]
num_cells, num_time = dFF_.shape

# clip out the bad times (this depends on data)
print(num_cells, num_time)
time_clip = np.ones(num_time).astype('bool')
time_clip[:5000] = False
time_clip[30000:] = False

# clip out the bad cell/voxels
dFF_max = dFF_.max(axis=-1)
zero_counts = (dFF_==0).mean(axis=1)
valid_dFF_ = (dFF_max>0.2) & (dFF_max<4) & (zero_counts<0.01)
dFF_ = dFF_[valid_dFF_][:, time_clip]

# check data if a boxcar smooth is required for data with oscillations
# num_splits = 200
# dFF_ = parallel_to_chunks(num_splits, smooth_boxcar, dFF_, axis=1)[0].mean(axis=0)

# mean dynamics profile
num_splits = 200
dFF_mean = parallel_to_chunks(num_splits, zscore_, dFF_, axis=1)[0].mean(axis=0)

num_splits = 200
corr_ = parallel_to_chunks(num_splits, spearmanr_vec, dFF_, vec=dFF_mean[None, :], axis=1)[0]

np.savez(save_root+'mean_act_corr.npz', corr_=corr_, valid_dFF_=valid_dFF_)

