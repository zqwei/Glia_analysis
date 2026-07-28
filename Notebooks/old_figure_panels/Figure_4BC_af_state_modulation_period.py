'''
Statistics of modulation period across fish
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.5, style='ticks')
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter, uniform_filter
atlas = np.load('/nrs/ahrens/Ziqiang/Atlas/atlas.npy')
df = pd.read_csv('../Datalists/data_list_in_analysis_pulse_cells_v2.csv')

cells_center_ = []
sig_time_sum_ = []
cell_labels_ = []
for ind, row in df.iterrows():
    if ind==7:
        continue
    if ind>8:
        continue
    save_root = row['save_dir']+'/'
    cells_center = np.load(save_root+'cell_center_registered.npy')
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    cells_center = cells_center[cell_in_brain]
    _ = np.load(save_root+'cell_label_pulse_resp_state_af.npz', allow_pickle=True)
    cell_labels = _['cell_labels']
    _ = np.load(save_root + 'cell_state_pulse_long_with_catch.npz', allow_pickle=True)
    p_ = _['p_']
    cell_idx = _['cell_idx']
    sig_time = p_<0.05
    sig_time = convolve1d(sig_time.astype('float'), np.ones(5)/5, axis=1)[:, 2:-2]
    sig_time_sum = (sig_time>0.2).sum(axis=1)
    sig_idx_ = sig_time_sum>5
    if sig_idx_.sum()==0:
        continue
    sig_time_sum_.append(sig_time_sum[sig_idx_]/3)
    cells_center_.append(cells_center[cell_idx][sig_idx_])
    cell_labels_.append(cell_labels[cell_idx][sig_idx_])

cells_center_ = np.concatenate(cells_center_)
sig_time_sum_ = np.concatenate(sig_time_sum_)
cell_labels_ = np.concatenate(cell_labels_)

### brain maps of selected brain clusters
plt.figure(figsize=(8, 6))
plt.imshow(atlas.max(0), cmap='Greys_r')
idx_ = sig_time_sum_>3
plt.scatter(cells_center_[idx_, 2], cells_center_[idx_, 1], c=cell_labels_[idx_], s = 1, \
            vmin=0, vmax=16, cmap=plt.cm.nipy_spectral)
plt.axis('off')
plt.savefig('convergence_time_xy.svg')

plt.figure(figsize=(8, 3))
plt.imshow(atlas.max(1), cmap='Greys_r', origin='lower', aspect='auto')
plt.scatter(cells_center_[idx_, 2], cells_center_[idx_, 0], c=cell_labels_[idx_], s = 1, \
            vmin=0, vmax=16, cmap=plt.cm.nipy_spectral)
plt.axis('off')
plt.savefig('convergence_time_zy.svg')

### Statistics on brain clusters
cell_labels__ = cell_labels_.copy()
# 0: DRN-IPN
cell_labels__[cell_labels_ == 2] = 1 #OT
cell_labels__[cell_labels_ == 4] = 3 # forebrain
# 5: IO
cell_labels__[cell_labels_ == 9] = 5
# 6: PT
cell_labels__[cell_labels_ == 7] = 6
# 8: TL
cell_labels__[cell_labels_ == 12] = 10 # SloMO
cell_labels__[cell_labels_ == 14] = 11 # Cb
#15 AP

idx_ = np.zeros(len(cell_labels_)).astype('bool')
for l in [0, 1, 3, 5, 6, 8, 10, 11, 15]:
    idx_ = idx_ | (cell_labels_==l)

plt.figure(figsize=(6, 4))
sns.violinplot(x=cell_labels_[idx_], y=np.log2(sig_time_sum_[idx_]))
plt.ylim([0, 6])
sns.despine()
# plt.savefig('converagent time.pdf')
plt.show()


### brain maps of modulation period
idx_ = sig_time_sum_>3
cells_center__ = cells_center_[idx_]
sig_time_sum__ = sig_time_sum_[idx_]
z, y, x = cells_center__.T
rz, ry, rx = 2.5, 5, 5
result_ = np.zeros(atlas.shape)
result_cnt = np.zeros(atlas.shape)
num_cell_loc = z.shape[0]
z_, y_, x_ = np.round([z, y, x]).astype('int')
result_[z_, y_, x_] = sig_time_sum__
result_cnt[z_, y_, x_] = 1
result_filter = gaussian_filter(result_, [rz, ry, rx], truncate=1.0)
result_cnt_filter = gaussian_filter(result_cnt, [rz, ry, rx], truncate=1.0)
result_filter[result_cnt_filter>0] = result_filter[result_cnt_filter>0]/result_cnt_filter[result_cnt_filter>0]

plt.figure(figsize=(8, 6))
plt.imshow(atlas.max(0), cmap='gray')
plt.imshow(result_filter.max(0), alpha=0.8, vmax=32, vmin=0) #, cmap=plt.cm.Reds)
plt.axis('off')
plt.savefig('state_period_xy.svg')
plt.show()

plt.figure(figsize=(8, 3))
plt.imshow(atlas.max(1), cmap='gray', aspect='auto',origin='lower')
plt.imshow(result_filter.max(1), alpha=0.8, aspect='auto',origin='lower', vmax=32, vmin=0) #, cmap=plt.cm.Reds)
plt.axis('off')
plt.savefig('state_period_yz.svg')
plt.show()