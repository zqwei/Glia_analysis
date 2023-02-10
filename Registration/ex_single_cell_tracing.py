from utils import *
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from swc import from_swc_array
from glob import glob

## on single-cell reference brain
sbrain_map_ = imread('/nrs/ahrens/Ziqiang/Atlas/Single_cell/T_AVG_jf5Tg.tif')
z_, y_, x_ = sbrain_map_.shape
_ = np.load('/nrs/ahrens/Ziqiang/Atlas/Single_cell/single_cells_morph.npz', allow_pickle=True)
arr_list = _['swc_arr']
somas = _['somas']
f_list = []
for _ in arr_list:
    f_list.append(from_swc_array(_))

colors = np.array([plt.cm.Dark2(_)[:3] for _ in range(8)])
fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(sbrain_map_.max(0), cmap=plt.cm.Greys_r) #, alpha=0.3
for n, f in enumerate(f_list[0:4000:100]):
    x, y, z = somas[n]
    f.plot(fig=fig, ax=ax, color=colors[n%8], linewidth=3, axis='z')
    ax.scatter(x, y, s=60, c='k')
ax.set_axis_off()
plt.show()


## on registered brain
atlas_path = r'/groups/ahrens/ahrenslab/jing/zebrafish_atlas/yumu_confocal/20150519/im/cy14_1p_stitched.h5'
atlas = np.swapaxes(read_h5(atlas_path, dset_name='channel0'),1,2).astype('float64')[::-1]

_ = np.load('/nrs/ahrens/Ziqiang/Atlas/Single_cell/single_cells_morph_reg.npz', allow_pickle=True)
arr_list = _['arr_list_new']
somas = _['somas_new']
f_list = []
for _ in arr_list:
    f_list.append(from_swc_array(_))

fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(atlas.max(0).T, cmap=plt.cm.Greys_r) #, alpha=0.3
for n, f in enumerate(f_list[0:4000:100]):
    x, y, z = somas[n]
    f.plot(fig=fig, ax=ax, color=colors[n%8], linewidth=3, axis='z')
    ax.scatter(x, y, s=60, c='k')
ax.set_axis_off()
plt.show()
