'''
Precompute the xy (projection along z) and xz (projection along y) brain
maps used by Figure_2bcd_brain_map.ipynb, so the notebook itself doesn't
need to load the four ~4.9GB per-voxel cell-density volumes directly.

Run from this src/ folder: `python Fig_2bcd_generate_processed_data.py`.
Rerun whenever the underlying brain maps change.
'''

import numpy as np

brain_map_dir = '/nrs/ahrens/Ziqiang/Jing_Glia_project/brain_maps/'

pulse_pos = np.load(brain_map_dir + 'af_pulse_pos_cells.npy')
pulse_pos_xy = np.sqrt(pulse_pos.max(0))
pulse_pos_xz = np.sqrt(pulse_pos.max(1))

# integrative-cell maps are restricted to pulse-position-responsive voxels
int_pos = np.load(brain_map_dir + 'af_int_pos_cells.npy')
int_pos[pulse_pos == 0] = -0.1
int_pos_xy = np.percentile(int_pos, 99, axis=0)
int_pos_xz = np.percentile(int_pos, 99, axis=1)
del pulse_pos, int_pos

int_neg = np.load(brain_map_dir + 'af_int_neg_cells.npy')
int_neg_xy = np.percentile(int_neg, 99, axis=0)
int_neg_xz = np.percentile(int_neg, 99, axis=1)
del int_neg

pulse_neg = np.load(brain_map_dir + 'af_pulse_neg_cells.npy')
pulse_neg_xy = pulse_neg.max(0)
pulse_neg_xz = pulse_neg.max(1)
del pulse_neg

np.savez('../processed_data/Fig_2bcd_brain_map_projections.npz',
         pulse_pos_xy=pulse_pos_xy, pulse_pos_xz=pulse_pos_xz,
         pulse_neg_xy=pulse_neg_xy, pulse_neg_xz=pulse_neg_xz,
         int_pos_xy=int_pos_xy, int_pos_xz=int_pos_xz,
         int_neg_xy=int_neg_xy, int_neg_xz=int_neg_xz)
