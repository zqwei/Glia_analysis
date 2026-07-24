'''
Precompute the xy (projection along z) and xz (projection along y) brain
maps used by Figure_2bcdf_4f_brain_map.ipynb, so the notebook itself
doesn't need to load the per-voxel cell-density volumes (~4.9GB each)
directly.

Run from this src/ folder: `python Fig_2bcdf_4f_generate_processed_data.py`.
Rerun whenever the underlying brain maps change.
'''

import numpy as np

brain_map_dir = '/nrs/ahrens/Ziqiang/Jing_Glia_project/brain_maps/'

# Fig 2b
pulse_pos = np.load(brain_map_dir + 'af_pulse_pos_cells.npy')
pulse_pos_xy = np.sqrt(pulse_pos.max(0))
pulse_pos_xz = np.sqrt(pulse_pos.max(1))

# Fig 2c -- restricted to pulse-position-responsive voxels
int_pos = np.load(brain_map_dir + 'af_int_pos_cells.npy')
int_pos[pulse_pos == 0] = -0.1
int_pos_xy = np.percentile(int_pos, 99, axis=0)
int_pos_xz = np.percentile(int_pos, 99, axis=1)
del pulse_pos, int_pos

# Fig 2d
pulse_neg = np.load(brain_map_dir + 'af_pulse_neg_cells.npy')
pulse_neg_xy = pulse_neg.max(0)
pulse_neg_xz = pulse_neg.max(1)
del pulse_neg

# Fig 2f
motor_pos = np.load(brain_map_dir + 'af_motor_pos_cells.npy')
motor_pos_xy = motor_pos.max(0)
motor_pos_xz = motor_pos.max(1)
del motor_pos

# Fig 4f
motor_neg = np.load(brain_map_dir + 'af_motor_neg_cells.npy')
motor_neg_xy = motor_neg.max(0)
motor_neg_xz = motor_neg.max(1)
del motor_neg

np.savez('../processed_data/Fig_2bcdf_4f_brain_map_projections.npz',
         pulse_pos_xy=pulse_pos_xy, pulse_pos_xz=pulse_pos_xz,
         pulse_neg_xy=pulse_neg_xy, pulse_neg_xz=pulse_neg_xz,
         int_pos_xy=int_pos_xy, int_pos_xz=int_pos_xz,
         motor_pos_xy=motor_pos_xy, motor_pos_xz=motor_pos_xz,
         motor_neg_xy=motor_neg_xy, motor_neg_xz=motor_neg_xz)
