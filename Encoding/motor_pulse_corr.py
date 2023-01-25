'''
This code is used to find cells
that correlates with instantaneous motor
and pulse varibles

author: Ziqiang Wei
email: weiz@janelia.hhmi.org
'''

import numpy as np
import os, sys
import pandas as pd
from scipy.stats import spearmanr, zscore
from tqdm import tqdm
# df = pd.read_csv('../Datalists/data_list_in_analysis_pulse_cells_v2.csv')
# df = pd.read_csv('../Datalists/data_list_in_analysis_NGGU.csv')
# df = pd.read_csv('../Datalists/data_list_in_analysis_glia_v1.csv')
df = pd.read_csv('../Datalists/data_list_in_analysis_NE_v0.csv')


def process_n_file(ind):
    row = df.iloc[ind]
    save_root = row['save_dir']+'/'
    print(save_root)
    print(ind)
    if os.path.exists(save_root+'cell_motor_corr.npz'):
        print('motor exists')
    if os.path.exists(save_root+'cell_pulse_series_corr.npz'):
        print('pulse exists')

    #############
    # motor
    #############
    cells_center = np.load(save_root+'cell_center.npy')
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    cells_center = cells_center[cell_in_brain]
    row = df.iloc[ind]
    save_root = row['save_dir']+'/'
    _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
    probe_amp=_['probe_amp']
    swim_t_frame=_['swim_t_frame']
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF']
    len_dff = min(dFF_.shape[1], _['epoch_frame'].shape[0])
    epoch_frame=_['epoch_frame'][:len_dff]
    pulse_frame=_['pulse_frame'][:len_dff]
    visu_frame=_['visu_frame'][:len_dff]
    lswim_frame=_['lswim_frame'][:len_dff]
    rswim_frame=_['rswim_frame'][:len_dff]
    visu_frame_=_['visu_frame_'][:len_dff]
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    dFF_ = dFF_[cell_in_brain][:, :len_dff]
    # CL_idx = epoch_frame<=1
    # rl, _ = spearmanr(lswim_frame[CL_idx], visu_frame[CL_idx])
    # rr, _ = spearmanr(rswim_frame[CL_idx], visu_frame[CL_idx])
    rl = row['rl']
    rr = row['rr']
    if rl >= rr:
        swim_frame_ = lswim_frame
    else:
        swim_frame_ = rswim_frame

    num_cells = dFF_.shape[0]
    p_cell = np.zeros(num_cells)
    r_cell = np.zeros(num_cells)
    if epoch_frame.max()<5:
        print('NG only task')
        idx_ = epoch_frame==1 # OL period -- NG only task
    else:
        idx_ = epoch_frame==6 # OL period
    for n in tqdm(range(num_cells)):
        r_, p_ = spearmanr(dFF_[n, idx_], swim_frame_[idx_])
        p_cell[n] = p_
        r_cell[n] = r_
    np.savez(save_root+'cell_motor_corr.npz', p_cell=p_cell, r_cell=r_cell)

    #############
    # pulse
    #############
    epoch_on = np.where((epoch_frame[1:]==3) & (epoch_frame[:-1]==2))[0]+1
    epoch_off = np.where((epoch_frame[1:]!=3) & (epoch_frame[:-1]==3))[0]
    len_ = min(len(epoch_on), len(epoch_off))
    pulse_idx_ = np.zeros(dFF_.shape[1]).astype('bool')
    for n_ in range(len_):
        on_ = epoch_on[n_]
        off_ = epoch_off[n_]
        vis_ = visu_frame[on_:off_]
        if len(vis_)==0:
            continue
        if (vis_<0).mean()<=0.1: # pulse trial
                continue
        pulse_idx_[on_:off_] = True

    idx_ = pulse_idx_ & (swim_frame_==0)
    num_cells = dFF_.shape[0]
    p_cell = np.zeros(num_cells)
    r_cell = np.zeros(num_cells)
    idx_ = pulse_idx_ & (swim_frame_==0)
    for n in tqdm(range(num_cells)):
        r_, p_ = spearmanr(dFF_[n, pulse_idx_][:6000], pulse_frame[pulse_idx_][:6000])
        p_cell[n] = p_
        r_cell[n] = r_
    np.savez(save_root+'cell_pulse_series_corr.npz', p_cell=p_cell, r_cell=r_cell)
    return None


for ind, _ in df.iterrows():
    process_n_file(ind)