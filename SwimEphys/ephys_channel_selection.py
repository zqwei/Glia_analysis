'''
This code is used to find the channel of swim
(ephys) that close to visual input (VR feedback)

author: Ziqiang Wei
email: weiz@janelia.hhmi.org
'''

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
import os
warnings.filterwarnings('ignore')

# dfile = '../../Datalists/data_list_in_analysis_pulse_cells_v2.csv'
# dfile = '../../Datalists/data_list_in_analysis_NGGU.csv'
# dfile = '../Datalists/data_list_in_analysis_slimmed_v4.csv'
# dfile = '../Datalists/data_list_in_analysis_glia_v3.csv'
# dfile = '../Datalists/data_list_in_analysis_NE_v2.csv'
dfile = '../Datalists/data_list_in_analysis_neuron_v0.csv'
df = pd.read_csv(dfile, index_col=0)


rr_list = []
rl_list = []
for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    
    if not os.path.exists(save_root + 'KA_ephys.npz'):
        print(ind)
        rr_list.append(np.nan)
        rl_list.append(np.nan)
        continue

    _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
    epoch_frame=_['epoch_frame']
    visu_frame=_['visu_frame']
    lswim_frame=_['lswim_frame']
    rswim_frame=_['rswim_frame']
    visu_frame_=_['visu_frame_']
    
    CL_idx = epoch_frame<=1
    rl, _ = spearmanr(lswim_frame[CL_idx], visu_frame[CL_idx])
    rr, _ = spearmanr(rswim_frame[CL_idx], visu_frame[CL_idx])
    rr_list.append(rr)
    rl_list.append(rl)
    
df['rr'] = rr_list
df['rl'] = rl_list

df.to_csv(dfile)