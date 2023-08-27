#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

from sensory_motor_bc import *
from brain_state_bcr import *
from brain_state_rl_bcr import *
from multi_vel_bc import *
from pathlib import Path
from sensory_motor_pvi_task import *
from brain_state_pvi_bcr import *

df = pd.read_csv('../Processing/data_list_in_analysis_downsample.csv')

for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    if not os.path.exists(save_root+'ephys.npz'):
        continue
    if 'simulate-visual' in row['taskType']:
        continue
    if 'LG_vs_NGGUmultivel' in row['taskType']:
        continue
    print(ind, row['save_dir'])
    sensory_motor_bar_code(row)
    brain_state_bar_code_raw(row)


# for ind, row in df.iterrows():
#     save_root = row['save_dir']+'/'
#     if not 'LG_vs_NGGUmultivel' in row['taskType']:
#         continue
#     print(row['save_dir'])
#     multi_vel_bar_code(row)


# for ind, row in df.iterrows():
#     save_root = row['save_dir']+'/'
#     if not 'MG_vs_replayGU' in row['taskType']:
#         continue
#     print(row['save_dir'])
#     if not os.path.exists(row['save_dir']+'cell_type_stats_msensory.npz'):
#         sensory_motor_bar_code(row)
#     if not os.path.exists(row['save_dir']+'cell_pulse_ap_stats_raw.npz'):
#         brain_state_rl_bar_code_raw(row)
#     sensory_motor_bar_code(row)
#     brain_state_rl_bar_code_raw(row)


# for ind, row in df.iterrows():
#     save_root = row['save_dir']+'/'
#     if ind<20:
#         continue
#     if not 'simulate-visual' in row['taskType']:
#         continue
#     if not os.path.exists(save_root+'cell_dff.npz'):
#         continue
#     print(row['save_dir'])
#     sensory_motor_pvi_bar_code(row)
#     brain_state_pvi_bar_code_raw(row)