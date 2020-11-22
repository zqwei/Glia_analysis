from sensory_motor_bc import *
from brain_state_bcr import *
from brain_state_rl_bcr import *
from multi_vel_bc import *
from pathlib import Path

df = pd.read_csv('../Processing/data_list_in_analysis.csv')

for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    if 'replay' in row['taskType']:
        continue
#     if ind<15:
#         continue
    print(row['save_dir'])
#     sensory_motor_bar_code(row)
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
#     brain_state_rl_bar_code_raw(row)
    