from sensory_motor_bar_code import *
from brain_state_bar_code import *
from brain_state_bar_code_raw import *
from pathlib import Path

df = pd.read_csv('../Processing/data_list_in_analysis.csv')

for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    if not 'LG' in row['taskType']:
        continue
    if not os.path.exists(save_root+'brain_seg_factors.npz'):
        continue
    print(row['save_dir'])
#     if ind<7:
#         continue
    # visual vs motor
#     if not os.path.exists(row['save_dir']+'/'+'sensory_motor_bar_code.done'):
#     sensory_motor_bar_code(row)
#     Path(row['save_dir']+'/'+'sensory_motor_bar_code_raw.done').touch()
    # state-depdendent swim
#     if not os.path.exists(row['save_dir']+'/'+'brain_state_bar_code.done'):
#     brain_state_bar_code(row)
#     Path(row['save_dir']+'/'+'brain_state_bar_code.done').touch()
    # state-depdendent visual
#     if not os.path.exists(row['save_dir']+'/'+'brain_state_bar_code_raw.done'):
#         brain_state_bar_code_raw(row)
#         Path(row['save_dir']+'/'+'brain_state_bar_code_raw.done').touch()
    brain_state_bar_code_raw(row)