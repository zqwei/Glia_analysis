from sensory_motor_bar_code import *
from brain_state_bar_code import *
from pathlib import Path

df = pd.read_csv('../Processing/data_list.csv')

for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    if not 'LG' in row['taskType']:
        continue
    if not os.path.exists(save_root+'brain_seg_factors.npz'):
        continue
    print(row['save_dir'])
    if not os.path.exists(row['save_dir']+'/'+'sensory_motor_bar_code.done'):
        sensory_motor_bar_code(row)
        Path(row['save_dir']+'/'+'sensory_motor_bar_code.done').touch()
    if not os.path.exists(row['save_dir']+'/'+'brain_state_bar_code.done'):
        brain_state_bar_code(row)
        Path(row['save_dir']+'/'+'brain_state_bar_code.done').touch()