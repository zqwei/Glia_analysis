from single_cell_type_cluster import *

df = pd.read_csv('../Processing/data_list.csv')

for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    if not (row['taskType']=='LGNGGU'):
        continue
    if not os.path.exists(save_root+'brain_seg_factors.npz'):
        continue
#     if os.path.exists(save_root+'cell_active_pulse_stats.npz'):
#         continue  
    print(save_root)
    bar_code(row)