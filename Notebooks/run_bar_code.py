from single_cell_type_cluster import *

df = pd.read_csv('../Processing/data_list.csv')
row = df.iloc[15]

for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    if not os.path.exists(save_root+'brain_seg_factors.npz'):
        continue
    if not os.path.exists(save_root+'cell_active_pulse_stats.npz'):
        continue    
    bar_code(row)