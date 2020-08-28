import matplotlib
matplotlib.use('Agg')
from utils import *
from brain_seg import brain_seg_factor, brain_layer_seg_factor

df = pd.read_csv('../Processing/data_list.csv')
len_ = len(df)

for file_id in range(32, 33):
    row = df.iloc[file_id]
    save_root = row['save_dir']+'/'
    brain_layer_seg_factor(row, t_min=5000, t_max=30000, l_thres_=0.5, n_thres = 0.8)
    brain_seg_factor(row, t_min=5000, t_max=30000, num_cluster=300, l_thres_=0.5, n_thres = 0.8)

