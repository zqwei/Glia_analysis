import matplotlib
matplotlib.use('Agg')
from utils import *
from brain_seg import brain_seg_factor, brain_layer_seg_factor

df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')

for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    if os.path.exists(save_root+'brain_seg_factors.npz'):
        continue
    if not os.path.exists(save_root+'cell_dff.npz'):
        continue
    print(save_root)
    if (ind<15) or (ind>16):
        continue
    brain_layer_seg_factor(row, t_min=0, t_max=np.inf, l_thres_=0.01, n_thres = 0.8)
    brain_seg_factor(row, num_cluster=50, l_thres_=0.01, n_thres = 0.7)

