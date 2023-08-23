import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
# df = pd.read_csv('../Datalists/data_list_in_analysis_glia_v3.csv', index_col=0)
df = pd.read_csv('../Datalists/data_list_in_analysis_NE_v2.csv', index_col=0)
# df = pd.read_csv('../Datalists/data_list_in_analysis_slimmed_v4.csv', index_col=0)

m = 0
for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    if os.path.exists(save_root+'cell_center_voluseg.npy'):
        # print(ind)
        m += 1
print(m)