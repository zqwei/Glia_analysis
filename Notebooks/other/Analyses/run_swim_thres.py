'''
get swim threshold and swim channel from jing's json file
'''

import pandas as pd
import numpy as np
from glob import glob
import os

df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')

for ind, row in df.iterrows():
    dat_dir = row['dat_dir'].replace('/im/', '/')
    dat_dir = dat_dir.replace('/im_CM0/', '/')
    dat_dir = dat_dir.replace('/im_CM1/', '/')
    swim_j_file = glob(dat_dir+'*.json')
    save_root = row['save_dir']+'/'
    if len(swim_j_file)==0:
        continue
    swim_j_file = swim_j_file[0]
    metadata = pd.read_json(swim_j_file, orient='columns')
    try:
        swim_channel = metadata.loc['swim_channel', 'value']
        swim_threshold = metadata.loc['swim_threshold', 'value']
        np.savez(save_root+'swim_thres.npz', swim_channel=swim_channel, swim_threshold=swim_threshold)
    except Exception as e:
        if os.path.exists(save_root+'swim_thres.npz'):
            os.remove(save_root+'swim_thres.npz')
        print(swim_j_file)
        

df = pd.read_csv('../Processing/data_list_in_analysis_downsample.csv')

for ind, row in df.iterrows():
    dat_dir = row['dat_dir'].replace('/im/', '/')
    dat_dir = dat_dir.replace('/im_CM0/', '/')
    dat_dir = dat_dir.replace('/im_CM1/', '/')
    swim_j_file = glob(dat_dir+'*.json')
    save_root = row['save_dir']+'/'
    if len(swim_j_file)==0:
        print('no json file')
        print(dat_dir)
        continue
    swim_j_file = swim_j_file[0]
    metadata = pd.read_json(swim_j_file, orient='columns')
    try:
        swim_channel = metadata.loc['swim_channel', 'value']
        swim_threshold = metadata.loc['swim_threshold', 'value']
        np.savez(save_root+'swim_thres.npz', swim_channel=swim_channel, swim_threshold=swim_threshold)
    except Exception as e:
        print(swim_j_file)