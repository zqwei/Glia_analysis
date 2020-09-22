from single_cell_type_cluster import *

df = pd.read_csv('../Processing/data_list.csv')
# df = pd.read_csv('../Processing/datasets.csv')

for ind, row in df.iterrows():
    dat_dir = row['dat_dir'].replace('/im/', '/')
    dat_dir = dat_dir.replace('/im_CM0/', '/')
    dat_dir = dat_dir.replace('/im_CM1/', '/')
    dat_dir = dat_dir+'/'
    p_dir = dat_dir + 'processed/'
    ephys_dir = dat_dir + 'ephys/'
    if not os.path.exists(row['dat_dir']):
        print(row['dat_dir'])
    if not os.path.exists(ephys_dir):
        print(ephys_dir)
    if not os.path.exists(row['save_dir']):
        print(row['save_dir'])
#     if not row['hr_dir'] and (not os.path.exists(row['hr_dir'])):
#         print(row['hr_dir'])
#     if not row['cf_dir'] and (not os.path.exists(row['cf_dir'])):
#         print(row['cf_dir'])