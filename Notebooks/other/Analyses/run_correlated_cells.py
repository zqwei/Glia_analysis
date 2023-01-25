from utils import *

df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')

for ind, row in df.iterrows():
    save_root = row['save_dir']+'/'
    if os.path.exists(save_root+'correlated_cells.npz'):
        continue
    print(row['taskType'])
    print(save_root)
    if 'im_CM1' in save_root:
        mask_thres_=1
    else:
        mask_thres_=2
    # print(mask_thres_)
    dFF, A_center = load_masked_data(row, mask_thres=mask_thres_) # for glia using thres=1
    num_cells = A_center.shape[0]
    idx_list=[]
    ev_list=[]
    num_z = A_center[:,0].max().astype('int')+1
    n_c = 10
    for nz in tqdm(range(num_z)):
        if (A_center[:,0]==nz).sum()==0:
            continue
        idx = np.where(A_center[:,0]==nz)[0]
        idx_list.append(idx)
        ev_list.append(FA_ev(dFF[idx], n_c=10))

    idx_list_ = np.concatenate(idx_list)
    ev_list_ = np.concatenate(ev_list)
    np.savez(save_root+'correlated_cells.npz', idx_list_=idx_list_, ev_list_=ev_list_)