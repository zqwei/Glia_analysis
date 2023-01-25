from utils import *
from h5py import File
import pickle


df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')

for nrow, row in df.iterrows():
    dat_dir = row['dat_dir'].replace('/im/', '/')
    dat_dir = dat_dir.replace('/im_CM0/', '/')
    dat_dir = dat_dir.replace('/im_CM1/', '/')
    save_root = row['save_dir']+'/'
    transformed_cells_dir = dat_dir + 'processed/transformed_ind/'
    transformed_cells_pkl = transformed_cells_dir + 'transformed_cells_df.pkl'
    confocal_shape_path = transformed_cells_dir + 'atlas-shape.pkl'
    if (not os.path.exists(transformed_cells_pkl)) or (not os.path.exists(confocal_shape_path)):
        continue
    with open(transformed_cells_pkl, 'rb') as f: transformed_cells = pickle.load(f)
    with open(confocal_shape_path, 'rb') as f: confocal_shape = pickle.load(f)
    
    num_cells = transformed_cells.shape[0]
    A_center = np.zeros((num_cells, 3))
    A_ext = np.zeros(num_cells, dtype=object)

    for icell in range(num_cells):
        ind, wghts = transformed_cells.iloc[icell]
        arr_ = np.unravel_index(ind, confocal_shape, order='C')
        arr_ = np.array(arr_)
        center_ = np.zeros(3)
        center_[:] = np.nan
        if arr_.shape[1]>0:
            A_ext[icell] = arr_[:, wghts>(wghts.max()*0.01)]
            for n, a_ in enumerate(arr_):
                valid_idx = (~np.isnan(wghts)) & (~np.isnan(a_))
                center_[n] = (wghts[valid_idx]*a_[valid_idx]).sum()/(wghts[valid_idx].sum())
        else:
            A_ext[icell] = arr_
        A_center[icell] = center_

    np.save(save_root+'cell_center_registered.npy', A_center)
    np.save(save_root+'cell_center_registered_ext.npy', A_ext)