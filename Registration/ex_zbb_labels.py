import numpy as np
from scipy import sparse
_ = np.load('/nrs/ahrens/Ziqiang/Atlas/ZBrain/atlas_label_mat.npz',allow_pickle=True)
atlas_label_mat = _['atlas_label_mat']
mask_mat = sparse.csc_matrix((_['mask_data'], _['mask_ir'], _['mask_jc']), shape=_['shape'])
mask_labels = _['mask_labels']

cell_locs = np.random.rand((10, 3))

num_labels = mask_mat.shape[1]
num_c_cells = tectum_neurons.shape[0]
atlas_zbrain_list = np.zeros((num_c_cells, num_labels))
z, y, x = cell_locs.astype('int').T
atlas_zbrain_list = mask_mat[atlas_label_mat[z, y, x]].toarray()

cell_per_region = atlas_zbrain_list.sum(axis=0)
labels_ = mask_labels[np.where(atlas_zbrain_list.sum(axis=0)>0)[0]]

[print(l_, n_) for (l_, n_) in zip(cell_per_region[cell_per_region>0], labels_)]