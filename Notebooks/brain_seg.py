from utils import *
from factor import factor_, thres_factor_


def brain_layer_seg_factor(row, t_min=5000, t_max=30000, l_thres_=0.5, n_thres = 0.8):
    save_root = row['save_dir']+'/'
    print('Processing data at: '+row['dat_dir'])
    print('Saving at: '+save_root)
    
    brain_map = np.load(save_root+'Y_ave.npy').astype('float').squeeze()
    trans_ = np.load(save_root+'trans_affs.npy')
    processed_dir = row['dat_dir'] + 'processed/'
    
    if t_min>(trans_.shape[0]//4*3):
        t_min = 0
    t_max=np.min([t_max, trans_.shape[0]//4*3])
    
    plt.figure(figsize=(4, 3))
    plt.plot(trans_[t_min:t_max, 1, -1])
    plt.plot(trans_[t_min:t_max, 2, -1])
    sns.despine()
    plt.savefig(save_root+'registration.png')
    
    print('========Load data file========')
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    A = _['A']
    A_loc = _['A_loc']
    dFF = _['dFF'].astype('float')[:, t_min:t_max]
    _ = None
    
    print('========Compute cell mass center========')
    if not os.path.exists(save_root+'cell_center.npy'):
        A_center = np.zeros((dFF.shape[0],3))
        (X,Y) = np.meshgrid(np.arange(100),np.arange(100))
        for n_, A_ in enumerate(A):
            A_loc_ = A_loc[n_]
            z, x, y = A_loc_
            A_[A_<A_.max()*0.4]=0
            cx = (X*A_).sum()/A_.sum()
            cy = (Y*A_).sum()/A_.sum()
            A_center[n_]=np.array([z, x+cx, y+cy])
        np.save(save_root+'cell_center.npy', A_center)
    A_center = np.load(save_root+'cell_center.npy')
    
    num_z=A_center[:,0].max().astype('int')
    nc = 15
    print(f'Number of layers in brain stacks: {num_z}')
    print(f'Number of clusters per layer: {nc}')
    
    print('========Compute layer factor results========')
    for nz in range(num_z):
        if (A_center[:,0]==nz).sum()==0:
            continue
        print(f'Processing layer: {nz}')
        valid_cell, lam, loadings, rotation_mtx, phi, scores, scores_rot = factor_(dFF[A_center[:,0]==nz], \
                                                                                   n_c=nc, \
                                                                                   noise_thres=n_thres)
        if valid_cell is not None:
            idx=A_center[:,0]==nz
            x = A_center[idx, 2]
            y = A_center[idx, 1]
            loadings_, valid_c_=thres_factor_(x, y, valid_cell, loadings, l_thres_=l_thres_)
            print(f'fraction of valid cells: {valid_cell.mean():.2f}')
            np.savez(save_root+'layer_factors_{:03d}'.format(nz), \
                     valid_cell=valid_cell, \
                     lam=lam, loadings=loadings, \
                     rotation_mtx=rotation_mtx, phi=phi, \
                     scores=scores, scores_rot=scores_rot, \
                     x=x, y=y, loadings_=loadings_, valid_c_=valid_c_)



def brain_seg_factor(row, t_min=5000, t_max=30000, num_cluster=200, l_thres_=0.5, n_thres = 0.8):
    save_root = row['save_dir']+'/'
    print('Processing data at: '+row['dat_dir'])
    print('Saving at: '+save_root)
    print('========Load data file========')
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    A = _['A']
    A_loc = _['A_loc']
    if t_min>(dFF.shape[1]):
        t_min = 0
    dFF = _['dFF'].astype('float')[:, t_min:t_max]
    dFF_ = _['dFF'].astype('float')
    _ = None
    A_center = np.load(save_root+'cell_center.npy')
    num_z = A_center[:,0].max().astype('int')
    
    print('========Downsample neurons according to layer factor results========')
    corr_idx = []
    for nz in range(num_z):
        if os.path.exists(save_root+'layer_factors_{:03d}.npz'.format(nz)):
            _ = np.load(save_root+'layer_factors_{:03d}.npz'.format(nz), allow_pickle=True)
            valid_cell=_['valid_cell']
            loadings_=_['loadings_']
            layer_idx = np.where(A_center[:,0]==nz)[0]
            sub_= (np.abs(loadings_)>0).sum(axis=1)>0
            corr_idx.append(layer_idx[valid_cell][sub_])
    
    corr_idx=np.hstack(corr_idx)
    
    valid_cell, lam, loadings, rotation_mtx, phi, scores, scores_rot = factor_(dFF[corr_idx], n_c=num_cluster, noise_thres=n_thres)
    x=A_center[corr_idx, 2]
    y=A_center[corr_idx, 1]
    z=A_center[corr_idx, 0]
    
    loadings_, valid_c_=thres_factor_(x, y, valid_cell, loadings, l_thres_=-0.1, shape_thres_=10)
    sub_=(np.abs(loadings_)>0).sum(axis=1)>0
    valid_c=(np.abs(loadings_)>0).sum(axis=0)>100
    
    np.savez(save_root+'brain_seg_factors', \
             dFF=dFF_[corr_idx][valid_cell], \
             lam=lam, loadings=loadings, \
             rotation_mtx=rotation_mtx, phi=phi, \
             scores=scores, scores_rot=scores_rot, \
             x=x[valid_cell], y=y[valid_cell], z=z[valid_cell])
    
    
    
    