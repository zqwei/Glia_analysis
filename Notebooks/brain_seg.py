from utils import *
from factor import factor_, thres_factor_


def brain_seg_factor(row, t_min=5000, t_max=30000, num_cluster=200, l_thres_=0.5, n_thres = 0.8):
    save_root = row['save_dir']+'/'
    print('Processing data at: '+row['dat_dir'])
    print('Saving at: '+save_root)
    
    brain_map = np.load(save_root+'Y_ave.npy').astype('float').squeeze()
    trans_ = np.load(save_root+'trans_affs.npy')
    processed_dir = row['dat_dir'] + 'processed/'
    
    t_max=np.min([t_max, trans_.shape[0]//4*3])
    
    plt.figure(figsize=(4, 3))
    plt.plot(trans_[:, 1, -1])
    plt.plot(trans_[:, 2, -1])
    plt.xlim([t_min, t_max])
    sns.despine()
    plt.savefig(save_root+'registration.png')
    
    print('========Load data file========')
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
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
    print(f'Number of layers in brain stacks: {num_z}')
       
    nc = np.max([10, num_cluster//num_z+1])
    
    for nz in range(num_z):
        print(f'Processing layer: {nz}')
        valid_cell, lam, loadings, rotation_mtx, phi, scores, scores_rot = factor_(dFF[A_center[:,0]==nz], n_c=nc, noise_thres=n_thres)
        idx=A_center[:,0]==nz
        x = A_center[idx, 2]
        y = A_center[idx, 1]
        loadings_, valid_c_=thres_factor_(x, y, valid_cell, loadings, l_thres_=l_thres_)
        np.savez(save_root+'layer_factors_{:03d}'.format(nz), \
                 valid_cell=valid_cell, \
                 lam=lam, loadings=loadings, \
                 rotation_mtx=rotation_mtx, phi=phi, \
                 scores=scores, scores_rot=scores_rot, \
                 x=x, y=y, loadings_=loadings_, valid_c_=valid_c_)
    
    print('========Downsample neurons according to layer factor results========')
    dFF_ = []
    zs = []
    ys = []
    xs = []
    loading_list=[]
    for nz in range(num_z):
        _ = np.load(save_root+'layer_factors_{:03d}.npz'.format(nz), allow_pickle=True)
        valid_cell=_['valid_cell']
        x=_['x']
        y=_['y']
        loadings_=_['loadings_']
        sub_= (np.abs(loadings_)>0).sum(axis=1)>0
        dFF_.append(dFF[A_center[:,0]==nz][valid_cell][sub_])
        zs.append(nz*np.ones(sub_.sum()))
        xs.append(x[valid_cell][sub_])
        ys.append(y[valid_cell][sub_])
    
    dFF_=np.vstack(dFF_)
    zs=np.concatenate(zs)
    ys=np.concatenate(ys)
    xs=np.concatenate(xs)
    
    valid_cell, lam, loadings, rotation_mtx, phi, scores, scores_rot = factor_(dFF_, n_c=num_cluster, noise_thres=n_thres)
    np.savez(save_root+'brain_seg_factors', \
             dFF=dFF_[valid_cell], \
             lam=lam, loadings=loadings, \
             rotation_mtx=rotation_mtx, phi=phi, \
             scores=scores, scores_rot=scores_rot, \
             x=xs[valid_cell], y=ys[valid_cell], z=zs[valid_cell])
    
    
    
    