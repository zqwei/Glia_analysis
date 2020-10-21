from utils import *
from factor import factor_, thres_factor_


def cell_location(row):
    save_root = row['save_dir']+'/'
    print('Processing data at: '+row['dat_dir'])
    print('Saving at: '+save_root)
    brain_map = np.load(save_root+'Y_ave.npy').astype('float').squeeze()    
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    A = _['A']
    A_loc = _['A_loc']
    _ = None
    A_center = np.zeros((A.shape[0],3))
    (X,Y) = np.meshgrid(np.arange(100),np.arange(100))
    for n_, A_ in enumerate(A):
        A_loc_ = A_loc[n_]
        z, x, y = A_loc_
        A_[A_<A_.max()*0.4]=0
        cx = (X*A_).sum()/A_.sum()
        cy = (Y*A_).sum()/A_.sum()
        A_center[n_]=np.array([z, x+cy, y+cx])
    np.save(save_root+'cell_center.npy', A_center)
    
# this is rotation mistake location of the cells
# just leaving the data for reference
def cell_location_(row):
    save_root = row['save_dir']+'/'
    print('Processing data at: '+row['dat_dir'])
    print('Saving at: '+save_root)
    brain_map = np.load(save_root+'Y_ave.npy').astype('float').squeeze()    
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    A = _['A']
    A_loc = _['A_loc']
    _ = None
    A_center = np.zeros((A.shape[0],3))
    (X,Y) = np.meshgrid(np.arange(100),np.arange(100))
    for n_, A_ in enumerate(A):
        A_loc_ = A_loc[n_]
        z, x, y = A_loc_
        A_[A_<A_.max()*0.4]=0
        cx = (X*A_).sum()/A_.sum()
        cy = (Y*A_).sum()/A_.sum()
        A_center[n_]=np.array([z, x+cx, y+cy])
    np.save(save_root+'cell_center_.npy', A_center)
    

if __name__ == "__main__":
    df = pd.read_csv('../Processing/data_list_in_analysis.csv')
    for ind, row in df.iterrows():
        save_root = row['save_dir']+'/'
        if not os.path.exists(save_root+'cell_center.npy'):
            cell_location(row)
        if not os.path.exists(save_root+'cell_center_.npy'):
            cell_location_(row)
