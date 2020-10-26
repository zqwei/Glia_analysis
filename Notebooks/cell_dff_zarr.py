from utils import *
import shutil
import dask.array as da
import zarr
brain_map_thres = 2

def cell_dff_zarr(row):
    save_root = row['save_dir']+'/'    
    print(save_root)
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    A = _['A']
    A_loc = _['A_loc']
    dFF = _['dFF'].astype('float')
    _ = None
    numCore = 450
    if os.path.exists(save_root+'cell_dff.zarr'):
        shutil.rmtree(save_root+'cell_dff.zarr')
    dFF_ = zarr.array(dFF, chunks=(dFF.shape[0]//(numCore-2), dFF.shape[1]))
    zarr.save(save_root+'cell_dff.zarr', dFF_)
    return None


if __name__ == "__main__":
    df = pd.read_csv('../Processing/data_list_in_analysis.csv')
    for ind, row in df.iterrows():
        cell_dff_zarr(row)