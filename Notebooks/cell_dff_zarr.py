from utils import *
import shutil
import dask.array as da
import zarr


def cell_dff_zarr(row):
    save_root = row['save_dir']+'/'    
    # print(save_root)
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    dFF = _['dFF'].astype('float')
    _ = None
    numCore = 450
    if os.path.exists(save_root+'cell_dff.zarr'):
        shutil.rmtree(save_root+'cell_dff.zarr')
    dFF_ = zarr.array(dFF, chunks=(dFF.shape[0]//(numCore-2), dFF.shape[1]))
    zarr.save(save_root+'cell_dff.zarr', dFF_)
    return None


if __name__ == "__main__":
#     df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')
    df = pd.read_csv('../Processing/data_list_in_analysis_downsample.csv')
    for ind, row in df.iterrows():
        save_root = row['save_dir']+'/'
        if not os.path.exists(save_root+'cell_dff.npz'):
            continue
        if not os.path.exists(save_root+'cell_dff.zarr'):
            cell_dff_zarr(row)