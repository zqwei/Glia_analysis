import pandas as pd
import shutil, os


if __name__ == "__main__":
#     df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')
    df = pd.read_csv('../Processing/data_list_in_analysis_downsample.csv')
    for ind, row in df.iterrows():
        save_root = row['save_dir']+'/'
        if not os.path.exists(save_root+'cell_dff.npz'):
            continue
        if not os.path.exists(save_root+'cell_dff.zarr'):
            continue
        else:
            shutil.rmtree(save_root+'cell_dff.zarr')
