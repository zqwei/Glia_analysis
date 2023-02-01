import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
%matplotlib inline
df = pd.read_csv('../Datalists/data_list_in_analysis_slimmed_v4.csv')


def decompose_affine_matrices(txms):
    """Decompose multiple 4x4 affine matrix into translations, scaling and rotations
    Parameters
    ===========
    txm [np.ndarray]: Nx4x4 transformation matrices
    
    Returns
    ========
    txs [float]: x translations
    tys [float]: y translations
    tzs [float]: z translations
    sxs [float]: x scalings
    sys [float]: y scalings
    szs [float]: z scalings
    rzs [float]: z rotations
    rys [float]: y rotations
    rxs [float]: x rotations
    """
    
    # find translation
    txs = txms[:, 0, -1]
    tys = txms[:, 1, -1]
    tzs = txms[:, 2, -1]

    # find scaling
    sxs = np.linalg.norm(txms[:, :3, 0], axis=1)
    sys = np.linalg.norm(txms[:, :3, 1], axis=1)
    szs = np.linalg.norm(txms[:, :3, 2], axis=1)

    # find rotation
    txms[:, :3, 0] = txms[:, :3, 0] / sxs[:, np.newaxis]
    txms[:, :3, 1] = txms[:, :3, 1] / sys[:, np.newaxis]
    txms[:, :3, 2] = txms[:, :3, 2] / szs[:, np.newaxis]

    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(txms[:, :3, :3])
    r_euler = r.as_euler('zyx')

    rzs = r_euler[:, 0]
    rys = r_euler[:, 1]
    rxs = r_euler[:, 2]
    return txs, tys, tzs, sxs, sys, szs, rzs, rys, rxs


for ind, row in df.iterrows():
    save_path = row['save_dir']
    if 'voluseg' not in save_path:
        continue
    dir_path = row['dat_dir']
    if 'im_CM0' in save_path:
        dir_path_ = dir_path + '/im_CM0_voluseg/im_CM0_transforms/'
    if 'im_CM1' in save_path:
        dir_path_ = dir_path + '/im_CM1_voluseg/im_CM1_transforms/'
    if not os.path.exists(dir_path_):
        if 'im_CM0' in save_path:
            dir_path_ = dir_path + '/processed/im_CM0_transforms/'
        if 'im_CM1' in save_path:
            dir_path_ = dir_path + '/processed/im_CM1_transforms/'
    if not os.path.exists(dir_path_):
        # print(ind)
        # print(dir_path)
        continue
    print(dir_path_)
    txm_files = glob(dir_path_ + '*mat')
    txm_files.sort()
    txms = np.stack(list(map(np.loadtxt, txm_files)))
    txs, tys, tzs, sxs, sys, szs, rzs, rys, rxs = decompose_affine_matrices(txms)
    fig, ax = plt.subplots(3,1, figsize=(10.5, 6), sharex=True)
    ax[0].set_title('translation [real space]')
    ax[0].set_ylabel(r'[$\mu{}m$]')
    ax[0].plot(txs-txs[0], label='x')
    ax[0].plot(tys-tys[0], label='y')
    ax[0].plot(tzs-tzs[0], label='z')
    ax[0].legend()
    ax[1].set_title('scaling')
    ax[1].plot(sxs, label='x')
    ax[1].plot(sys, label='y')
    ax[1].plot(szs, label='z')
    ax[1].legend()
    ax[2].set_title('rotation')
    ax[2].plot(rxs, label='x')
    ax[2].plot(rys, label='y')
    ax[2].plot(rzs, label='z')
    ax[2].set_ylabel('[rad]')
    ax[2].legend()
    ax[2].set_xlabel('time points')
    fig.tight_layout()
    plt.show()
    plt.close()