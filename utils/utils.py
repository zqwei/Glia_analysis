import numpy as np
import os, sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import dask.array as da
import pandas as pd
from glob import glob
from tqdm import tqdm
import seaborn as sns
from scipy.stats import ranksums, wilcoxon
from sklearn.cluster import KMeans
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import spearmanr, linregress, wilcoxon
from matplotlib.colors import ListedColormap
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from scipy.stats import zscore
from scipy.signal import medfilt
from tqdm import tqdm
sns.set(font_scale=1.5, style='ticks')


def FA_ev(X, n_c=10):
    X = zscore(X, axis=-1)
    FA_=FactorAnalysis(n_components=n_c, svd_method='randomized', random_state=None, iterated_power=7)
    FA_.fit(X.T)
    coeff_= FA_.components_
    x_new = (FA_.transform(X.T).dot(coeff_)).T
    return 1-((X-x_new)**2).mean(axis=-1)


def moving_average(x, win, axis=0):
    return pd.DataFrame(x).rolling(win, win_type='boxcar', center=False, min_periods=1, axis=axis).mean().to_numpy()


def plt_error(x, Y, ax, axis=0, sem=False, color='k', linewidth=2):
    n = Y.shape[axis]
    mean_ = np.nanmean(Y, axis=axis)
    error = np.nanstd(Y, axis=axis)
    if sem:
        error = error/np.sqrt(n)
    ax.plot(x, mean_, '-', color=color, linewidth=linewidth)
    ax.fill_between(x, mean_-error, mean_+error, alpha=0.5, color=color)
    
    
def weight_proj(weight, axis=0):
    weight_max = weight.max(axis)
    weight_min = weight.min(axis)
    w_ = weight_max.copy()
    w_[np.abs(weight_max)<np.abs(weight_min)] = weight_min[np.abs(weight_max)<np.abs(weight_min)]
    w_ = np.ma.array (w_, mask=(w_==0))
    return w_


def load_masked_data(row, mask_thres=2):
    save_root = row['save_dir']+'/'
    _ = np.load(save_root+'cell_dff.npz', allow_pickle=True)
    A = _['A']
    A_loc = _['A_loc']
    dFF = _['dFF'].astype('float')
    num_dff = _['dFF'].shape[-1]
    _ = None
    brain_map = np.load(save_root+'Y_ave.npy').astype('float').squeeze()
    A_center = np.load(save_root+'cell_center.npy')
    A_center_grid = np.round(A_center).astype('int')
    cells_in_mask = []
    for n_layer in range(brain_map.shape[0]):
        layer_ = A_center[:, 0]==n_layer
        cell_ids = np.where(layer_)[0]
        mask_ = brain_map[n_layer]>mask_thres
        y = A_center_grid[cell_ids, 2]
        x = A_center_grid[cell_ids, 1]
        x_max, y_max = mask_.shape
        num_cells = len(cell_ids)
        in_mask_ = np.zeros(num_cells).astype('bool')
        for n in range(num_cells):
            if (x[n]<x_max) and (y[n]<y_max):
                in_mask_[n] = mask_[x[n], y[n]]
        cells_in_mask.append(cell_ids[in_mask_])
    cells_in_mask = np.concatenate(cells_in_mask)
    A_center = A_center[cells_in_mask]
    dFF = dFF[cells_in_mask]
    return dFF, A_center


def ep2frame(camtrig, thres=3.8):
    arr_ = (camtrig>thres).astype('int')
    return np.where((arr_[:-1]-arr_[1:])==-1)[0]+1


def WNtest(zdFF, lags=30):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    num_cell = zdFF.shape[0]
    valid_ = np.zeros(num_cell).astype('int')
    for n_cell in tqdm(range(num_cell)):
        _, p = acorr_ljungbox(zdFF[n_cell], lags=lags)
        valid_[n_cell]=(p<.05).sum()
    return valid_


def layer_corr(zdFF, A_loc, corr_thres=0.25, corr_size=10):
    from scipy.stats import spearmanr
    num_cell = zdFF.shape[0]
    num_z = A_loc[:,0].max()+1
    valid_ = np.zeros(num_cell).astype('bool')
    for nz in tqdm(range(num_z)):
        nz_list = np.where(A_loc[:,0]==nz)[0]
        corr_, p_ = spearmanr(zdFF[nz_list], axis=1)
        valid_thres = ((p_<0.05).sum(axis=-1)>corr_size) & ((np.abs(corr_)>corr_thres).sum(axis=-1)>corr_size)
        valid_[nz_list[valid_thres]]=True
    return valid_


def _rankdata_(X, axis=1):
    from bottleneck import rankdata
    X = rankdata(X, axis=axis)
    return (X - X.mean(axis=axis, keepdims=True)) / X.std(axis=axis, keepdims=True)/np.sqrt(X.shape[axis]),


def rankdata_(X, axis=1):
    from fish_proc.utils.np_mp import parallel_to_chunks
    X_r, = parallel_to_chunks(_rankdata_, X, axis=1)
    return X_r


def matmul(a, b):
    import mkl
    mkl.set_num_threads(96)
    print(f'Test thread at: {mkl.get_max_threads()}')
    import numpy as np
    s = np.matmul(a, b)
    mkl.set_num_threads(1)
    print(f'Test thread at: {mkl.get_max_threads()}')
    import numpy as np
    return s


def rolling_perc(x, window=180000, perc=1):
    import pandas as pd
    s = pd.Series(x).rolling(window=window).quantile(perc).values
    x_ = np.zeros(len(x))
    x_[window//2:-window//2]=s[window:]
    x_[:window//2]=s[window]
    x_[-window//2:]=s[-1]
    return x_


def wrap_data(x, indx, len_):
    x_ = np.zeros((len_, len(indx)))
    for n in range(len_):
        x_[n] = x[indx+n]
    return x_


def open_ephys_metadata(xml):
    import xml.etree.ElementTree as et
    import collections
    import pandas as pd
    def tryfloat (x):
        try: return float(x)
        except: return(x)
    tree = et.parse(xml)
    root = tree.getroot()
    StimConds = []
    for r in root.getchildren():
        StimCond = collections.OrderedDict()
        for e in r:
            StimCond[e.tag] = (tryfloat(e.text))
        StimConds.append(StimCond)
    columns = list(StimConds[0].keys())
    columns.remove('epoch')
    index = [s['epoch'] for s in StimConds]
    return pd.DataFrame(StimConds, index=index, columns=columns)
