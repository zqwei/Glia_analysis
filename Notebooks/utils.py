import numpy as np
import os, sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import fish_proc.wholeBrainDask.cellProcessing_single_WS as fwc
import dask.array as da
import pandas as pd
from glob import glob
from tqdm import tqdm
import seaborn as sns
sns.set(font_scale=2, style='ticks')



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
