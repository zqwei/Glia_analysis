import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from rastermap import mapping
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set(font_scale=1.5, style='ticks')


def read_h5(filename, dset_name='default'):
    import h5py
    with h5py.File(filename, 'r') as hf:
        return hf[dset_name][()]


def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
