'''
Shared plotting helpers for the Clusters figure scripts.
'''

import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
sns.set(font_scale=1.5, style='ticks')

def plot_shade_err(x, y, axis=-1, plt=plt, linespec='-k', shadespec='k', err_f=1., is_sem=True):
    from scipy.stats import sem
    mean_ = np.nanmean(y, axis=axis)
    if is_sem:
        error = sem(y, axis=axis, nan_policy='omit')/err_f
    else:
        error = np.nanstd(y, axis=axis)/err_f
    plt.plot(x, mean_, linespec)
    plt.fill_between(x, mean_-error, mean_+error, edgecolor=None, linewidth=0.0, facecolor=shadespec, alpha=0.8)