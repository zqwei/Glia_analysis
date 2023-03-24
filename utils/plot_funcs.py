import numpy as np
import os, sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, zscore
from tqdm import tqdm
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5, style='ticks')


def plot_shade_err(x, y, axis=-1, plt=plt, linespec='-k', shadespec='k'):
    from scipy.stats import sem
    mean_ = np.nanmean(y, axis=axis)
    error = sem(y, axis=axis, nan_policy='omit')
    plt.plot(x, mean_, linespec)
    plt.fill_between(x, mean_-error, mean_+error, edgecolor=None, linewidth=0.0, facecolor=shadespec, alpha=0.8)