import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr, linregress, wilcoxon
from scipy.stats import zscore, mannwhitneyu
sns.set(font_scale=1.5, style='ticks')
import warnings
warnings.filterwarnings('ignore')


def mid_mean(x, p_ = 15):
    x_low, x_high = np.percentile(x, [p_, 100-p_])
    return x[(x>x_low) & (x<x_high)].mean()


def plot_shade_err(x, y, axis=-1, plt=plt, linespec='-k', shadespec='k'):
    from scipy.stats import sem
    mean_ = np.nanmean(y, axis=axis)
    error = sem(y, axis=axis, nan_policy='omit')
    plt.plot(x, mean_, linespec)
    plt.fill_between(x, mean_-error, mean_+error, edgecolor=None, facecolor=shadespec, alpha=0.8)