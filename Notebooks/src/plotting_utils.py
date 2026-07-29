'''
Shared plotting helpers for the Notebooks/ figure notebooks.
'''

import numpy as np


def plot_shade_err(x, y, axis=-1, plt=None, linespec='-k', shadespec='k', err_f=1.):
    from scipy.stats import sem
    if plt is None:
        import matplotlib.pyplot as plt
    mean_ = np.nanmean(y, axis=axis)
    error = sem(y, axis=axis, nan_policy='omit')/err_f
    plt.plot(x, mean_, linespec)
    plt.fill_between(x, mean_-error, mean_+error, edgecolor=None, linewidth=0.0, facecolor=shadespec, alpha=0.8)
