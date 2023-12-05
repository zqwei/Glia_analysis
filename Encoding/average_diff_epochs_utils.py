import numpy as np
import os, sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.5, style='ticks')

from scipy.stats import mannwhitneyu
import multiprocessing as mp
def parallel_to_chunks(func1d, arr, *args, **kwargs):
    if mp.cpu_count() == 1:
        raise ValueError('Multiprocessing is not running on single core cpu machines, and consider to change code.')

    mp_count = min(mp.cpu_count(), arr.shape[0]) # fix the error if arr is shorter than cpu counts
    print(f'Number of processes to parallel: {mp_count}')
    chunks = [(func1d, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, mp_count)]
    pool = mp.Pool(processes=mp_count)
    individual_results = pool.map(unpacking_apply_func, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    results = ()
    # print(len(individual_results[0]))
    for i_tuple in range(len(individual_results[0])):
        results = results + (np.concatenate([_[i_tuple] for _ in individual_results]), )
    return results


def unpacking_apply_func(list_params):
    func1d, arr, args, kwargs = list_params
    return func1d(arr, *args, **kwargs)


def diff_p_value(mat):
    num_cells = mat.shape[0]
    p_mat = np.zeros(num_cells)
    for n_cell in range(num_cells):
        x = mat[n_cell, ~CL_trial]
        y = mat[n_cell, CL_trial]
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        _, p_mat[n_cell] = mannwhitneyu(x, y)
    return p_mat,