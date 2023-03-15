import numpy as np
import multiprocessing as mp
from scipy.stats import zscore, spearmanr


def parallel_to_chunks(numsplit, func1d, arr, *args, **kwargs):
    mp_count = min(mp.cpu_count(), arr.shape[0]) # fix the error if arr is shorter than cpu counts
    print(f'Number of processes to parallel: {mp_count}')
    chunks = [(func1d, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, numsplit)]
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


def zscore_(x, axis=1):
    return (x - x.mean(axis=axis, keepdims=True))/x.std(axis=axis, keepdims=True),


def smooth_boxcar(x, slide_window=2, axis=1):
    N = slide_window*2+1
    cumsum = np.cumsum(x, axis=1)
    return (cumsum[:, N:] - cumsum[:, :-N])/N # boxcar smooth


def spearmanr_vec(a, vec=np.zeros((1, 1000)), axis=1):
    c, p = spearmanr(a, vec, axis=axis)
    return c[-1][:-1],