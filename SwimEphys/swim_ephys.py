import numpy as np


def load(in_file, num_channels=10, memmap=False):
    """Load multichannel binary data from disk, return as a [channels,samples] sized numpy array
    """
    from numpy import fromfile, float32

    if memmap:
        from numpy import memmap

        data = memmap(in_file, dtype=float32)
    else:
        with open(in_file, "rb") as fd:
            data = fromfile(file=fd, dtype=float32)
    trim = data.size % num_channels
    # transpose to make dimensions [channels, time]
    data = data[: (data.size - trim)].reshape(data.size // num_channels, num_channels).T
    if trim > 0:
        print("Data needed to be truncated!")

    return data



def windowed_variance(signal, kern_mean=None, kern_var=None, fs=6000):
    """
    Estimate smoothed sliding variance of the input signal
    signal : numpy array
    kern_mean : numpy array
        kernel to use for estimating baseline
    kern_var : numpy array
        kernel to use for estimating variance
    fs : int
        sampling rate of the data
    """
    from scipy.signal import gaussian, fftconvolve

    # set the width of the kernels to use for smoothing
    kw = int(0.04 * fs)

    if kern_mean is None:
        kern_mean = gaussian(kw, kw // 10)
        kern_mean /= kern_mean.sum()

    if kern_var is None:
        kern_var = gaussian(kw, kw // 10)
        kern_var /= kern_var.sum()

    mean_estimate = fftconvolve(signal, kern_mean, "same")
    var_estimate = (signal - mean_estimate) ** 2
    fltch = fftconvolve(var_estimate, kern_var, "same")

    return fltch, var_estimate, mean_estimate


def estimate_swims(signal, fs=6000, scaling=1.6):
    """ Estimate swim timing from ephys recording of motor neurons
    Parameters
    __________
    signal : numpy array, 1 dimensional. Windowed variance of ephys signal.
    fs : int
        sampling rate of the data, in Hz
    """

    from numpy import zeros, where, diff, concatenate

    # set dead time between peaks, in seconds. This prevents duplicate swims.
    dead_time = 0.010 * fs

    # set minimum duration between swim bursts in seconds
    inter_swim_min = 0.12 * fs

    # estimate swim threshold
    thr = estimate_threshold(signal, fs * 60, scaling=scaling)

    peaksT, peaksIndT = estimate_peaks(signal, dead_time)

    burstIndT = peaksIndT[where(signal[peaksIndT] > thr[peaksIndT])]
    burstT = zeros(signal.shape)
    burstT[burstIndT] = 1

    interSwims = diff(burstIndT)
    swimEndIndB = where(interSwims > inter_swim_min)[0]
    swimEndIndB = concatenate((swimEndIndB, [burstIndT.size - 1]))

    swimStartIndB = swimEndIndB[:-1] + 1
    swimStartIndB = concatenate(([0], swimStartIndB))
    nonShort = where(swimEndIndB != swimStartIndB)[0]
    swimStartIndB = swimStartIndB[nonShort]
    swimEndIndB = swimEndIndB[nonShort]

    bursts = zeros(signal.size)
    starts = zeros(signal.size)
    stops = zeros(signal.size)
    bursts[burstIndT] = 1
    starts[burstIndT[swimStartIndB]] = 1
    stops[burstIndT[swimEndIndB]] = 1

    return starts, stops, thr


def estimate_peaks(signal, dead_time):
    """
    Estimate peak times in a signal, with a minimum distance between estimated peaks.
    Parameters
    __________
    signal : numpy array, 1-dimensional
    dead_time : int
        minimum number of sample between estimated peaks
    """

    from numpy import diff, where, zeros

    aa = diff(signal)
    peaks = (aa[:-1] > 0) * (aa[1:] < 0)
    inds = where(peaks)[0]

    # take the difference between consecutive indices
    d_inds = diff(inds)

    # find differences greater than deadtime
    to_keep = d_inds > dead_time

    # only keep the indices corresponding to differences greater than deadT
    inds[1:] = inds[1:] * to_keep
    inds = inds[inds.nonzero()]

    peaks = zeros(signal.shape[0])
    peaks[inds] = 1

    return peaks, inds


def estimate_threshold(signal, window=180000, scaling=1.6, lower_percentile=0.01):
    """
    Return non-sliding windowed threshold of input ndarray vec.
    Parameters
    ----------
    signal : ndarray, 1-dimensional
        Input array from which to estimate a threshold
    window : int
        Step size / window length for the resulting threshold.
    scaling : float or int
        scaling factor applied to estimated spread of the noise distribution of vec. Sets magnitude of threshold
        relative to the estimated upper bound of the noise distribution.
    lower_percentile : float
        Percentile of signal to use when estimating the lower bound of the noise distribution.
    """
    from numpy import zeros, percentile, arange, median

    th = zeros(signal.shape)
    for t in arange(0, signal.size - window, window):
        plr = arange(t, min(t + window, signal.size))
        sig = signal[plr]
        med = median(sig)
        bottom = percentile(sig, lower_percentile)
        th[t:] = med + scaling * (med - bottom)
    return th


def estimate_bot_threshold(signal, window=180000, lower_percentile=0.01):
    from numpy import zeros, percentile, arange, median
    th = zeros(signal.shape)
    for t in arange(0, signal.size - window, window):
        plr = arange(t, min(t + window, signal.size))
        sig = signal[plr]
        med = median(sig)
        th[t:] = percentile(sig, lower_percentile)
    return th


def ind2frame(x, ind):
    x_ = x.copy()
    for n, _ in enumerate(x):
        x_[n] = (_>ind).sum()
    return x_


def ep2frame(camtrig, thres=3.8):
    arr_ = (camtrig>thres).astype('int')
    return np.where((arr_[:-1]-arr_[1:])==-1)[0]+1