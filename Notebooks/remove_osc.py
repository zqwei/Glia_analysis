import numpy as np


highPassFilterCircleLengthByPoint = 50
harmonicPeakLst = np.array([1, 2, 4])
filterWindowsSizeMean = 3
method="average"

def remove_osc(data, dataBackground):
    F0 = findFundamentalFrequency(dataBackground.squeeze())
    return removeHarmonicNoise(data.squeeze(),F0)

# find fundamental frequency
def findFundamentalFrequency(y_org,highPassFilterCircleLengthByPoint=highPassFilterCircleLengthByPoint):
    import scipy.fft as fft
    L = len(y_org)
    paddingSize = L*5
    LowFs = paddingSize//highPassFilterCircleLengthByPoint
    y_withPadding = np.zeros(paddingSize)
    y_withPadding[:L] = y_org
    power_=np.abs(fft.fft(y_withPadding)[:(paddingSize//2)])
    power_[:LowFs]=0;
    idx_=np.argmax(power_);
    return idx_/(paddingSize-1);


# reconstruction
def removeHarmonicNoise(data,F0,filterWindowsSizeMean=filterWindowsSizeMean,harmonicPeakLst=harmonicPeakLst,method=method):
    L = len(data)
    timeWindow = np.round(filterWindowsSizeMean/F0).astype('int')
    CircleLength = np.round(1/F0)
    harmonicFreqs = F0*harmonicPeakLst
    y_re = data
    if method=="gaussian":
        from scipy.stats import norm
        import scipy.signal as sps
        HalfTimeWindow = timeWindow//2
        GaussianWindow = norm.pdf(np.arange(-HalfTimeWindow,HalfTimeWindow)/CircleLength)
        GaussianWindow = GaussianWindow/np.sum(GaussianWindow)
        for harmonicFreq in harmonicFreqs:
            y_sin = np.sin(np.arange(L)*2*np.pi*harmonicFreq)
            y_cos = np.cos(np.arange(L)*2*np.pi*harmonicFreq)
            y_sin_amp_p = y_re*y_sin
            y_cos_amp_p = y_re*y_cos
            y_sin_amp = sps.lfilter(GaussianWindow,1,y_sin_amp_p)*2
            y_cos_amp = sps.lfilter(GaussianWindow,1,y_cos_amp_p)*2
            y_re = y_re-y_sin_amp*y_sin
            y_re = y_re-y_cos_amp*y_cos
    else:
        for harmonicFreq in harmonicFreqs:
            y_sin = np.sin(np.arange(L)*2*np.pi*harmonicFreq)
            y_cos = np.cos(np.arange(L)*2*np.pi*harmonicFreq)
            y_sin_amp_p = y_re*y_sin
            y_cos_amp_p = y_re*y_cos   
            y_sin_amp = movmean(y_sin_amp_p, timeWindow)*2
            y_cos_amp = movmean(y_cos_amp_p, timeWindow)*2
            y_re = y_re-y_sin_amp*y_sin
            y_re = y_re-y_cos_amp*y_cos            
    return y_re


def movmean(x, N):
    import pandas as pd
    return pd.Series(x).rolling(window=N, min_periods=1, center=True).mean().values