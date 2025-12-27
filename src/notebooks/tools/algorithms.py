import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as spstats


def fft(v: pd.DataFrame | np.ndarray, fs: float = 1.0) -> tuple[float, float]:
    """
    implementazione della FFT

    :returns: frequenze e trasformata di fourie
    """
    if isinstance(v, pd.DataFrame):
        v = v.values.squeeze()

    if v.ndim != 1:
        raise ValueError("Input must be a 1-D signal")

    n = v.size
    yf = sp.fft.fft(v)
    xf = sp.fft.fftfreq(n, d=1 / fs)

    return xf, yf

def rms(signal):
    return np.sqrt(np.mean(signal**2))

def moving_rms_with_stop(signal, stop, N, o=None):
    """
    signal : array-like (1D)
    stop   : DataFrame con index = sample in cui avviene la riparazione
    N      : window size
    o      : step size (default = N, no overlap)
    """

    if o is None:
        o = N

    stop_points = stop.index.to_numpy()
    stop_ptr = 0

    res = {}
    group_id = 0
    res[group_id] = []

    i = 0
    L = len(signal)

    while i + N <= L:

        # se supero il prossimo evento di riparazione
        if stop_ptr < len(stop_points) and i >= stop_points[stop_ptr]:
            group_id += 1
            res[group_id] = []
            stop_ptr += 1

        window = signal[i:i + N]
        res[group_id].append(rms(window))

        i += o

    return res

def moving_rms(signal, N, o = None):
    rmss = []
    i = 0
    if o is None:
        o = N
    while i + N < len(signal):
        rmss.append(rms(signal[i:i+N]))
        i += o
    return rmss

def shape_factor(signal):
    mav = np.mean(np.abs(signal))
    return rms(signal)/mav

def skewness(signal, bias=False):
    return spstats.skew(signal, bias)

def kurtosis(signal, fisher=True, bias=False):
    return spstats.kurtosis(signal, fisher, bias)
