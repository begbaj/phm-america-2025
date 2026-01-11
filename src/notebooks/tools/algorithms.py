import itertools
from re import I
from time import monotonic
import pandas as pd
import numpy as np
import progpy.datasets
import progpy.datasets.nasa_battery
import progpy.metrics
import progpy.mixture_of_experts
import scipy as sp
import scipy.stats as spstats
import progpy
import numba


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

def rms(x):
    return np.sqrt(np.mean(x**2))

def rms_signal(signal):
    return rms(np.array(signal))

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
        res[group_id].append(rms_signal(window))

        i += o

    return res

def moving_rms(signal, N, o = None):
    rmss = []
    i = 0
    if o is None:
        o = N
    while i + N < len(signal):
        rmss.append(rms_signal(signal[i:i+N]))
        i += o
    return rmss

def shape_factor(signal):
    mav = np.mean(np.abs(signal))
    return rms_signal(signal)/mav

def skewness(signal, bias=False):
    return spstats.skew(signal, bias)

def kurtosis(signal, fisher=True, bias=False):
    return spstats.kurtosis(signal, fisher, bias)

def moving_features_with_stop(signal: list[tuple[int, float]], stop: list[int], N, step=None):
    if step is None:
        step = N

    def new_group():
        return {"rms": [], "mean": [], "std": [], "kurtosis": [], "skewness": [], "shape_factor": []}

    group_id = 0
    res = {}
    res[group_id] = new_group()

    i = 0
    L = len(signal)
    stop_ptr = 0
    
    # Ensure stop list is sorted, otherwise logic breaks
    stop = sorted(stop)

    while i + N <= L:
        # Check if current signal index matches or passes the next stop point
        if stop_ptr < len(stop) and signal[i][0] >= int(stop[stop_ptr]):
            stop_ptr += 1
            group_id += 1
            res[group_id] = new_group()
            # We continue to re-evaluate in case multiple stops overlap
            # or to proceed to processing with the new group_id
            continue

        # Extract values for the window
        window_list = [a[1] for a in signal[i:i + N]]
        
        # FIX: Convert to numpy array for math operations
        window = np.array(window_list)

        if len(window) == 0:
            i += step
            continue

        # Vectorized calculations are safer on np.array
        m = np.mean(window)
        s = np.std(window)
        
        # Use np.square to handle the array/list safe RMS calculation
        r = np.sqrt(np.mean(np.square(window))) 

        res[group_id]["mean"].append(m)
        res[group_id]["std"].append(s)
        res[group_id]["rms"].append(r)

        res[group_id]["kurtosis"].append(float(spstats.kurtosis(window, axis=0, fisher=True)))
        res[group_id]["skewness"].append(float(spstats.skew(window, axis=0)))
        
        # Safety check for division by zero
        mean_abs = np.mean(np.abs(window))
        sf = r / mean_abs if mean_abs != 0 else 0
        res[group_id]["shape_factor"].append(sf)

        i += step

    return res

def monotonicity(signal):
    diffs = np.diff(signal)
    n = len(diffs)
    pos_diffs = np.sum(diffs > 0)
    neg_diffs = np.sum(diffs < 0)
    return np.abs(pos_diffs - neg_diffs) / n


def evaluate_feature_groups_stats(featgroups: dict[str, list[tuple[int, float]]]):
    results = {}
    for feat_name, signal in featgroups.items():
        m = monotonicity(signal)

        results[feat_name] = {}
        results[feat_name]["monotonicity"] = m
    return results
