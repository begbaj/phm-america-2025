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
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

def train_models(df, operating_vars, degradation_vars) -> dict[int, dict[str,LinearRegression]]:
    X_train = df[operating_vars]
    Y_train = df[degradation_vars]
    models = {}
    for i in range(0,8):
        X_temp = pd.DataFrame(np.roll(X_train, i, axis=1))
        models[i] = {}
        models[i]["model"] = train_model(X_temp, Y_train)
    return models

def train_model(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def s_pred(s_o, model):
    return model.predict(s_o)

def residual(s_d, s_o, model):
    return s_d - s_pred(s_o, model)

def wind(y_p, y, a):
    diff = y - y_p
    num = np.where(diff >= 0, 2.0, 1.0)
    if isinstance(y_p, pd.DataFrame) or isinstance(y_p, pd.Series):
        y_p = y_p.values
    return num / (1 + a * y_p)

def TWE(y_p, y, a, b):
    if isinstance(y_p, pd.DataFrame): y_p = y_p.values
    weight = wind(y_p, y, a)
    squared_error = (y - y_p) ** 2
    return weight * squared_error * b

def HI(T3_res, T45_res, alpha):
    return -alpha * T3_res - T45_res

def objective(alpha, T3, T45, RUL):
    hi = -alpha*T3 - T45
    RUL = RUL.dropna()
    hi = hi.dropna()
    corr = stats.pearsonr(RUL,hi)
    # return np.sqrt(np.mean((hi - RUL)**2)) + 1
    return - corr[0]

def objective_beta(params, T3, T45, RUL):
    alpha, beta = params
    hi = -alpha*T3 - beta*T45
    RUL = RUL.dropna()
    hi = hi.dropna()
    corr = stats.pearsonr(RUL,hi)
    # return np.sqrt(np.mean((hi - RUL)**2)) + 1
    return - corr[0]

def HIE(params, vars):
    #return np.sum([-params[i]*vars.iloc[:,i] for i in range(0, 8)])
    return vars.dot(-np.array(params))

def objective_experimental(params, vars, RUL):
    hi = HIE(params, vars)
    # RUL = RUL.dropna()
    corr = stats.pearsonr(RUL,hi)
    return -corr[0]

def objective_deviation(params, vars, RUL):
    hi = HIE(params, vars)
    hi_min, hi_max = hi.min(), hi.max()
    if hi_max == hi_min:
        # Penalità se l'HI è una linea piatta
        return 1.0
    hi_norm = (hi - hi_min) / (hi_max - hi_min)
    mse = np.mean((hi_norm - RUL)**2)
    return mse

def get_rolling_slope_intercept(series, window):
    slopes = []
    intercepts = []
    series = np.asarray(series).flatten()
    for i in range(len(series)):
        if i < window:
            slopes.append(0.0)
            intercepts.append(0.0)
        else:
            y = series[i-window:i]
            x = np.arange(window)
            # Fit polinomiale di grado 1 (retta) -> ritorna [slope, intercept]
            poly = np.polyfit(x, y, 1)
            slopes.append(float(poly[0]))
            intercepts.append(float(poly[1]))
    return np.array(slopes), np.array(intercepts)

def get_slope(y):
    """Calcola la pendenza della retta di regressione per una finestra y"""
    x = np.arange(len(y))
    # Polyfit di grado 1 restituisce [pendenza, intercetta]
    slope = np.polyfit(x, y, 1)[0]
    return slope


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

def looc(df: pd.DataFrame, model, features: list[str], target: str) -> dict[str, float]:
    """
    Leave-one-out cross-validation
    """
