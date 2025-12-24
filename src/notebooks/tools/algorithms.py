import pandas as pd
import numpy as np
import scipy as sp


def fft(v: pd.DataFrame | np.ndarray, fs: float = 1.0):
    if isinstance(v, pd.DataFrame):
        v = v.values.squeeze()

    if v.ndim != 1:
        raise ValueError("Input must be a 1-D signal")

    n = v.size
    yf = sp.fft.fft(v)
    xf = sp.fft.fftfreq(n, d=1 / fs)

    return xf, yf
