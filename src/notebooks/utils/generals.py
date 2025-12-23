import os.path as path
import numpy as np
import os
import pandas as pd
from typing import Type
from enum import Enum

PLOT_PATH=f"./img/"
DATA_PATH=f"../../Data/PHM2025_training_data/training_data.csv"


def iter_enum(S):
    for p in S:
        try:
            assert(issubclass(p.value, Type[Enum]))
            iter_enum(p.value)
        except (AssertionError, TypeError):
            yield p.value


def filter(df, key, val, cols=None) -> pd.DataFrame:
    if cols is None:
        return df.loc[df[key] == val].copy()
    return df.loc[df[key] == val, cols].copy()

def load_training():
    """
    Carica il dataset di training
    """
    with open(DATA_PATH, "r") as f:
        return pd.read_csv(f)


def get_shift(col, df):
    """
    Ritorna i punti in cui il valore successivo è minore del precedente
    
    :param col: Colonna da controllare
    :param df: DataFrame 
    """
    return df.loc[df[col] > df[col].shift(1), ["ESN", col]].copy()

ESN = [101, 102, 103, 104]

Snapshot = [1, 2, 3, 4, 5, 6, 7, 8]

def sensors_subset(df):
    return df.loc[:, SENSORS.tolist()].copy()

def plot_path(dirname, *args, filename=None):
    full_path = path.join(PLOT_PATH, dirname, *map(str, args))
    os.makedirs(full_path, exist_ok=True)
    if filename:
        full_path = os.path.join(full_path, filename)
    return full_path

def my_fft(v: pd.DataFrame | np.dtype):
    pass



class SENSORS(Enum):
    Sensed_Altitude="Sensed_Altitude"
    Sensed_Mach="Sensed_Mach"
    Sensed_Pamb="Sensed_Pamb"
    Sensed_Pt2="Sensed_Pt2"
    Sensed_TAT="Sensed_TAT"
    Sensed_WFuel="Sensed_WFuel"
    Sensed_VAFN="Sensed_VAFN"
    Sensed_VBV="Sensed_VBV"
    Sensed_Fan_Speed="Sensed_Fan_Speed"
    Sensed_Core_Speed="Sensed_Core_Speed"
    Sensed_T25="Sensed_T25"
    Sensed_T3="Sensed_T3"
    Sensed_Ps3="Sensed_Ps3"
    Sensed_T45="Sensed_T45"
    Sensed_P25="Sensed_P25"
    Sensed_T5="Sensed_T5"

    @staticmethod
    def tolist():
        return list(iter_enum(SENSORS))

    @staticmethod
    def iter():
        return list(iter_enum(SENSORS))


