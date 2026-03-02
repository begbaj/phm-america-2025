from datetime import datetime
from enum import Enum
from pandas import DataFrame, Series
from plotly.graph_objs import Data
from scipy.optimize import minimize
from . import plotting
from .config import DATA_PATH, PLOT_PATH
from .types.enums import ESENSORS, SENSORS as SENSORS_ENUM, RepairEventType, Snapshots
from types import FunctionType
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
import pandas as pd
import random
from . import config as c
from . import features as f

###
# ENUMS
###

EVENTS = ["ww", "hpc", "hpt"]

ESN = range(101, 105)
"""
for esn in u.ESN:
"""
SNAPSHOTS = range(1, 9)
# Use unprefixed sensor names present in the dataset (e.g., 'Altitude', 'Mach')
SENSORS = SENSORS_ENUM.values()
FEATURES = ["mean", "std", "kurtosis", "skewness"]
META_COLS = [
    "Cycles_Since_New",
    "ESN",
    "esn_index",
    "fault_hpc_cycle",
    "fault_hpt_cycle",
    "fault_ww_cycle",
    "global_index",
    "Cumulative_HPC_SVs",
    "hpc_cycle_index",
    "Cumulative_HPT_SVs",
    "hpt_cycle_index",
    "Snapshot",
    "snap_index",
    "Cycles_to_HPC_SV",
    "Cycles_to_HPT_SV",
    "Cycles_to_WW",
    "Cumulative_WWs",
    "ww_cycle_index",
]


def get_timestamp():
    """
    Restituisce il timestamp attuale nel formato stringa YYMMHHmm.
    Esempio: 12 gennaio 2026 alle 10:45 -> '26011045'
    """
    return datetime.now().strftime("%y%m%H%M")


def preproc_features(
    dfi, group, incols, features, sortcol, window_size=None, step=1, outcols=None
):
    f._feature_aggregator(
        dfi,
        group,
        incols,
        features,
        sortcol,
        window_size=window_size,
        step=step,
        outcols=outcols,
    )


def plot_path(dirname: str, *args, filename=None) -> str:
    """
    Genera il path per il plot e crea le cartelle se non esistono

    """
    return pathfinder(PLOT_PATH, dirname, *map(str, args), filename=filename)


def pathfinder(dirname: str, *args, filename=None):
    full_path = path.join(dirname, *map(str, args))
    os.makedirs(full_path, exist_ok=True)
    if filename:
        full_path = os.path.join(full_path, filename)
    return full_path


def ess_iter(
    df: DataFrame, order=["esn", "sensor", "snapshot"], plotdata=False, rand=False
):
    order = [o.lower() for o in order]
    order_enum = {
        "esn": ESN,
        "sensor": ESENSORS.values(),
        "snapshot": Snapshots.values(),
        "event": RepairEventType.values(),
    }

    if rand:
        order_enum = {k: random.sample(list(v), len(v))
                      for k, v in order_enum.items()}

    esn: int = 0
    sensor: str = "Sensor"
    snapshot: int = 0

    def matcher(o: int, v, oesn, osensor, osnapshot) -> tuple[int, str, int]:
        match order[o]:
            case "esn":
                return int(v), osensor, int(osnapshot)
            case "sensor":
                return int(oesn), v, int(osnapshot)
            case "snapshot":
                return int(oesn), osensor, int(v)
            case _:
                return int(oesn), osensor, int(osnapshot)

    for A in order_enum[order[0]]:
        esn, sensor, snapshot = matcher(0, A, esn, sensor, snapshot)
        for B in order_enum[order[1]]:
            esn, sensor, snapshot = matcher(1, B, esn, sensor, snapshot)
            for C in order_enum[order[2]]:
                esn, sensor, snapshot = matcher(2, C, esn, sensor, snapshot)
                yield df_ess_filter(df, esn, sensor, snapshot, all=True, pdata=plotdata)


def iter_enum(S):
    """
    Funzione che itera un enum
    """
    for p in S:
        try:
            assert issubclass(p.value, Enum)
            iter_enum(p.value)
        except (AssertionError, TypeError):
            yield p.value


def to_signal(df, column) -> list[tuple[int, float]]:
    ixs = df.index
    vals = df[column].values

    if len(ixs) != len(vals):
        print("Length of ixs and vals are not the same")
        return [(0, 0)]

    return list(zip(ixs, vals))


###
# DATA
###


def WrapData(data: DataFrame):
    """
    Generator di dati.

    :param data: Description
    :type data: DataFrame
    """

    def access() -> DataFrame:
        return data.copy()

    return access


def load_training() -> FunctionType:
    """
    Carica il dataset di training
    """
    with open(c.DATA_PATH, "r") as f:
        return WrapData(pd.read_csv(f))


def load_ffill_training() -> FunctionType:
    """
    Carica il dataset di training
    """
    with open(c.DATA_PATH_FFILL, "r") as f:
        return WrapData(pd.read_csv(f))


def load_forward_fill() -> FunctionType:
    with open(c.DATA_TRAINING_PATH + "training_ffill.csv", "r") as f:
        return WrapData(pd.read_csv(f))


def load_smooth_training(orig: FunctionType, span: int) -> DataFrame:
    """
    Genera il dataset con smoothing

    :param orig: dataframe da filtrare
    :type orig: DataFrame
    :param span: finestra di punti considerati
    :type span: int
    """
    data = orig()
    try:
        for idx, sensor in enumerate(ESENSORS.values()):
            data[sensor] = data.groupby(["ESN", "Snapshot"], group_keys=False)[
                sensor
            ].transform(lambda x: x.ewm(span=span, adjust=False).mean())
        print(f"Dataset filtrato con successo")
    except:
        print("Errore nel filtraggio del dataset")
    return WrapData(data)


def load_testing() -> FunctionType:
    """
    Carica il dataset di training
    """
    with open(DATA_PATH, "r") as f:
        return WrapData(pd.read_csv(f))


def load_validation() -> FunctionType:
    """
    Carica il dataset di training
    """
    with open(DATA_PATH, "r") as f:
        return WrapData(pd.read_csv(f))


def load_event_points(
    df: DataFrame,
) -> tuple[FunctionType, FunctionType, FunctionType] | tuple[None, None, None]:
    """
    Bisogna dare come argomento il dataset dal quale
    estrarre gli eventi

    usa la funzione df_get_step_points su colonne predefinite.
    """

    wws = df_get_step_points(df, "Cumulative_WWs")
    hpc = df_get_step_points(df, "Cumulative_HPC_SVs")
    hpt = df_get_step_points(df, "Cumulative_HPT_SVs")

    if (
        isinstance(wws, DataFrame)
        and isinstance(hpc, DataFrame)
        and isinstance(hpt, DataFrame)
    ):
        return WrapData(wws), WrapData(hpc), WrapData(hpt)
    return None, None, None


###
# DATAFRAME MANIPULATION AND LOGICS
###


def df_reset_index(df: DataFrame, base=0):
    df.index -= base


def df_ess_filter(
    df: DataFrame, esn: int, sensor: str, snapshot: int, all=False, pdata=False
):
    if all:
        if pdata:
            pdd = plotting.PlotData()
            pdd.esn = esn
            pdd.sensor = sensor
            pdd.snap = snapshot
            return df.loc[
                (df["ESN"] == esn) & (df["Snapshot"] == snapshot), [sensor]
            ], pdd
        return (
            df.loc[(df["ESN"] == esn) & (
                df["Snapshot"] == snapshot), [sensor]],
            esn,
            sensor,
            snapshot,
        )
    else:
        return df.loc[(df["ESN"] == esn) & (df["Snapshot"] == snapshot), [sensor]]


def df_avg_stdd_cycles_to_event(
    df: DataFrame, groupby: str = "ESN"
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    rww = []
    rhpc = []
    rhpt = []

    ww = df.groupby("Cumulative_WWs").size()
    hpc = df.groupby("Cumulative_HPC_SVs").size()
    hpt = df.groupby("Cumulative_HPT_SVs").size()

    rww = (ww.mean(), ww.std())
    rhpc = (hpc.mean(), hpc.std())
    rhpt = (hpt.mean(), hpt.std())

    return rww, rhpc, rhpt


def df_row_filter(df: DataFrame | Series, val=0) -> DataFrame | Series | None:
    """
    restituisce un dataframe filtrato per riga
    """
    try:
        a = df[(df.T != val)]
        if isinstance(a, DataFrame) or isinstance(a, Series):
            return a
        return None
    except Exception:
        print("NON PRATICABILE")
        return df


def df_col_filter(df: DataFrame, val=0) -> DataFrame | None:
    """
    restituisce un dataframe filtrato colonna
    """
    try:
        a = df.loc[:, (df != 0).any(axis=0)]
        if isinstance(a, DataFrame):
            return a
        return None
    except Exception:
        print("NON PRATICABILE")
        return df


def df_get_step_points(df: DataFrame, column: str, up=False) -> DataFrame | None:
    """
    Restituisce in output una lista di record che corrispondono
    ai record subito prima di uno step.
    Uno step è definito come un valore che va nella direzione opposta a quella attesa.
    """
    if up:
        a = df[df[column] < df[column].shift(1)]
    else:
        a = df[df[column] > df[column].shift(1)]
    if isinstance(a, DataFrame):
        return a
    return None


def df_filter_by_key(df: DataFrame, col: str, val, cols=None) -> DataFrame:
    """
    filtra il DataFrame in base al valore che assume una colonna.
    Equivalente di WHERE in SQL.
    shortcut per df.loc[df[col] == val]
    """
    if cols is None:
        return df.loc[df[col] == val].copy()
    return df.loc[df[col] == val, cols].copy()


def df_get_shift(df: DataFrame, col: str) -> DataFrame:
    """
    Ritorna i punti in cui il valore successivo è minore del precedente

    :param df: DataFrame
    :param col: Colonna da controllare
    """
    return df.loc[df[col] > df[col].shift(1), ["ESN", col]].copy()


def df_sensors_subset(df: DataFrame) -> DataFrame:
    """
    Ritorna il dataframe con solo i sensori
    shortcut per df.loc[:, SENSORS]
    """
    return df.loc[:, SENSORS].copy()


###
# GLOBAL STATE
###
DEBUG_ENABLED = False

def set_debug(enabled: bool):
    global DEBUG_ENABLED
    DEBUG_ENABLED = enabled

def debug_print(*args, **kwargs):
    if DEBUG_ENABLED:
        print(*args, **kwargs)

###
# SCORING
###

def calculate_twe(y_true, y_pred, alpha=0.001, beta=1.0) -> np.ndarray:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = y_pred - y_true
    denom = 1 + alpha * y_true
    w = np.where(diff >= 0, 2.0 / denom, 1.0 / denom)
    return w * (diff**2) * beta

def target_score(all_y_true, all_y_pred, alpha=0.001, beta=1.0):
    twe_sum  = calculate_twe(all_y_true, all_y_pred, alpha, beta)
    return np.mean(twe_sum)

def calculate_score(ww: tuple, hpt: tuple, hpc: tuple, alpha=0.001, beta=1.0):
    ww_score = target_score(ww[0], ww[1], alpha, beta)
    hpt_score = target_score(hpt[0], hpt[1], alpha, beta)
    hpc_score = target_score(hpc[0], hpc[1], alpha, beta)
    return np.mean([ww_score, hpt_score, hpc_score])
