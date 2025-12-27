import os.path as path
import os
import pandas as pd
import numpy as np
from types import FunctionType
from pandas import DataFrame, Series
from enum import Enum
from plotly.graph_objs import Data
from tools.config import DATA_PATH, PLOT_PATH
import tools.config as c

###
### ENUMS
###


class ESENSORS(Enum):
    """
    è una lista dei sensori presenti nel dataset
    serve solo per evitare di scrivere a mano i nomi dei sensori
    """

    Sensed_Altitude = "Sensed_Altitude"
    Sensed_Mach = "Sensed_Mach"
    Sensed_Pamb = "Sensed_Pamb"
    Sensed_Pt2 = "Sensed_Pt2"
    Sensed_TAT = "Sensed_TAT"
    Sensed_WFuel = "Sensed_WFuel"
    Sensed_VAFN = "Sensed_VAFN"
    Sensed_VBV = "Sensed_VBV"
    Sensed_Fan_Speed = "Sensed_Fan_Speed"
    Sensed_Core_Speed = "Sensed_Core_Speed"
    Sensed_T25 = "Sensed_T25"
    Sensed_T3 = "Sensed_T3"
    Sensed_Ps3 = "Sensed_Ps3"
    Sensed_T45 = "Sensed_T45"
    Sensed_P25 = "Sensed_P25"
    Sensed_T5 = "Sensed_T5"

    @classmethod
    def values(cls) -> list[str]:
        """Ritorna la lista dei valori."""
        return [e.value for e in cls]

    @classmethod
    def iter(cls) -> list[str]:
        """DEPRECATO, usa values. Ritorna la lista dei valori."""
        return [e.value for e in cls]

    @classmethod
    def members(cls) -> list["ESENSORS"]:
        """Ritorna la lista dei membri Enum."""
        return list(cls)

ESN = range(101, 105)
"""
for esn in u.ESN:
"""

SNAPSHOTS = range(1, 9)
"""
for esn in u.SNAPSHOTS:
"""

SENSORS = ESENSORS.values()


def plot_path(dirname: str, *args, filename=None) -> str:
    """
    Genera il path per il plot e crea le cartelle se non esistono

    """
    full_path = path.join(PLOT_PATH, dirname, *map(str, args))
    os.makedirs(full_path, exist_ok=True)
    if filename:
        full_path = os.path.join(full_path, filename)
    return full_path

def ess_iter(df: DataFrame, order=["esn", "sensor", "snapshot"]):
    order = [o.lower() for o in order]
    order_enum = {
        "esn": ESN,
        "sensor": SENSORS,
        "snapshot": SNAPSHOTS,
    }
    esn: int = 0
    sensor: str = "Sensor"
    snapshot: int = 0

    def matcher(o: int, v, oesn, osensor, osnapshot) -> tuple[int,str,int]:
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
        esn, sensor, snapshot = matcher(0,A, esn, sensor, snapshot)
        for B in order_enum[order[1]]:
            esn, sensor, snapshot = matcher(1,B, esn, sensor, snapshot)
            for C in order_enum[order[2]]:
                esn, sensor, snapshot = matcher(2,C, esn, sensor, snapshot)
                yield df_ess_filter(df, esn, sensor, snapshot, all=True)

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

###
### DATA
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

def load_forward_fill() -> FunctionType:
    with open(c.DATA_TRAINING_PATH + "training_ffill.csv", "r") as f:
        return WrapData(pd.read_csv(f))



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
### DATAFRAME MANIPULATION AND LOGICS
###

def df_ess_filter(df: DataFrame, esn: int, sensor: str, snapshot: int, all=False):
    if all:
        return df.loc[
            (df["ESN"] == esn) & (df["Snapshot"] == snapshot),
            [sensor]
        ], esn, sensor, snapshot
    else:
        return df.loc[
            (df["ESN"] == esn) & (df["Snapshot"] == snapshot),
            [sensor]
        ]


def df_avg_stdd_cycles_to_event(df: DataFrame, groupby: str = "ESN") -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
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



