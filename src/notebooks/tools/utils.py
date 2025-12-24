import os.path as path
import os
import pandas as pd
from pandas import DataFrame
from enum import Enum
from config import DATA_PATH, PLOT_PATH

ESN = range(101, 105)
"""
for esn in u.ESN:
"""

SNAPSHOTS = range(1, 9)
"""
for esn in u.SNAPSHOTS:
"""


def plot_path(dirname: str, *args, filename=None) -> str:
    """
    Genera il path per il plot e crea le cartelle se non esistono

    """
    full_path = path.join(PLOT_PATH, dirname, *map(str, args))
    os.makedirs(full_path, exist_ok=True)
    if filename:
        full_path = os.path.join(full_path, filename)
    return full_path


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


def load_training() -> DataFrame:
    """
    Carica il dataset di training
    """
    with open(DATA_PATH, "r") as f:
        return pd.read_csv(f)


def filter_by_key(df: DataFrame, col: str, val, cols=None) -> DataFrame:
    """
    filtra il DataFrame in base al valore che assume una colonna.
    Equivalente di WHERE in SQL.
    shortcut per df.loc[df[col] == val]
    """
    if cols is None:
        return df.loc[df[col] == val].copy()
    return df.loc[df[col] == val, cols].copy()


def get_shift(df: DataFrame, col: str) -> DataFrame:
    """
    Ritorna i punti in cui il valore successivo è minore del precedente

    :param df: DataFrame
    :param col: Colonna da controllare
    """
    return df.loc[df[col] > df[col].shift(1), ["ESN", col]].copy()


def sensors_subset(df: DataFrame) -> DataFrame:
    """
    Ritorna il dataframe con solo i sensori
    shortcut per df.loc[:, SENSORS.tolist()]
    """
    return df.loc[:, SENSORS.tolist()].copy()


class SENSORS(Enum):
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

    @staticmethod
    def tolist():
        return list(iter_enum(SENSORS))

    @staticmethod
    def iter():
        return list(iter_enum(SENSORS))
