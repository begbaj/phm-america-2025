import os.path as path
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pandas as pd
import numpy as np
from types import FunctionType
from pandas import DataFrame, Series
from enum import Enum
from plotly.graph_objs import Data
from tools import plotting
from tools.config import DATA_PATH, PLOT_PATH
import random
import tools.config as c
from tools.types.enums import ESENSORS, RepairEventType, Snapshots
import tools.features as f
from scipy import stats
from pykalman import KalmanFilter
from sklearn.ensemble import IsolationForest

###
### ENUMS
###

EVENTS = ["wws", "hpc", "hpt"]

ESN = range(101, 105)
"""
for esn in u.ESN:
"""
SNAPSHOTS = range(1,9)
SENSORS = ESENSORS.values()
FEATURES = ["mean", "std", "kurtosis", "skewness"]
META_COLS = [
    'ww_cycle', 'hpc_cycle', 'hpt_cycle',
    'ww_cycle_index', 'hpc_cycle_index', 'hpt_cycle_index',
    'to_next_ww_cycle', 'to_next_hpc_cycle', 'to_next_hpt_cycle',
    'fault_ww_cycle', 'fault_hpc_cycle', 'fault_hpt_cycle'
]


def get_timestamp():
    """
    Restituisce il timestamp attuale nel formato stringa YYMMHHmm.
    Esempio: 12 gennaio 2026 alle 10:45 -> '26011045'
    """
    return datetime.now().strftime("%y%m%H%M")

def preproc_features(dfi, group, incols, features, sortcol, window_size=None,step=1, outcols=None):
    f._feature_aggregator(dfi, group, incols, features, sortcol, window_size=window_size, step=step, outcols=outcols)

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

def ess_iter(df: DataFrame, order=["esn", "sensor", "snapshot"], plotdata=False, rand = False):
    order = [o.lower() for o in order]
    order_enum = {
        "esn": ESN,
        "sensor": ESENSORS.values(),
        "snapshot": Snapshots.values(),
        "event": RepairEventType.values(),
    }

    if rand:
        order_enum = {k: random.sample(list(v), len(v)) for k, v in order_enum.items()}

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

def to_signal(df, column) -> list[tuple[int,float]]:
    ixs = df.index
    vals = df[column].values

    if len(ixs) != len(vals):
        print("Length of ixs and vals are not the same")
        return [(0,0)]

    return list(zip(ixs, vals)) 

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
            data[sensor] = data.groupby(['ESN', 'Snapshot'], group_keys=False)[sensor].transform(
                lambda x: x.ewm(span=span, adjust=False).mean())
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
### DATAFRAME MANIPULATION AND LOGICS
###

def df_reset_index(df: DataFrame, base=0):
    df.index -= base

def df_ess_filter(df: DataFrame, esn: int, sensor: str, snapshot: int, all=False, pdata=False):
    if all:
        if pdata:
            pdd = plotting.PlotData()
            pdd.esn = esn
            pdd.sensor = sensor
            pdd.snap = snapshot
            return df.loc[
                (df["ESN"] == esn) & (df["Snapshot"] == snapshot),
                [sensor]
            ], pdd
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

def refactor_table(df: DataFrame, snap: int, span: int) -> DataFrame:
    """
    Funzione per il refactoring del dataset e per la moving average
    
    :param df: Dataset per il refactoring
    :type df: DataFrame
    :param snap: Numero di snapshot
    :type snap: int
    :param span: Ampiezza della finestra per la moving average
    :type span: int
    :return: Dataset modificato
    :rtype: DataFrame
    """
    subset = df[df['Snapshot'] == snap]
    final_table = subset.pivot(index='Cycles_Since_New', columns='ESN', values=SENSORS)
    final_table.columns = [f"{sensor}_{esn}" for sensor, esn in final_table.columns]
    final_table = final_table.reset_index()
    for column in final_table.columns:
        final_table[column] = final_table[column].transform(lambda x: x.ewm(span=span, adjust=False).mean())
    return final_table

def apply_kalman(series: Series, transition_matrices=[1], observation_matrices=[1]) -> np.ndarray:
    """
    Applica il filtro di Kalman a una serie di dati, gestendo i valori mancanti (NaN).
    
    :param series: La serie di dati da filtrare.
    :param transition_matrices: Matrice di transizione per il filtro di Kalman. Default [1].
    :param observation_matrices: Matrice di osservazione per il filtro di Kalman. Default [1].
    :return: Array con i dati filtrati appiattiti.
    """
    try:
        from pykalman import KalmanFilter
    except ImportError:
        raise ImportError("pykalman is required for apply_kalman")

    # Inizializza il filtro di Kalman
    kf = KalmanFilter(transition_matrices=transition_matrices, observation_matrices=observation_matrices)
    
    # Se la serie è tutta NaN, ritorna la serie originale
    if series.isnull().all():
        return series
    
    # Maschera i valori non validi (NaN)
    masked_data = np.ma.masked_invalid(series.values)
    
    # Applica il filtro
    (filtered_means, _) = kf.filter(masked_data)
    
    return filtered_means.flatten()

def missingfill(df: DataFrame, align_cols=['Snapshot', 'Cycles_Since_New'], sensor_cols=None) -> DataFrame:
    """
    Riempie i valori mancanti (NaN) integrando i dati presenti negli altri motori.
    
    Strategia:
    1. Calcola la media della flotta (tutti i motori disponibili) per lo stesso (Snapshot, Ciclo).
    2. Riempie i NaN con questa media.
    3. Per i valori ancora mancanti (es. nessun dato nella flotta per quel punto), esegue interpolazione lineare per ESN.
    
    :param df: DataFrame contenente i dati.
    :param align_cols: Colonne usate per allineare i cicli tra motori.
    :param sensor_cols: Lista di sensori da processare. Se None, usa i SENSORS globali.
    :return: DataFrame con i missing values riempiti.
    """
    # 1. Determinazione colonne sensori
    if sensor_cols is None:
        # Usa i sensori globali definiti in questo modulo
        raw_sensors = list(SENSORS)
    else:
        raw_sensors = list(sensor_cols)
        
    # Risoluzione nomi sensori (se sono Enum)
    valid_cols = []
    for s in raw_sensors:
        s_name = s.value if hasattr(s, 'value') else str(s)
        if s_name in df.columns:
            valid_cols.append(s_name)
    
    if not valid_cols:
        print("Nessuna colonna sensore valida trovata per missingfill.")
        return df

    df_out = df.copy()
    
    print(f"Esecuzione missingfill su {len(valid_cols)} sensori...")
    
    # 2. Riempimento tramite Media Flotta (Fleet Mean)
    # Verifica che le colonne di allineamento esistano
    if all(col in df_out.columns for col in align_cols):
        try:
            # Calcola media raggruppata per align_cols sui sensori
            # transform('mean') restituisce un DF/Series allineato all'originale con le medie dei gruppi
            fleet_means = df_out.groupby(align_cols)[valid_cols].transform('mean')
            
            # Sostituzione dei NaN con la media calcolata
            df_out[valid_cols] = df_out[valid_cols].fillna(fleet_means)
        except Exception as e:
            print(f"Warning: Errore durante il calcolo della media flotta: {e}")
    else:
        print(f"Warning: Colonne di allineamento {align_cols} non trovate. Salto step flotta.")
        
    # 3. Interpolazione Residua (Per ESN)
    # Se la media flotta non ha coperto tutto (es. cicli dove nessuno ha dati), eseguiamo forward fill
    if 'ESN' in df_out.columns:
        # Applica interpolazione per ogni sensore, raggruppando per ESN
        for col in valid_cols:
            # Interpolazione residua: Forward Fill come richiesto
            df_out[col] = df_out.groupby('ESN')[col].transform(lambda x: x.ffill())
            
            # Fallback: Backward Fill per coprire eventuali NaN all'inizio della serie
            df_out[col] = df_out.groupby('ESN')[col].transform(lambda x: x.bfill())

    return df_out

def remove_outliers(df: DataFrame, sensor_cols=None, threshold=3, method='zscore') -> DataFrame:
    """
    Identifica e rimuove gli outliers dai sensori, impostandoli a NaN.
    Supporta metodi basati su Z-score, IQR e Isolation Forest.
    
    :param df: DataFrame di input.
    :param sensor_cols: Lista di sensori.
    :param threshold: Soglia per lo z-score (default 3) o moltiplicatore per IQR (default 1.5/3).
    :param method: 'zscore', 'iqr', o 'isoforest'.
    :return: DataFrame con outliers sostituiti da NaN.
    """
    df_out = df.copy()
    
    if sensor_cols is None:
        target_sensors = [s.value if hasattr(s, 'value') else s for s in SENSORS]
    else:
        target_sensors = [s.value if hasattr(s, 'value') else s for s in sensor_cols]
    
    target_sensors = [s for s in target_sensors if s in df_out.columns]

    if method == 'zscore':
        for sensor in target_sensors:
            # Calcolo z-score ignorando i NaN
            series = df_out[sensor]
            if series.dropna().empty: continue
            z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
            df_out.loc[z_scores > threshold, sensor] = np.nan
            
    elif method == 'iqr':
        for sensor in target_sensors:
            Q1 = df_out[sensor].quantile(0.25)
            Q3 = df_out[sensor].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_out.loc[(df_out[sensor] < lower_bound) | (df_out[sensor] > upper_bound), sensor] = np.nan
            
    elif method == 'isoforest':
        for sensor in target_sensors:
            series_nonan = df_out[sensor].dropna()
            if series_nonan.empty: continue
            data = series_nonan.values.reshape(-1, 1)
            # Contamination 'auto' o basata sulla soglia se interpretata come percentuale
            iso = IsolationForest(contamination='auto', random_state=42)
            preds = iso.fit_predict(data)
            # preds == -1 sono gli outliers
            outlier_indices = series_nonan.index[preds == -1]
            df_out.loc[outlier_indices, sensor] = np.nan
                
    return df_out

def preprocess_pipeline(df: pd.DataFrame, sensor_cols=None, outlier_method='zscore', outlier_threshold: int | float = 3, smoothing_window=5, smoothing_step=2) -> pd.DataFrame:
    """
    Esegue la pipeline di preprocessing con visualizzazione comparativa finale.
    """
    # Salviamo i dati originali per il confronto finale
    history = {}
    history['Original'] = df.copy()

    # 1. Missing Fill
    print("Step 1: Filling Missing Values...")
    df_filled = missingfill(df, sensor_cols=sensor_cols)
    # history['Missing Filled'] = df_filled.copy()

    # 2. Outlier Removal
    print(f"Step 2: Removing Outliers ({outlier_method})...")
    df_cleaned = remove_outliers(df_filled, sensor_cols=sensor_cols, method=outlier_method, threshold=outlier_threshold)
    history['Outliers Removed'] = df_cleaned.copy()

    # 3. Missing Fill per gli outlier rimossi impostati a NaN
    print("Step 3: Filling Missing Values...")
    df_refilled = missingfill(df_cleaned, sensor_cols=sensor_cols)
    history['Missing Filled'] = df_refilled.copy()
    
    # Identifica sensori per Kalman
    target_sensors = sensor_cols if sensor_cols else [s.value if hasattr(s, 'value') else s for s in SENSORS]
    target_sensors = [s for s in target_sensors if s in df_refilled.columns]
    
    # 3. Kalman Filter
    # print("Step 3: Applying Kalman Filter...")
    # if 'ESN' in df_filled.columns and 'Snapshot' in df_filled.columns:
    #     for sensor in target_sensors:
    #         df_filled[sensor] = df_filled.groupby(['ESN', 'Snapshot'])[sensor].transform(apply_kalman)
    print("Step 3: Smoothing...")
    for sensor in target_sensors:
        df_refilled[sensor] = df_refilled.groupby(["ESN"])[sensor].transform(
            lambda x: x.rolling(window=smoothing_window, min_periods=smoothing_step).mean()
        ).reset_index(drop=True)
    
    history['Smoothing'] = df_refilled.copy()
    
    print("Pipeline completata.")
    return df_refilled, history
