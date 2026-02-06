from datetime import datetime
from enum import Enum
from pandas import DataFrame, Series
from plotly.graph_objs import Data
from pykalman import KalmanFilter
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tools import plotting
from tools.config import DATA_TRAINING_DATA, PLOT_PATH, DATA_TESTING_PATH, DATA_VALIDATION_PATH
from tools.types.enums import ESENSORS, RepairEventType, Snapshots
from types import FunctionType
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
import pandas as pd
import random
import tools.config as c
import tools.features as f
from sklearn.svm import SVR
import pwlf
from sklearn.model_selection import ParameterGrid


###
### ENUMS
###

EVENTS = ["ww", "hpc", "hpt"]

ESN = range(101, 105)
"""
for esn in u.ESN:
"""
SNAPSHOTS = range(1,9)
SENSORS = ESENSORS.values()
FEATURES = ["mean", "std", "kurtosis", "skewness"]
META_COLS = [
    'cycle', 'esn', 'esn_index', 'fault_hpc_cycle', 'fault_hpt_cycle',
    'fault_ww_cycle', 'global_index', 'hpc_cycle', 'hpc_cycle_index', 'hpt_cycle', 'hpt_cycle_index',
    'snap', 'snap_index', 'to_next_hpc_cycle', 'to_next_hpt_cycle', 'to_next_ww_cycle', 'ww_cycle',
    'ww_cycle_index'
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
    with open(c.DATA_TRAINING_DATA, "r") as f:
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

def _carica(ind: int | list[int] | range, path, col, typef):
    # Gestione caso lista o range
    if isinstance(ind, (list, range)):
        esn_cycle_tracker = {}
        buffer = 10
        all_df = []
        
        for i in ind:
            pathf = f"{path}{typef}_{i}.csv"
            df = pd.read_csv(pathf)
            for esn in df['ESN'].unique():
                mask = df['ESN'] == esn
                
                # Se l'ESN è già stato visto, applica l'offset
                if esn in esn_cycle_tracker:
                    previous_max = esn_cycle_tracker[esn]
                    offset = previous_max + buffer
                    df.loc[mask, 'Cycles'] += offset
                
                # Aggiorna il tracker con il nuovo massimo
                current_max = df.loc[mask, col].max()
                esn_cycle_tracker[esn] = current_max
            
            all_df.append(df)
            
        # CORREZIONE: Concatenare la lista, non ritornare l'ultimo df!
        return pd.concat(all_df, ignore_index=True)

    # Gestione caso file singolo
    else:
        path = f"{path}{typef}_{ind}.csv"
        return pd.read_csv(path)

def load_testing(ind: int | list[int] | range = 0) -> pd.DataFrame:
    """
    Carica il dataset di training concatenando i file e aggiustando i cicli.
    """
    return _carica(ind, DATA_TESTING_PATH, col="Cycles", typef="test")


def load_validation(ind: int | list[int] | range = 0) -> pd.DataFrame:
    """
    Carica il dataset di training
    """
    return _carica(ind, DATA_VALIDATION_PATH, col="Cycles", typef="val")


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

def preprocess_pipeline(
        df: pd.DataFrame,
        sensor_cols=None,
        outlier_method='zscore',
        outlier_threshold: int | float = 3,
        smoothing_window: int=5,
        smoothing_step: int =2
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Esegue la pipeline di preprocessing con visualizzazione comparativa finale.
    """

    history = {}
    history['Original'] = df.copy()

    # 1. Missing Fill
    print("Step 1: Filling Missing Values...")
    dfo = missingfill(df, sensor_cols=sensor_cols)
    history['Missing Filled First'] = dfo.copy()

    # 2. Outlier Removal
    print(f"Step 2: Removing Outliers ({outlier_method})...")
    dfo = remove_outliers(dfo, sensor_cols=sensor_cols, method=outlier_method, threshold=outlier_threshold)
    history['Outliers Removed'] = dfo.copy()

    # 3. Missing Fill per gli outlier rimossi impostati a NaN
    print("Step 3: Filling Missing Values...")
    dfo = missingfill(dfo, sensor_cols=sensor_cols)
    history['Missing Filled Second'] = dfo.copy()
    
    # Identifica sensori
    target_sensors = sensor_cols if sensor_cols else [s.value if hasattr(s, 'value') else s for s in SENSORS]
    target_sensors = [s for s in target_sensors if s in dfo.columns]
    
    # 3. Kalman Filter
    # print("Step 3: Applying Kalman Filter...")
    # if 'ESN' in df_filled.columns and 'Snapshot' in df_filled.columns:
    #     for sensor in target_sensors:
    #         df_filled[sensor] = df_filled.groupby(['ESN', 'Snapshot'])[sensor].transform(apply_kalman)

    print("Step 3: Smoothing...")
    for sensor in target_sensors:
        dfo[sensor] = dfo.groupby(["ESN"])[sensor].transform(
            lambda x: x.rolling(window=smoothing_window, min_periods=smoothing_step).mean()
        ).reset_index(drop=True)

    history['Smoothing'] = dfo.copy()
    
    print("Pipeline completata.")
    return dfo, history

def train_evaluate_logo_pca(df, features, target_col, n_components=10, model_type='xgb'):
    """
    Esegue Leave-One-Group-Out (LOGO) su ESN integrando Scaler e PCA.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame contenente una colonna 'esn' per il raggruppamento
    features : list
        Lista dei nomi delle colonne delle feature
    target_col : str
        Nome della colonna target (RUL o simile)
    n_components : int, default=10
        Numero di componenti PCA
    model_type : str, default='xgb'
        Tipo di modello: 'xgb' (XGBoost), 'rf' (RandomForest), 'lr' (LinearRegression)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con colonne: 'esn', 'cycle', 'True_RUL', 'Pred_RUL'
    """
    
    esn_list = df['esn'].unique()
    results = []
    
    print(f"\n--- LOGO Cross-Validation su {len(esn_list)} motori ---")
    print(f"Target: {target_col} | Modello: {model_type} | PCA Comp: {n_components}")
    print(f"{'Test ESN':<12} {'RMSE':<10} {'R2':<10}")
    print("-" * 32)
    
    for test_esn in esn_list:
        # 1. Split Train/Test basato sull'ESN
        train_data = df[df['esn'] != test_esn]
        test_data = df[df['esn'] == test_esn]
        
        X_train_raw = train_data[features]
        y_train = train_data[target_col]
        X_test_raw = test_data[features]
        y_test = test_data[target_col]

        # 2. Imputazione (se ci sono NaNs)
        imputer = SimpleImputer(strategy='mean')
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw)
        
        # 3. Scaling (Fondamentale prima della PCA)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_test_sc = scaler.transform(X_test_imp)
        
        # 4. PCA: Calcolata SOLO sul Train
        pca = PCA(n_components=n_components)
        X_train_final = pca.fit_transform(X_train_sc)
        X_test_final = pca.transform(X_test_sc)
        
        # 5. Selezione Modello
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        elif model_type == 'xgb':
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42, verbosity=0)
        else:
            model = LinearRegression()
            
        # 6. Training e Predizione
        model.fit(X_train_final, y_train)
        preds = model.predict(X_test_final)
        
        # 7. Salvataggio risultati del fold
        fold_res = pd.DataFrame({
            'esn': test_data['esn'].values,
            'cycle': test_data['cycle'].values,
            'True_RUL': y_test.values,
            'Pred_RUL': preds
        })
        results.append(fold_res)
        
        # Metriche fold corrente
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"{test_esn:<12} {rmse:<10.2f} {r2:<10.4f}")

    # Concatenazione di tutti i risultati
    all_results = pd.concat(results, ignore_index=True)
    
    # Metriche Globali
    global_rmse = np.sqrt(mean_squared_error(all_results['True_RUL'], all_results['Pred_RUL']))
    global_r2 = r2_score(all_results['True_RUL'], all_results['Pred_RUL'])
    print(f"\n{'='*32}")
    print(f"--- METRICHE GLOBALI ---")
    print(f"Global RMSE: {global_rmse:.2f}")
    print(f"Global R2:   {global_r2:.4f}")
    print(f"{'='*32}\n")
    
    return all_results




def calculate_hpt_health_index(df, t3_col, t45_col, to_next_hpc_col):
    df['HI_HPT'] = np.nan

    for esn in df['esn'].unique():
        engine_data = df[df['esn'] == esn].sort_values('snap_index')

        # Individuo l'evento HPC
        zero_indices = engine_data.index[engine_data[to_next_hpc_col] == 0].tolist()

        if zero_indices:
            idx_zero = zero_indices[-2]
            pos_zero = engine_data.index.get_indexer([idx_zero])[0]

            # Prendo una finestra a cavallo dell'evento di manutenzione dell'hpc
            start = max(0, pos_zero - 10)
            end = min(len(engine_data), pos_zero + 11)
            calib_window = engine_data.iloc[start:end]

            # Funzione obiettivo: minimizzare la deviazione standard dell'HI
            # mentre l'altro componente (HPC) viene sostituito
            def objective(a):
                hi = -a * calib_window[t3_col] - calib_window[t45_col]
                return np.std(hi)

            res = minimize(objective, x0=1.0)
            best_alpha = res.x[0]

        else:
            # Fallback se non c'è l'evento: uso tutto il motore
            best_alpha = 1.0 # o una media globale

        # Applico l'alpha trovato a tutto il motore
        df.loc[df['esn'] == esn, 'HI_HPT'] = -best_alpha * engine_data[t3_col] - engine_data[t45_col]

    return df


def calculate_hpt_health_index_all(df, t3_col, t45_col, to_next_hpc_col):
    df['HI_HPT'] = np.nan
    
    for esn in ESN:
        engine_mask = df['esn'] == esn
        engine_data = df[engine_mask].sort_values('snap_index')
        # Centratura dei dati (Mean Removal) per ogni motore
        # engine_data[t3_col] = engine_data[t3_col] - engine_data[t3_col].mean()
        # engine_data[t45_col] = engine_data[t45_col] - engine_data[t45_col].mean()   
        
        # Pulizia dati: rimuoviamo eventuali NaN per la calibrazione
        calib_data = engine_data.dropna(subset=[t3_col, t45_col, to_next_hpc_col])
        
        if not calib_data.empty:
            # FUNZIONE OBIETTIVO:
            # Vogliamo che HI_HPT non "segua" il trend dell'HPC.
            # Minimizziamo la deviazione standard dell'HI su tutta la vita.
            def objective(a):
                hi = -a * calib_data[t3_col] - calib_data[t45_col]
                return np.std(hi)
            
            res = minimize(objective, x0=1.0)
            best_alpha = res.x[0]
        else:
            best_alpha = 1.0
            
        # Applichiamo a tutto il motore
        df.loc[engine_mask, 'HI_HPT'] = -best_alpha * engine_data[t3_col] - engine_data[t45_col]
        
    return df


def fit_hpt_mapping(df, to_next_hpt_col):
    """
    Usa l'evento HPT per definire la retta di predizione 
    assumendo cicli = 0 all'evento HPT.
    """
    engine_params_hpt = {}
    
    for esn in ESN:
        engine_data = df[df['esn'] == esn].dropna(subset=['HI_HPT', to_next_hpt_col])
        
        if not engine_data.empty:
            # Regressione lineare: HI_HPT -> to_next_hpt_cycle
            X = engine_data[['HI_HPT']].values
            y = engine_data[to_next_hpt_col].values
            
            model = LinearRegression().fit(X, y)
            
            # Salviamo i parametri specifici per questo motore
            engine_params_hpt[esn] = {
                'slope': model.coef_[0],
                'intercept': model.intercept_
            }
            
            # Creiamo la colonna della predizione lineare
            df.loc[df['esn'] == esn, 'hpt_linear_pred'] = model.predict(X)
            
    return df, engine_params_hpt




def fit_hpt_mapping_svr(df, hi_col, target_col, C=10, epsilon=0.2, gamma='scale'):
    """
    Sostituisce il fit lineare con un Support Vector Regressor.
    C: Regolarizzazione (più alto = segue meglio i dati, ma rischio overfitting)
    epsilon: Larghezza del tubo dove l'errore non viene penalizzato
    """
    engine_models = {}
    df[f'{hi_col}_pred_svr'] = np.nan

    for esn in df['esn'].unique():
        # Isoliamo i dati del motore e puliamo i NaN
        mask = (df['esn'] == esn) & df[hi_col].notna() & df[target_col].notna()
        engine_data = df[mask]
        
        if len(engine_data) > 10:
            X = engine_data[[hi_col]].values
            y = engine_data[target_col].values
            
            # SVR beneficia enormemente dallo scaling (fondamentale!)
            scaler_x = StandardScaler()
            X_scaled = scaler_x.fit_transform(X)
            
            # Definizione e fit del modello SVR
            model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
            model.fit(X_scaled, y)
            
            # Salvataggio del modello e dello scaler per l'inferenza futura
            engine_models[esn] = {'model': model, 'scaler': scaler_x}
            
            # Predizione sui dati attuali
            df.loc[mask, f'{hi_col}_pred_svr'] = model.predict(X_scaled)
            
    return df, engine_models



def fit_piecewise_auto(df, hi_col, target_col):
    x = df[hi_col].values
    y = df[target_col].values
    
    # Inizializza il modello
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    
    # Trova i breakpoint ottimali (es. cerchiamo 2 segmenti, quindi 1 breakpoint)
    breakpoints = my_pwlf.fit(2)
    
    # Predici i valori
    df['hpt_pred'] = my_pwlf.predict(x)
    return df, breakpoints



def evaluate_svr_params(df, hi_col, target_col, params):
    """Calcola il TWE medio per una specifica combinazione di parametri SVR"""
    # Usiamo la tua funzione esistente per fittare i modelli
    df_temp, _ = fit_hpt_mapping_svr(
        df.copy(), hi_col, target_col, 
        C=params['C'], 
        epsilon=params['epsilon'], 
        gamma=params['gamma']
    )
    
    # Rimuoviamo i NaN nati da motori con troppi pochi dati
    mask = df_temp[f'{hi_col}_pred_svr'].notna()
    y_true = df_temp.loc[mask, target_col]
    y_pred = df_temp.loc[mask, f'{hi_col}_pred_svr']
    
    # Calcolo della metrica del paper (TWE)
    # Assicurati di aver definito calculate_twe_score come visto prima
    return calculate_twe_score(y_true, y_pred, alpha=0.001)


def standardize_residuals(df, cols=['T3_res', 'T45_res']):
    df_std = df.copy()
    for col in cols:
        # Standardizzazione: (valore - media) / deviazione_standard
        mean = df_std[col].mean()
        std = df_std[col].std()
        df_std[col] = (df_std[col] - mean) / std
    return df_std


def calculate_twe_score(y_true, y_pred, alpha=0.001, beta=1.0):
    """
    Implementazione del Time-Weighted Error (TWE)
    y_true: Cicli reali rimanenti (to_next_target_cycle)
    y_pred: Cicli predetti dal tuo modello lineare
    alpha: parametro di decadimento (es. 0.001)
    beta: fattore di normalizzazione specifico per il target
    """
    error = y_pred - y_true
    
    # Calcolo del peso w(yi, y_hat_i) - Equazione (2)
    # Penalizza di più le over-predictions (ritardi nella manutenzione)
    weights = np.where(
        error >= 0, 
        2 / (1 + alpha * y_true), # Predizione in ritardo (più grave)
        1 / (1 + alpha * y_true)  # Predizione in anticipo (meno grave)
    )
    
    # TWE - Equazione (1)
    twe = weights * (error**2) * beta
    
    # Score finale (Media) - Equazione (3)
    return np.mean(twe)


def perform_grid_search(df, hi_col, target_col):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2, 0.5],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    best_score = float('inf')
    best_params = None
    
    print("Inizio Grid Search (ottimizzazione basata su TWE)...")
    
    for params in ParameterGrid(param_grid):
        current_score = evaluate_svr_params(df, hi_col, target_col, params)
        print(f"Params: {params} -> TWE Score: {current_score:.4f}")
        
        if current_score < best_score:
            best_score = current_score
            best_params = params
            
    print("-" * 30)
    print(f"MIGLIORI PARAMETRI: {best_params}")
    print(f"MIGLIOR TWE SCORE: {best_score:.4f}")
    
    return best_params