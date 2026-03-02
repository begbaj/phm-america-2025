import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from tools import utils as u
from tools import config as cfg

def apply_kalman(series: pd.Series, transition_matrices=[1], observation_matrices=[1]) -> np.ndarray:
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

def missingfill(df: pd.DataFrame, align_cols=['Snapshot', 'Cycles_Since_New'], align_alt=['Snapshot', 'Cycles'], sensor_cols=None) -> pd.DataFrame:
    """
    Riempie i valori mancanti (NaN) integrando i dati presenti negli altri motori.
    
    Strategia:
    1. Calcola la media della flotta (tutti i motori disponibili) per lo stesso (Snapshot, Ciclo).
    2. Riempie i NaN con questa media.
    3. Per i valori ancora mancanti (es. nessun dato nella flotta per quel punto), esegue interpolazione lineare per ESN.
    
    :param df: DataFrame contenente i dati.
    :param align_cols: Colonne usate per allineare i cicli tra motori.
    :param align_alt: Colonne alternative usate per allineare i cicli tra motori se align_cols fallisce.
    :param sensor_cols: Lista di sensori da processare. Se None, usa i SENSORS globali.
    :return: DataFrame con i missing values riempiti.
    """
    # 1. Determinazione colonne sensori
    if sensor_cols is None:
        # Usa i sensori globali definiti in questo modulo
        raw_sensors = list(u.SENSORS)
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
    elif all(col in df_out.columns for col in align_alt):
        fleet_means = df_out.groupby(align_alt)[valid_cols].transform('mean')
        # Sostituzione dei NaN con la media calcolata
        df_out[valid_cols] = df_out[valid_cols].fillna(fleet_means)
    else:
        print(f"Warning: Colonne di allineamento {align_cols} o {align_alt} non trovate. Salto step flotta.")
        
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

def minmax(df, column):
    col_min = df[column].min()
    col_max = df[column].max()
    return (df[column] - col_min) / (col_max - col_min)

def minmax_all(df):
    newdf = pd.DataFrame()
    for column in df.columns:
        col_min = df[column].min()
        col_max = df[column].max()
        newdf[column] = (df[column] - col_min) / (col_max - col_min)
    return newdf

def normalize(col):
  col_min, col_max = col.min(), col.max()
  col = (col - col_min) / (col_max - col_min)
  col = col.to_frame()
  return col

def median_norm(df):
    for i in range(df.shape[1]):
        m = df.iloc[:,i].median()
        df.iloc[:,i] -= m
    return df

def calculate_rolling_slope(df, column, window, groupby='ESN'):
    from tools.algorithms import get_slope
    return df.groupby(groupby)[column].transform(
        lambda x: x.rolling(window=window).apply(get_slope)
    )

def calculate_cycle_increment(df, value_col, reset_col, groupby='ESN'):
    """
    Calculates the increment of a value since the last reset (where reset_col == 0).
    """
    df_temp = df.copy()
    # Identify reset points
    df_temp['temp_reset'] = df_temp.loc[df_temp[reset_col] == 0, value_col]
    # Forward fill the reset values within each group
    df_temp['temp_reset'] = df_temp.groupby(groupby)['temp_reset'].ffill()
    # Calculate increment
    return df_temp[value_col] - df_temp['temp_reset']

def remove_outliers(df: pd.DataFrame, sensor_cols=None, threshold=3, method='zscore') -> pd.DataFrame:
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
        target_sensors = [s.value if hasattr(s, 'value') else s for s in u.SENSORS]
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
            iso = IsolationForest(contamination='auto')
            preds = iso.fit_predict(data)
            # preds == -1 sono gli outliers
            outlier_indices = series_nonan.index[preds == -1]
            df_out.loc[outlier_indices, sensor] = np.nan
                
    return df_out

def common_pipeline(df,
                    outlier_sensors=None,
                    outlier_threshold=0.8,
                    outlier_method="isoforest",
                    missing_align_cols=["Snapshot","Cycles_Since_New"],
                    missing_sensors=None
) -> pd.DataFrame:
    df = remove_outliers(df, sensor_cols=outlier_sensors, threshold=outlier_threshold, method=outlier_method)
    df = missingfill(df, align_cols=missing_align_cols, sensor_cols=missing_sensors).dropna()
    return df


# def preprocess_pipeline(
#         df: pd.DataFrame,
#         sensor_cols=None,
#         outlier_method='zscore',
#         outlier_threshold: int | float = 3,
#         smoothing_window: int=5,
#         smoothing_step: int =2
#     ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
#     """
#     Esegue la pipeline di preprocessing con visualizzazione comparativa finale.
#     """

#     history = {}
#     history['Original'] = df.copy()

#     # 1. Missing Fill
#     print("Step 1: Filling Missing Values...")
#     dfo = missingfill(df, sensor_cols=sensor_cols)
#     history['Missing Filled First'] = dfo.copy()

#     # 2. Outlier Removal
#     print(f"Step 2: Removing Outliers ({outlier_method})...")
#     dfo = remove_outliers(dfo, sensor_cols=sensor_cols, method=outlier_method, threshold=outlier_threshold)
#     history['Outliers Removed'] = dfo.copy()

#     # 3. Missing Fill per gli outlier rimossi impostati a NaN
#     print("Step 3: Filling Missing Values...")
#     dfo = missingfill(dfo, sensor_cols=sensor_cols)
#     history['Missing Filled Second'] = dfo.copy()
    
#     # Identifica sensori
#     target_sensors = sensor_cols if sensor_cols else [s.value if hasattr(s, 'value') else s for s in u.SENSORS]
#     target_sensors = [s for s in target_sensors if s in dfo.columns]
    
#     # 3. Kalman Filter
#     # print("Step 3: Applying Kalman Filter...")
#     # if 'ESN' in df_filled.columns and 'Snapshot' in df_filled.columns:
#     #     for sensor in target_sensors:
#     #         df_filled[sensor] = df_filled.groupby(['ESN', 'Snapshot'])[sensor].transform(apply_kalman)

#     print("Step 3: Smoothing...")
#     for sensor in target_sensors:
#         dfo[sensor] = dfo.groupby(["ESN"])[sensor].transform(
#             lambda x: x.rolling(window=smoothing_window, min_periods=smoothing_step).mean()
#         ).reset_index(drop=True)

#     history['Smoothing'] = dfo.copy()
    
#     print("Pipeline completata.")
#     return dfo, history

# def preprocess_data(train: pd.DataFrame):
#     # 1. PREPARAZIONE INDICI
#     # Reset dell'indice per preservare l'indice originale come 'global_index'
#     dfp = train.reset_index().rename(columns={"index": "global_index"})

#     ## PREPROCESSING
#     dfp , history = preprocess_pipeline(dfp,
#                                     outlier_method='isoforest',
#                                     outlier_threshold=0.08,
#                                     smoothing_window=100,
#                                     smoothing_step=25,
#                                     )

#     ### PREPROCESSING 
#     # Rinominazione di alcune colonne per semplicità di scrittura
#     rename_map = {
#         'ESN': 'esn',
#         'Snapshot': 'snap',
#         'Cumulative_WWs': 'ww_cycle',
#         'Cumulative_HPC_SVs': 'hpc_cycle',
#         'Cumulative_HPT_SVs': 'hpt_cycle',
#         'Cycles_to_WW': 'to_next_ww_cycle',
#         'Cycles_to_HPC_SV': 'to_next_hpc_cycle',
#         'Cycles_to_HPT_SV': 'to_next_hpt_cycle',
#         'Cycles_Since_New': 'cycle'
#     }
#     dfp = dfp.rename(columns=rename_map)

#     # Rimozione Sensed_ dai sensori (clogged view)
#     sensor_cols = [c for c in dfp.columns if c.startswith('Sensed_')]
#     sensor_rename_map = {c: c.replace('Sensed_', '') for c in sensor_cols}
#     dfp = dfp.rename(columns=sensor_rename_map)
#     final_sensor_names = list(sensor_rename_map.values())

#     # Nuovi indici
#     dfp['esn_index'] = dfp.groupby('esn').cumcount()
#     dfp['snap_index'] = dfp.groupby(["esn", 'snap']).cumcount()
#     dfp['ww_cycle_index']  = dfp.groupby(['ww_cycle', "snap", "esn"]).cumcount()
#     dfp['hpc_cycle_index'] = dfp.groupby(['hpc_cycle', "snap", "esn"]).cumcount()
#     dfp['hpt_cycle_index'] = dfp.groupby(['hpt_cycle', "snap", "esn"]).cumcount()

#     # Aggiunta della colonna "faulty" per ogni tipo di evento
#     fault_map = {
#         'to_next_ww_cycle': 'fault_ww_cycle',
#         'to_next_hpc_cycle': 'fault_hpc_cycle',
#         'to_next_hpt_cycle': 'fault_hpt_cycle'
#     }

#     for source_col, fault_name in fault_map.items():
#         dfp[fault_name] = 0
#         dfp.loc[dfp[source_col] == 0, fault_name] = 1
#         dfp.loc[dfp.groupby('esn').cumcount(ascending=False) == 0, fault_name] = 1
#     new_fault_columns = ['fault_ww_cycle', 'fault_hpc_cycle', 'fault_hpt_cycle']

#     # 4. DEFINIZIONE ORDINE COLONNE
#     # Definiamo l'ordine esatto in cui vogliamo che appaiano nel CSV Wide
#     # Prima gli identificatori, poi gli indici di manutenzione, infine i sensori
#     cols_order = [
#         'snap_index',
#         'esn',
#         'cycle',
#         'snap',
#         'esn_index',
#         'global_index',
#         'ww_cycle_index',
#         'hpc_cycle_index',
#         'hpt_cycle_index',
#         'ww_cycle',
#         'hpc_cycle',
#         'hpt_cycle',
#         'to_next_ww_cycle',
#         'to_next_hpc_cycle',
#         'to_next_hpt_cycle'
#     ] + final_sensor_names + new_fault_columns

#     dfp = dfp[cols_order]

#     # for snap_id, group_data in dfp.groupby('snap'):
#     #     print(f"Scrittura file per SNAP {snap_id}...")
#     #     filename = f"snapshot_{snap_id}.csv"
#     #     path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename=filename)
#     #     # index=False perché 'global_index' è già una colonna esplicita
#     #     group_data.to_csv(path, index=False)

#     # u.SENSORS = final_sensor_names
#     del new_fault_columns, sensor_cols, sensor_rename_map
#     return dfp, history, final_sensor_names