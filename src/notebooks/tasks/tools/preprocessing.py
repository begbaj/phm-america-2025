import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from . import utils as u
from . import config as cfg
from .types.enums import SENSORS


def apply_kalman(
    series: pd.Series, transition_matrices=[1], observation_matrices=[1]
) -> np.ndarray:
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
    kf = KalmanFilter(
        transition_matrices=transition_matrices,
        observation_matrices=observation_matrices,
    )

    # Se la serie è tutta NaN, ritorna la serie originale
    if series.isnull().all():
        return series

    # Maschera i valori non validi (NaN)
    masked_data = np.ma.masked_invalid(series.values)

    # Applica il filtro
    (filtered_means, _) = kf.filter(masked_data)

    return filtered_means.flatten()


def missingfill(
    df: pd.DataFrame, align_cols=["Snapshot", "Cycles_Since_New"], sensor_cols=None
) -> pd.DataFrame:
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
        raw_sensors = list(u.SENSORS)
    else:
        raw_sensors = list(sensor_cols)

    # Risoluzione nomi sensori (se sono Enum)
    valid_cols = []
    for s in raw_sensors:
        s_name = s.value if hasattr(s, "value") else str(s)
        # Support both raw sensor names and 'Sensed_' prefixed ones
        candidates = [s_name, f"Sensed_{s_name}"]
        for cand in candidates:
            if cand in df.columns and cand not in valid_cols:
                valid_cols.append(cand)

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
            fleet_means = df_out.groupby(
                align_cols)[valid_cols].transform("mean")

            # Sostituzione dei NaN con la media calcolata
            df_out[valid_cols] = df_out[valid_cols].fillna(fleet_means)
        except Exception as e:
            print(
                f"Warning: Errore durante il calcolo della media flotta: {e}")
    else:
        print(
            f"Warning: Colonne di allineamento {align_cols} non trovate. Salto step flotta."
        )

    # 3. Interpolazione Residua (Per ESN)
    # Se la media flotta non ha coperto tutto (es. cicli dove nessuno ha dati), eseguiamo forward fill
    if "ESN" in df_out.columns:
        # Applica interpolazione per ogni sensore, raggruppando per ESN
        for col in valid_cols:
            # Interpolazione residua: Forward Fill come richiesto
            df_out[col] = df_out.groupby(
                "ESN")[col].transform(lambda x: x.ffill())

            # Fallback: Backward Fill per coprire eventuali NaN all'inizio della serie
            df_out[col] = df_out.groupby(
                "ESN")[col].transform(lambda x: x.bfill())

    return df_out


def remove_outliers(
    df: pd.DataFrame, sensor_cols=None, threshold=3, method="zscore"
) -> pd.DataFrame:
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
        target_sensors = [s.value if hasattr(
            s, "value") else s for s in u.SENSORS]
    else:
        target_sensors = [s.value if hasattr(
            s, "value") else s for s in sensor_cols]

    target_sensors = [s for s in target_sensors if s in df_out.columns]

    if method == "zscore":
        for sensor in target_sensors:
            # Calcolo z-score ignorando i NaN
            series = df_out[sensor]
            if series.dropna().empty:
                continue
            z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
            df_out.loc[z_scores > threshold, sensor] = np.nan

    elif method == "iqr":
        for sensor in target_sensors:
            Q1 = df_out[sensor].quantile(0.25)
            Q3 = df_out[sensor].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_out.loc[
                (df_out[sensor] < lower_bound) | (
                    df_out[sensor] > upper_bound), sensor
            ] = np.nan

    elif method == "isoforest":
        for sensor in target_sensors:
            series_nonan = df_out[sensor].dropna()
            if series_nonan.empty:
                continue
            data = series_nonan.values.reshape(-1, 1)
            # Contamination 'auto' o basata sulla soglia se interpretata come percentuale
            iso = IsolationForest(contamination="auto", random_state=42)
            preds = iso.fit_predict(data)
            # preds == -1 sono gli outliers
            outlier_indices = series_nonan.index[preds == -1]
            df_out.loc[outlier_indices, sensor] = np.nan

    return df_out


def preprocess_pipeline(
    df: pd.DataFrame,
    sensor_cols=None,
    outlier_method="zscore",
    outlier_threshold: int | float = 3,
    smoothing_window: int = 5,
    smoothing_step: int = 2,
    smoothing_method: str = "rolling_mean",
    do_missing_fill: bool = True,
    do_outlier_removal: bool = True,
    do_smoothing: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Esegue la pipeline di preprocessing con opzioni selezionabili per smoothing.
    
    Args:
        smoothing_method: 'rolling_mean', 'exponential', 'savitzky_golay'
    """

    history = {}
    history["Original"] = df.copy()
    dfo = df.copy()

    # 1. Missing Fill
    if do_missing_fill:
        print("Step 1: Filling Missing Values...")
        dfo = missingfill(dfo, sensor_cols=sensor_cols)
        history["Missing Filled First"] = dfo.copy()
    else:
        print("Step 1: Skipping Missing Values Fill.")

    # 2. Outlier Removal
    if do_outlier_removal:
        print(f"Step 2: Removing Outliers ({outlier_method})...")
        dfo = remove_outliers(
            dfo, sensor_cols=sensor_cols, method=outlier_method, threshold=outlier_threshold
        )
        history["Outliers Removed"] = dfo.copy()

        # 3. Missing Fill per gli outlier rimossi impostati a NaN
        if do_missing_fill:
            print("Step 3: Filling Missing Values (Post-Outlier)...")
            dfo = missingfill(dfo, sensor_cols=sensor_cols)
            history["Missing Filled Second"] = dfo.copy()
    else:
        print("Step 2: Skipping Outlier Removal.")

    # Identifica sensori
    potential_sensors = (
        sensor_cols
        if sensor_cols
        else [s.value if hasattr(s, "value") else s for s in u.SENSORS]
    )
    
    target_sensors = []
    for s in potential_sensors:
        if s in dfo.columns:
            target_sensors.append(s)
        elif f"Sensed_{s}" in dfo.columns:
            target_sensors.append(f"Sensed_{s}")

    # 4. Smoothing con metodo scelto
    if do_smoothing:
        print(f"Step 4: Smoothing (method={smoothing_method}, window={smoothing_window})...")
        
        if smoothing_method == "rolling_mean":
            # Rolling mean (media mobile semplice)
            for sensor in target_sensors:
                dfo[sensor] = (
                    dfo.groupby(["ESN"])[sensor]
                    .transform(
                        lambda x: x.rolling(
                            window=smoothing_window, min_periods=smoothing_step
                        ).mean()
                    )
                    .reset_index(drop=True)
                )
        elif smoothing_method == "exponential":
            # Exponential smoothing (EMA)
            span = smoothing_window
            for sensor in target_sensors:
                dfo[sensor] = (
                    dfo.groupby(["ESN"])[sensor]
                    .transform(lambda x: x.ewm(span=span, adjust=False).mean())
                    .reset_index(drop=True)
                )
        elif smoothing_method == "savitzky_golay":
            # Savitzky-Golay filter (richiede scipy.signal)
            from scipy.signal import savgol_filter
            window_length = min(smoothing_window, 51)  # SG requires odd window
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(3, window_length - 1)  # polynomial order < window length
            
            for sensor in target_sensors:
                def apply_sg(x):
                    if len(x) < window_length:
                        return x
                    try:
                        return pd.Series(savgol_filter(x, window_length, polyorder, mode='interp'), index=x.index)
                    except:
                        return x
                
                dfo[sensor] = dfo.groupby(["ESN"])[sensor].transform(apply_sg).reset_index(drop=True)
        else:
            print(f"Warning: smoothing_method '{smoothing_method}' not recognized. Using rolling_mean as fallback.")
            for sensor in target_sensors:
                dfo[sensor] = (
                    dfo.groupby(["ESN"])[sensor]
                    .transform(
                        lambda x: x.rolling(
                            window=smoothing_window, min_periods=smoothing_step
                        ).mean()
                    )
                    .reset_index(drop=True)
                )

        history["Smoothing"] = dfo.copy()
    else:
        print("Step 4: Skipping Smoothing.")

    print("Pipeline completata.")
    return dfo, history


def preprocess_data(
    train: pd.DataFrame, 
    outlier_method: str = "isoforest", 
    outlier_threshold: float = 0.08, 
    smoothing_window: int = 100, 
    smoothing_step: int = 25, 
    smoothing_method: str = "rolling_mean",
    do_missing_fill: bool = True,
    do_outlier_removal: bool = True,
    do_smoothing: bool = True
):
    # 1. PREPARAZIONE INDICI
    # Reset dell'indice per preservare l'indice originale come 'global_index'
    dfp = train.reset_index().rename(columns={"index": "global_index"})

    # PREPROCESSING
    dfp, history = preprocess_pipeline(
        dfp,
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
        smoothing_window=smoothing_window,
        smoothing_step=smoothing_step,
        smoothing_method=smoothing_method,
        do_missing_fill=do_missing_fill,
        do_outlier_removal=do_outlier_removal,
        do_smoothing=do_smoothing,
    )


    # PREPROCESSING
    # Rimozione Sensed_ dai sensori (clogged view)
    sensor_cols = [c for c in dfp.columns if c.startswith("Sensed_")]
    sensor_rename_map = {c: c.replace("Sensed_", "") for c in sensor_cols}
    dfp = dfp.rename(columns=sensor_rename_map)
    final_sensor_names = list(sensor_rename_map.values())

    # Nuovi indici
    dfp["esn_index"] = dfp.groupby("ESN").cumcount()
    dfp["snap_index"] = dfp.groupby(["ESN", "Snapshot"]).cumcount()
    dfp["ww_cycle_index"] = dfp.groupby(["Cumulative_WWs", "Snapshot", "ESN"]).cumcount()
    dfp["hpc_cycle_index"] = dfp.groupby(
        ["Cumulative_HPC_SVs", "Snapshot", "ESN"]).cumcount()
    dfp["hpt_cycle_index"] = dfp.groupby(
        ["Cumulative_HPT_SVs", "Snapshot", "ESN"]).cumcount()

    # Aggiunta della colonna "faulty" per ogni tipo di evento
    fault_map = {
        "Cycles_to_WW": "fault_ww_cycle",
        "Cycles_to_HPC_SV": "fault_hpc_cycle",
        "Cycles_to_HPT_SV": "fault_hpt_cycle",
    }

    for source_col, fault_name in fault_map.items():
        dfp[fault_name] = 0
        dfp.loc[dfp[source_col] == 0, fault_name] = 1
        dfp.loc[dfp.groupby("ESN").cumcount(
            ascending=False) == 0, fault_name] = 1
    new_fault_columns = ["fault_ww_cycle",
                         "fault_hpc_cycle", "fault_hpt_cycle"]

    # 4. DEFINIZIONE ORDINE COLONNE
    # Definiamo l'ordine esatto in cui vogliamo che appaiano nel CSV Wide
    # Prima gli identificatori, poi gli indici di manutenzione, infine i sensori
    cols_order = (
        [
            "snap_index",
            "ESN",
            "Cycles_Since_New",
            "Snapshot",
            "esn_index",
            "global_index",
            "ww_cycle_index",
            "hpc_cycle_index",
            "hpt_cycle_index",
            "Cumulative_WWs",
            "Cumulative_HPC_SVs",
            "Cumulative_HPT_SVs",
            "Cycles_to_WW",
            "Cycles_to_HPC_SV",
            "Cycles_to_HPT_SV",
        ]
        + final_sensor_names
        + new_fault_columns
    )

    dfp = dfp[cols_order]

    # for snap_id, group_data in dfp.groupby('snap'):
    #     print(f"Scrittura file per SNAP {snap_id}...")
    #     filename = f"snapshot_{snap_id}.csv"
    #     path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename=filename)
    #     # index=False perché 'global_index' è già una colonna esplicita
    #     group_data.to_csv(path, index=False)

    # u.SENSORS = final_sensor_names
    del new_fault_columns, sensor_cols, sensor_rename_map
    return dfp, history, final_sensor_names


def aggregate_snapshots(df: pd.DataFrame, sensors: list[str], method='mean') -> pd.DataFrame:
    """
    Aggrega i dati degli snapshot calcolando la media dei sensori.
    """
    meta_cols = [
        "Cycles_Since_New",
        "snap_index",
        "Cumulative_WWs",
        "Cumulative_HPC_SVs",
        "Cumulative_HPT_SVs",
        "ww_cycle_index",
        "hpc_cycle_index",
        "hpt_cycle_index",
        "Cycles_to_WW",
        "Cycles_to_HPC_SV",
        "Cycles_to_HPT_SV",
        "fault_ww_cycle",
        "fault_hpc_cycle",
        "fault_hpt_cycle",
    ]

    agg_logic = {}

    # SUI SENSORI: facciamo la MEDIA (qui passiamo da 8 righe a 1 riga)
    for col in sensors:
        if col in df.columns:
            agg_logic[col] = method

    # SULLE COLONNE META: prendiamo il PRIMO valore (perché è uguale in tutti gli snapshot di quel momento)
    for col in meta_cols:
        if col in df.columns:
            agg_logic[col] = "first"

    # Scegliamo solo ESN e il contatore di riga come chiavi.
    group_keys = ["ESN", "Cycles_Since_New"]

    # Rieseguiamo l'aggregazione usando queste nuove chiavi
    df_averaged = df.groupby(group_keys, as_index=False).agg(agg_logic)

    df_averaged = df_averaged.rename(columns={"snap_index": "esn_index"})
    df_averaged = df_averaged.sort_values(["ESN", "esn_index"]).dropna()

    return df_averaged
