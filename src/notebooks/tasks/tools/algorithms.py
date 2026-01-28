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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from . import plotting as up
from . import utils as u


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

        window = signal[i: i + N]
        res[group_id].append(rms_signal(window))

        i += o

    return res


def moving_rms(signal, N, o=None):
    rmss = []
    i = 0
    if o is None:
        o = N
    while i + N < len(signal):
        rmss.append(rms_signal(signal[i: i + N]))
        i += o
    return rmss


def shape_factor(signal):
    mav = np.mean(np.abs(signal))
    return rms_signal(signal) / mav


def skewness(signal, bias=False):
    return spstats.skew(signal, bias)


def kurtosis(signal, fisher=True, bias=False):
    return spstats.kurtosis(signal, fisher, bias)


def moving_features_with_stop(
    signal: list[tuple[int, float]], stop: list[int], N, step=None
):
    if step is None:
        step = N

    def new_group():
        return {
            "rms": [],
            "mean": [],
            "std": [],
            "kurtosis": [],
            "skewness": [],
            "shape_factor": [],
        }

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
        window_list = [a[1] for a in signal[i: i + N]]

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

        res[group_id]["kurtosis"].append(
            float(spstats.kurtosis(window, axis=0, fisher=True))
        )
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


def train_evaluate_logo_pca(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_components: int = 10,
    model_type: str = "xgb",
) -> pd.DataFrame:
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

    esn_list = df["ESN"].unique()
    results = []

    print(f"\n--- LOGO Cross-Validation su {len(esn_list)} motori ---")
    print(f"Target: {target_col} | Modello: {model_type} | PCA Comp: {n_components}")
    print(f"{'Test ESN':<12} {'RMSE':<10} {'R2':<10}")
    print("-" * 32)

    for test_esn in esn_list:
        # 1. Split Train/Test basato sull'ESN
        train_data = df[df["ESN"] != test_esn]
        test_data = df[df["ESN"] == test_esn]

        X_train_raw = train_data[features]
        y_train = train_data[target_col]
        X_test_raw = test_data[features]
        y_test = test_data[target_col]

        # 2. Imputazione (se ci sono NaNs)
        imputer = SimpleImputer(strategy="mean")
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
        if model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=100, n_jobs=-1, random_state=42)
        elif model_type == "xgb":
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            )
        else:
            model = LinearRegression()

        # 6. Training e Predizione
        model.fit(X_train_final, y_train)
        preds = model.predict(X_test_final)

        # 7. Salvataggio risultati del fold
        fold_res = pd.DataFrame(
            {
                "ESN": test_data["ESN"].values,
                "cycle": test_data["cycle"].values,
                "True_RUL": y_test.values,
                "Pred_RUL": preds,
            }
        )
        results.append(fold_res)

        # Metriche fold corrente
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"{test_esn:<12} {rmse:<10.2f} {r2:<10.4f}")

    # Concatenazione di tutti i risultati
    all_results = pd.concat(results, ignore_index=True)

    # Metriche Globali
    global_rmse = np.sqrt(
        mean_squared_error(all_results["True_RUL"], all_results["Pred_RUL"])
    )
    global_r2 = r2_score(all_results["True_RUL"], all_results["Pred_RUL"])
    print(f"\n{'=' * 32}")
    print(f"--- METRICHE GLOBALI ---")
    print(f"Global RMSE: {global_rmse:.2f}")
    print(f"Global R2:   {global_r2:.4f}")
    print(f"{'=' * 32}\n")

    return all_results


def train_linear_regression(
    dff: pd.DataFrame, tot: pd.DataFrame, window_size: int = 5000, target: str = None
):
    """
    Trains Linear Regression using ONLY the features from feature engineering.
    dff: DataFrame with engineered features
    tot: DataFrame with feature metadata (which features are the best)
    target: Target event (HPC, HPT, WW) - if None, auto-detect
    """
    # Extract target feature names from the feature evaluation metadata
    best_features = tot["feature"].tolist() if "feature" in tot.columns else []
    
    if not best_features:
        raise ValueError("No features found in feature metadata. Ensure feature engineering was run.")
    
    # Identify the target column
    target_col = None
    target_map = {
        'HPC': 'Cycles_to_HPC_SV',
        'HPT': 'Cycles_to_HPT_SV',
        'WW': 'Cycles_to_WW'
    }
    
    # Se target è specificato, usa quello
    if target and target in target_map:
        if target_map[target] in dff.columns:
            target_col = target_map[target]
    
    # Altrimenti, auto-detect (should be the last column or named as target)
    possible_targets = ['Cycles_to_HPC_SV', 'Cycles_to_HPT_SV', 'Cycles_to_WW', 'RUL']
    for col in possible_targets:
        if col in dff.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"No target column found in data. Expected one of: {possible_targets}")
    
    # Use ONLY the best engineered features
    features = [f for f in best_features if f in dff.columns]
    
    if not features:
        raise ValueError(f"No engineered features found in DataFrame. Features: {best_features}, Columns: {dff.columns.tolist()}")
    
    print(f"Using {len(features)} engineered features for training")
    print(f"Features: {features}")

    # --- STEP 2: Preparazione Dati ---
    train_df = dff[dff["ESN"] != 104].dropna(subset=features + [target_col])
    test_df = dff[dff["ESN"] == 104].dropna(subset=features + [target_col])

    # Fallback: se test_df è vuoto, usiamo l'ultimo 20% dei dati
    if len(test_df) == 0:
        print("Warning: ESN 104 not available or all NaN. Using last 20% of training data as test set.")
        split_idx = int(len(train_df) * 0.8)
        test_df = train_df.iloc[split_idx:].copy()
        train_df = train_df.iloc[:split_idx].copy()

    if len(test_df) == 0:
        raise ValueError("Test set is empty even after fallback. Ensure feature data is available.")

    # --- STEP 3: Standardizzazione ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    # --- STEP 4: Modello e Predizione ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = np.maximum(0, model.predict(X_test))

    # --- STEP 5: Visualizzazione ---
    up.plot_rul_prediction(y_test, y_pred, window_size=len(X_test),
                           is_reset=None)

    return model, y_pred


def train_transformer(dff: pd.DataFrame, tot: pd.DataFrame, target: str = None):
    """
    Trains Transformer using ONLY the features from feature engineering.
    dff: DataFrame with engineered features
    tot: DataFrame with feature metadata (which features are the best)
    target: Target event (HPC, HPT, WW) - if None, auto-detect
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    # Extract target feature names from the feature evaluation metadata
    best_features = tot["feature"].tolist() if "feature" in tot.columns else []
    
    if not best_features:
        raise ValueError("No features found in feature metadata. Ensure feature engineering was run.")
    
    # Identify the target column
    target_col = None
    target_map = {
        'HPC': 'Cycles_to_HPC_SV',
        'HPT': 'Cycles_to_HPT_SV',
        'WW': 'Cycles_to_WW'
    }
    
    # Se target è specificato, usa quello
    if target and target in target_map:
        if target_map[target] in dff.columns:
            target_col = target_map[target]
    
    # Altrimenti, auto-detect
    if target_col is None:
        possible_targets = ['Cycles_to_HPC_SV', 'Cycles_to_HPT_SV', 'Cycles_to_WW']
        for col in possible_targets:
            if col in dff.columns:
                target_col = col
                break
    
    if target_col is None:
        raise ValueError(f"No target column found in data. Expected one of: {possible_targets}")
    
    # Use ONLY the best engineered features that exist in the dataframe
    features_list = [f for f in best_features if f in dff.columns]
    
    if not features_list:
        raise ValueError(f"No engineered features found in DataFrame. Features: {best_features}, Columns: {dff.columns.tolist()}")
    
    print(f"Using {len(features_list)} engineered features for Transformer training")
    print(f"Features: {features_list}")
    print(f"Target: {target_col}")

    # --- 1. CONFIGURAZIONE ---
    WINDOW_SIZE = 30
    BATCH_SIZE = 64
    LR = 0.0005
    EPOCHS = 400
    MAX_RUL = 12500

    # --- 2. PREPARAZIONE DATI (Memory Efficient) ---
    # Creiamo il df di lavoro con le colonne corrette
    cols_to_use = list(set(["ESN", target_col] + features_list))
    df_work = dff[cols_to_use].copy()

    # Logica Chunk
    df_work["is_reset"] = (df_work.groupby(
        "ESN")[target_col].diff() > 0).astype(int)
    df_work["chunk_id"] = df_work.groupby("ESN")["is_reset"].cumsum()
    df_work["relative_cycle"] = df_work.groupby(["ESN", "chunk_id"]).cumcount()

    # Features finali per il transformer
    features_final = features_list + ["relative_cycle"]

    # Split train/test
    train_mask = df_work["ESN"] != 104
    test_mask = df_work["ESN"] == 104

    # Scaling e PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        df_work.loc[train_mask, features_final])
    X_test_scaled = scaler.transform(df_work.loc[test_mask, features_final])

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
    X_test_pca = pca.transform(X_test_scaled).astype(np.float32)

    y_train_all = np.minimum(
        df_work.loc[train_mask, target_col].values, MAX_RUL
    ).astype(np.float32)
    y_test_all = df_work.loc[test_mask, target_col].values.astype(np.float32)

    # --- 3. CREAZIONE SEQUENZE (Sliding Window per ESN) ---
    def create_sequences_by_esn(data, targets, esn_values, window_size):
        sequences, labels = [], []
        unique_esns = np.unique(esn_values)

        for esn in unique_esns:
            mask = esn_values == esn
            esn_data = data[mask]
            esn_targets = targets[mask]

            if len(esn_data) < window_size:
                continue

            for i in range(window_size, len(esn_data)):
                sequences.append(esn_data[i - window_size: i])
                labels.append(esn_targets[i])

        return np.array(sequences), np.array(labels)

    # Creiamo le sequenze separando correttamente i motori
    X_train_seq, y_train_seq = create_sequences_by_esn(
        X_train_pca, y_train_all, df_work.loc[train_mask,
                                              "ESN"].values, WINDOW_SIZE
    )
    X_test_seq, y_test_seq = create_sequences_by_esn(
        X_test_pca, y_test_all, df_work.loc[test_mask,
                                            "ESN"].values, WINDOW_SIZE
    )

    class RULDataset(Dataset):
        def __init__(self, x, y):
            self.x = torch.tensor(x)
            self.y = torch.tensor(y).unsqueeze(1)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.x[i], self.y[i]

    train_loader = DataLoader(
        RULDataset(X_train_seq, y_train_seq), batch_size=BATCH_SIZE, shuffle=True
    )

    # --- 4. ARCHITETTURA TRANSFORMER ---
    class RULTransformer(nn.Module):
        def __init__(self, input_dim, model_dim=64, nhead=4, num_layers=2):
            super().__init__()
            self.embedding = nn.Linear(input_dim, model_dim)
            self.pos_encoding = nn.Parameter(
                torch.zeros(1, WINDOW_SIZE, model_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=nhead, batch_first=True, dim_feedforward=128
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            self.head = nn.Sequential(
                nn.Linear(model_dim, 32), nn.ReLU(), nn.Linear(32, 1)
            )

        def forward(self, x):
            x = self.embedding(x) + self.pos_encoding
            x = self.transformer(x)
            return self.head(x[:, -1, :])

    # --- 5. TRAINING ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RULTransformer(input_dim=X_train_pca.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"Training su {device}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            loss = criterion(model(batch_x), batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # --- 6. PREDIZIONE E PLOT ---
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test_seq).to(device)).cpu().numpy()

    up.plot_rul_prediction_transformer(y_test_seq, y_pred)

    print(f"MAE Transformer: {mean_absolute_error(y_test_seq, y_pred):.2f}")

    return model, y_pred


def train_xgboost(dff: pd.DataFrame, tot: pd.DataFrame, target: str = None):
    """
    Trains XGBoost using ONLY the features from feature engineering.
    dff: DataFrame with engineered features
    tot: DataFrame with feature metadata (which features are the best)
    target: Target event (HPC, HPT, WW) - if None, auto-detect
    """
    # Extract target feature names from the feature evaluation metadata
    best_features = tot["feature"].tolist() if "feature" in tot.columns else []
    
    if not best_features:
        raise ValueError("No features found in feature metadata. Ensure feature engineering was run.")
    
    # Identify the target column
    target_col = None
    target_map = {
        'HPC': 'Cycles_to_HPC_SV',
        'HPT': 'Cycles_to_HPT_SV',
        'WW': 'Cycles_to_WW'
    }
    
    # Se target è specificato, usa quello
    if target and target in target_map:
        if target_map[target] in dff.columns:
            target_col = target_map[target]
    
    # Altrimenti, auto-detect
    if target_col is None:
        possible_targets = ['Cycles_to_HPC_SV', 'Cycles_to_HPT_SV', 'Cycles_to_WW']
        for col in possible_targets:
            if col in dff.columns:
                target_col = col
                break
    
    if target_col is None:
        raise ValueError(f"No target column found in data. Expected one of: {possible_targets}")
    
    # Use ONLY the best engineered features that exist in the dataframe
    features_list = [f for f in best_features if f in dff.columns]
    
    if not features_list:
        raise ValueError(f"No engineered features found in DataFrame. Features: {best_features}, Columns: {dff.columns.tolist()}")
    
    print(f"Using {len(features_list)} engineered features for XGBoost training")
    print(f"Features: {features_list}")
    print(f"Target: {target_col}")
    
    # --- 1. OTTIMIZZAZIONE MEMORIA E SELEZIONE ---
    cols_to_use = list(set(["ESN", target_col] + features_list))

    # Lavoriamo su una copia ridotta con precisione float32
    df_work = dff[cols_to_use].copy()
    for col in features_list:
        df_work[col] = df_work[col].astype(np.float32)

    # --- 2. LOGICA DI CHUNK (Reset Manutenzione) ---
    # Identifichiamo i reset e calcoliamo il ciclo relativo (fondamentale per la logica temporale)
    df_work["is_reset"] = (df_work.groupby(
        "ESN")[target_col].diff() > 0).astype(np.int8)
    df_work["relative_cycle"] = (
        df_work.groupby(["ESN", df_work["is_reset"].cumsum()])
        .cumcount()
        .astype(np.int32)
    )

    # Feature finali: Solo quelle fornite + la posizione nel chunk (permette di "agganciare" la RUL)
    features_final = features_list + ["relative_cycle"]

    # --- 3. PREPARAZIONE DATASET ---
    MAX_RUL = 14000  # Target Piecewise: risolve la "gara" iniziale dei grafici 2-5
    train_mask = df_work["ESN"] != 104
    test_mask = df_work["ESN"] == 104

    # Conversione in array NumPy per massimizzare la velocità
    X_train_raw = df_work.loc[train_mask, features_final].values
    y_train = np.minimum(df_work.loc[train_mask, target_col].values, MAX_RUL)

    X_test_raw = df_work.loc[test_mask, features_final].values
    y_test_real = df_work.loc[test_mask, target_col].values

    # --- 4. PIPELINE: SCALING + PCA (Fix Errore) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # n_components=0.95 richiede svd_solver='auto' o 'full'
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
    X_test_pca = pca.transform(X_test_scaled).astype(np.float32)

    print(
        f"PCA: {len(features_final)} feature ridotte a {pca.n_components_} componenti."
    )

    # --- 5. ADDESTRAMENTO OTTIMIZZATO (XGBoost Hist) ---
    # tree_method='hist' è essenziale per la velocità computazionale
    xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        # Aggiungiamo regolarizzazione per evitare il "jittering" delle immagini 6 e 7
        reg_lambda=2.0,
        reg_alpha=0.5,
    )

    param_dist = {
        "n_estimators": [600, 1000],
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 5, 7],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    # RandomizedSearch per velocità
    random_search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    random_search.fit(X_train_pca, y_train)
    best_model = random_search.best_estimator_

    # --- 6. PREDIZIONE E POST-PROCESSING ---
    y_pred = best_model.predict(X_test_pca)
    # Smoothing finale per pulire la curva (Window di 20 cicli)
    y_pred_smooth = (
        pd.Series(y_pred).rolling(
            window=20, min_periods=1, center=True).mean().values
    )
    y_pred_smooth = np.maximum(0, y_pred_smooth)

    # --- 7. VISUALIZZAZIONE ---
    up.plot_rul_prediction_xgb(y_test_real, y_pred_smooth)

    print(f"MAE Finale: {mean_absolute_error(y_test_real, y_pred_smooth):.2f}")

    return best_model, y_pred_smooth


def train_random_forest(dff: pd.DataFrame, tot: pd.DataFrame, target: str = None):
    """
    Trains Random Forest using ONLY the features from feature engineering.
    dff: DataFrame with engineered features
    tot: DataFrame with feature metadata (which features are the best)
    target: Target event (HPC, HPT, WW) - if None, auto-detect
    """
    # Extract target feature names from the feature evaluation metadata
    best_features = tot["feature"].tolist() if "feature" in tot.columns else []
    
    if not best_features:
        raise ValueError("No features found in feature metadata. Ensure feature engineering was run.")
    
    # Identify the target column
    target_col = None
    target_map = {
        'HPC': 'Cycles_to_HPC_SV',
        'HPT': 'Cycles_to_HPT_SV',
        'WW': 'Cycles_to_WW'
    }
    
    # Se target è specificato, usa quello
    if target and target in target_map:
        if target_map[target] in dff.columns:
            target_col = target_map[target]
    
    # Altrimenti, auto-detect
    if target_col is None:
        possible_targets = ['Cycles_to_HPC_SV', 'Cycles_to_HPT_SV', 'Cycles_to_WW']
        for col in possible_targets:
            if col in dff.columns:
                target_col = col
                break
    
    if target_col is None:
        raise ValueError(f"No target column found in data. Expected one of: {target_map.values()}")
    
    # Use ONLY the best engineered features that exist in the dataframe
    features = [f for f in best_features if f in dff.columns]
    
    if not features:
        raise ValueError(f"No engineered features found in DataFrame. Features: {best_features}, Columns: {dff.columns.tolist()}")
    
    print(f"Using {len(features)} engineered features for Random Forest training")
    print(f"Features: {features}")
    print(f"Target: {target_col}")
    
    # --- STEP 1: Definizione Feature e Target (usando le feature ingegnerizzate) ---

    # --- STEP 2: Split Train e Test ---
    train_df = dff[dff["ESN"] != 104].dropna(subset=[target_col] + features)
    test_df = dff[dff["ESN"] == 104].dropna(subset=[target_col] + features)
    
    # Fallback per ESN 104 se non presente
    if len(test_df) == 0:
        print("Warning: ESN 104 not found in test set, using last 20% of training data")
        split_idx = int(len(train_df) * 0.8)
        test_df = train_df.iloc[split_idx:].copy()
        train_df = train_df.iloc[:split_idx].copy()

    X_train_raw = train_df[features]
    y_train = train_df[target_col]
    X_test_raw = test_df[features]
    y_test = test_df[target_col]

    # --- STEP 3: Standardizzazione ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # --- STEP 4: Random Forest Regressor ---
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # --- STEP 5: Predizione ---
    y_pred = rf_model.predict(X_test)
    y_pred = np.maximum(0, y_pred)

    # --- STEP 6: Visualizzazione ---
    try:
        up.plot_rul_prediction_rf(y_test, y_pred)
    except Exception as e:
        print(f"Warning: Could not plot RF predictions: {e}")

    # --- STEP 7: Feature Importance ---
    importances = pd.Series(
        rf_model.feature_importances_, index=features
    ).sort_values(ascending=False)
    print("\nImportanza delle Feature:")
    print(importances)

    return rf_model, y_pred, importances
