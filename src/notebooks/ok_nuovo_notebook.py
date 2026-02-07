# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: phm-america-2025 (3.11.9)
#     language: python
#     name: python3
# ---

# %%
from numpy import sign
from pyparsing import line
from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sympy import O, deg
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import scipy.optimize as optimize
import scipy.stats as stats
import glob
import os
from sklearn.preprocessing import StandardScaler
from enum import Enum
from datetime import datetime
from pandas import DataFrame, Series
from plotly.graph_objs import Data
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from types import FunctionType
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os.path as path
import random
from sklearn.svm import SVR
import pwlf
from sklearn.model_selection import ParameterGrid
from xgboost import train


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# from tools import utils as u, config as cfg, plotting as up, preprocessing as pp
# %load_ext autoreload
# %autoreload 2

from tools import utils as u, config as cfg, plotting as up, preprocessing as pp


# %%
# CONFIG

DATA_TRAINING_DATA = f"training_data.csv"
"""training_data.csv"""

DATA_VALIDATION_DATA = f"test_data.csv"
"""test_data.csv"""

DATA_TEST_DATA = f"validation_data.csv"
"""validation_data.csv"""


# %%
# UTLIS

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

SENSORS = ESENSORS.values()

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
    with open(DATA_TRAINING_DATA, "r") as f:
        return WrapData(pd.read_csv(f))

def load_validation() -> pd.DataFrame:
    """
    Carica il dataset di validation
    """
    with open(DATA_TEST_DATA, "r") as f:
        return pd.read_csv(f)

def load_test() -> pd.DataFrame:
    """
    Carica il dataset di test
    """
    with open(DATA_VALIDATION_DATA, "r") as f:
        return pd.read_csv(f)

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
            # Contamination 'auto' o basata sulla soglia se interpretata come percentuale
            iso = IsolationForest(contamination='auto', random_state=42)
            preds = iso.fit_predict(data)
            # preds == -1 sono gli outliers
            outlier_indices = series_nonan.index[preds == -1]
            df_out.loc[outlier_indices, sensor] = np.nan
    return df_out

def missingfill(df: pd.DataFrame, align_cols=['Snapshot', 'Cycles_Since_New'], align_alt=['Snapshot', 'Cycles'], sensor_cols=None) -> pd.DataFrame:
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
        fleet_means = df_out.groupby(align_alt)[valid_cols].transform('mean')
        # Sostituzione dei NaN con la media calcolata
        df_out[valid_cols] = df_out[valid_cols].fillna(fleet_means)
        # print(f"Warning: Colonne di allineamento {align_cols} non trovate. Salto step flotta.")
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



# %%
# model_i = 0
testing_esn = 102
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
# degradation_vars = [s for s in u.SENSORS if s not in operating_vars]
degradation_vars = [s for s in SENSORS if s not in operating_vars and s != "Sensed_P25" and s != "Sensed_T5"]
managed_cols = set(degradation_vars) | set(operating_vars)

# %%
# DOWNSAMPLING TRAINING PER AVERE PER TUTTI LO STESSO NUMERO DI DATI
df = u.load_training()()
df = pp.remove_outliers(df, SENSORS)
df = pp.missingfill(df).dropna()
# Aggregazione dataset di training
other_cols_df = [col for col in df.columns if col not in managed_cols]
agg_logic = {col: 'median' for col in degradation_vars}
agg_logic.update({col: 'median' for col in operating_vars})
agg_logic.update({col: 'first' for col in other_cols_df})
df = df.groupby(['ESN', 'Cycles_Since_New']).agg(agg_logic).reset_index(drop=True)
rows = df.groupby('ESN').size().reset_index(name='rows').copy()
print(rows)

# DOWNSAMPLING VALIDATION PER AVERE PER TUTTI LO STESSO NUMERO DI DATI
dfv = u.load_validation(range(0,48))
dfv = pp.remove_outliers(dfv, SENSORS)
dfv = pp.missingfill(dfv, align_cols=["Snapshot", "Cycles"]).dropna()
# Aggregazione dataset di validation
other_cols_dfv = [col for col in dfv.columns if col not in managed_cols]
agg_logic_v = {col: 'median' for col in degradation_vars}
agg_logic_v.update({col: 'median' for col in operating_vars})
agg_logic_v.update({col: 'first' for col in other_cols_dfv})
dfv = dfv.groupby(['ESN', 'Cycles']).agg(agg_logic_v).reset_index(drop=True)
rows_val = dfv.groupby('ESN').size().reset_index(name='numero_righe').copy()
print(rows_val)

# DOWNSAMPLING TESTING PER AVERE PER TUTTI LO STESSO NUMERO DI DATI
dft = u.load_testing(range(0,52))
dft = pp.remove_outliers(dft, SENSORS)
dft = pp.missingfill(dft, align_cols=["Snapshot", "Cycles"]).dropna()
# Aggregazione dataset di training
other_cols_dft = [col for col in dft.columns if col not in managed_cols]
agg_logic_t = {col: 'median' for col in degradation_vars}
agg_logic_t.update({col: 'median' for col in operating_vars})
agg_logic_t.update({col: 'first' for col in other_cols_dft})
dft = dft.groupby(['ESN', 'Cycles']).agg(agg_logic_t).reset_index(drop=True)
rows_test = dft.groupby('ESN').size().reset_index(name='numero_righe').copy()
print(rows_test)


# %%
# FUNCTIONS

def train_models(df, operating_vars, degradation_vars) -> dict[int, dict[str,LinearRegression]]:
    X_train = df[operating_vars]
    Y_train = df[degradation_vars]
    models = {}
    for i in range(0,8):
        X_temp = pd.DataFrame(np.roll(X_train, i, axis=1))
        models[i] = {}
        models[i]["model"] = train_model(X_temp, Y_train)
    return models

def train_model(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def s_pred(s_o, model):
    return model.predict(s_o)

def residual(s_d, s_o, model):
    return s_d - s_pred(s_o, model)

def wind(y_p, y, a):
    diff = y - y_p
    num = np.where(diff >= 0, 2.0, 1.0)
    if isinstance(y_p, pd.DataFrame) or isinstance(y_p, pd.Series):
        y_p = y_p.values
    return num / (1 + a * y_p)

def TWE(y_p, y, a, b):
    if isinstance(y_p, pd.DataFrame): y_p = y_p.values
    weight = wind(y_p, y, a)
    squared_error = (y - y_p) ** 2
    return weight * squared_error * b

def HI(T3_res, T45_res, alpha):
    return -alpha * T3_res - T45_res

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

def median_norm(df):
    for i in range(0,7):
        m = df.iloc[:,i].median()
        df.iloc[:,i] -= m
    return df

def objective(alpha, T3, T45, RUL):
    hi = -alpha*T3 - T45
    RUL = RUL.dropna()
    hi = hi.dropna()
    corr = stats.pearsonr(RUL,hi)
    # return np.sqrt(np.mean((hi - RUL)**2)) + 1
    return - corr[0]

def objective_beta(params, T3, T45, RUL):
    alpha, beta = params
    hi = -alpha*T3 - beta*T45
    RUL = RUL.dropna()
    hi = hi.dropna()
    corr = stats.pearsonr(RUL,hi)
    # return np.sqrt(np.mean((hi - RUL)**2)) + 1
    return - corr[0]

def HIE(params, vars):
    #return np.sum([-params[i]*vars.iloc[:,i] for i in range(0, 8)])
    return vars.dot(-np.array(params))


def objective_experimental(params, vars, RUL):
    hi = HIE(params, vars)
    # RUL = RUL.dropna()
    corr = stats.pearsonr(RUL,hi)
    return -corr[0]

def objective_deviation(params, vars, RUL):
    hi = HIE(params, vars)
    hi_min, hi_max = hi.min(), hi.max()
    if hi_max == hi_min:
        # Penalità se l'HI è una linea piatta
        return 1.0
    hi_norm = (hi - hi_min) / (hi_max - hi_min)
    mse = np.mean((hi_norm - RUL)**2)
    return mse

def get_rolling_slope_intercept(series, window):
    slopes = []
    intercepts = []
    series = np.asarray(series).flatten()
    for i in range(len(series)):
        if i < window:
            slopes.append(0)
            intercepts.append(0)
        else:
            y = series[i-window:i]
            x = np.arange(window)
            # Fit polinomiale di grado 1 (retta) -> ritorna [slope, intercept]
            poly = np.polyfit(x, y, 1)
            slopes.append(float(poly[0]))
            intercepts.append(float(poly[1]))
    return np.array(slopes), np.array(intercepts)


def normalize(col):
  col_min, col_max = col.min(), col.max()
  col = (col - col_min) / (col_max - col_min)
  col = col.to_frame()
  return col


def get_slope(y):
    """Calcola la pendenza della retta di regressione per una finestra y"""
    x = np.arange(len(y))
    # Polyfit di grado 1 restituisce [pendenza, intercetta]
    slope = np.polyfit(x, y, 1)[0]
    return slope


# %%
# Regressore lineare per il calcolo dei residui

cycles_healthy = 60

val_data = df[df["ESN"] == testing_esn].reset_index().copy()
X_val = val_data[operating_vars]
Y_val = val_data[degradation_vars]

train_data_full = df[df["ESN"] != testing_esn].copy()
train_data_healthy = train_data_full.groupby("ESN").head(cycles_healthy).reset_index(drop=True)
X_train = train_data_healthy[operating_vars]
Y_train = train_data_healthy[degradation_vars]

print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)

# training regressore lineare
model = train_model(X_train, Y_train)
# %store model

# predict dei valori
res_list = []
for esn in train_data_full["ESN"].unique():
  mask = train_data_full["ESN"] == esn
  X_train = train_data_full.loc[mask, operating_vars]
  Y_train = train_data_full.loc[mask, degradation_vars]
  Y_pred = model.predict(X_train)
  res_temp = Y_train - Y_pred
  res_temp["ESN"] = esn
  res_list.append(res_temp)

res_train = pd.concat(res_list)

# residui
res_val_list = []
Y_pred = model.predict(X_val)
res_val_temp = Y_val - Y_pred
res_val_list.append(res_val_temp)
res_val = pd.concat(res_val_list)

# integrazione residui sul dataset originale
cleaned_chunks = []
for esn in train_data_full["ESN"].unique():
  temp = res_train[res_train["ESN"] == esn].copy()
  temp = remove_outliers(temp, SENSORS, threshold=0.8)
  temp[degradation_vars] = temp[degradation_vars].rolling(window=50, min_periods=1).mean()
  temp = temp.dropna()
  cleaned_chunks.append(temp)

res_train = pd.concat(cleaned_chunks).dropna()
train_data_full.update(res_train)
# res_train.index = res_train.groupby("ESN").cumcount()


# integrazione residui sul dataset originale
res_val = remove_outliers(res_val, SENSORS, threshold=0.8)
res_val = res_val.rolling(window=50, min_periods=1).mean()
res_val = res_val.dropna()
val_data.update(res_val)


# %store res_train
# %store res_val

valid_indices_train = res_train.index
hpt_rul_train = train_data_full.loc[valid_indices_train, ["ESN", "Cycles_to_HPT_SV"]]
hpc_rul_train = train_data_full.loc[valid_indices_train, ["ESN", "Cycles_to_HPC_SV"]]
ww_rul_train = train_data_full.loc[valid_indices_train, ["ESN", "Cycles_to_WW"]]
T3_res_train = train_data_full.loc[valid_indices_train, ["ESN", "Sensed_T3"]]
T45_res_train = train_data_full.loc[valid_indices_train, ["ESN", "Sensed_T45"]]

valid_indices_val = res_val.index
hpt_rul_val = val_data.loc[valid_indices_val, "Cycles_to_HPT_SV"]
hpc_rul_val = val_data.loc[valid_indices_val, "Cycles_to_HPC_SV"]
ww_rul_val  = val_data.loc[valid_indices_val, "Cycles_to_WW"]
T3_res_val = res_val["Sensed_T3"]
T45_res_val = res_val["Sensed_T45"]


# %store hpt_rul_train
# %store hpc_rul_train
# %store ww_rul_train
# %store T3_res_train
# %store T45_res_train
# %store hpt_rul_val
# %store hpc_rul_val
# %store ww_rul_val
# %store T3_res_val
# %store T45_res_val



# %%
res_test = pd.DataFrame()
res_list = []

for esn in dfv["ESN"].unique():
  mask = dfv["ESN"] == esn
  X_test = dfv.loc[mask, operating_vars]
  Y_test = dfv.loc[mask, degradation_vars]
  # Predict
  Y_pred = model.predict(X_test)
  res_temp = Y_test - Y_pred
  res_temp["ESN"] = esn
  res_list.append(res_temp)

res_test = pd.concat(res_list)


# PULIZIA E ROLLING
cleaned_chunks = []
for esn in dfv["ESN"].unique():
  temp = res_test[res_test["ESN"] == esn].copy()
  temp = remove_outliers(temp, SENSORS, threshold=0.8)
  temp[degradation_vars] = temp[degradation_vars].rolling(window=50, min_periods=1).mean()
  cleaned_chunks.append(temp)

res_test = pd.concat(cleaned_chunks).dropna()
# res_test.index = res_test.groupby("ESN").cumcount()

T3_res_test = res_test[["ESN", "Sensed_T3"]].copy()
T45_res_test = res_test[["ESN", "Sensed_T45"]].copy()

# Store dei risultati
# %store res_test
# %store T3_res_test
# %store T45_res_test

# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- 1. FUNZIONE PER IL CALCOLO DELLA PENDENZA (SLOPE) ---
def get_slope(y):
    if len(y) < 2 or np.all(np.isnan(y)):
        return 0.0
    x = np.arange(len(y))
    # Polyfit di grado 1: ritorna [pendenza, intercetta]
    return np.polyfit(x, y, 1)[0]

# --- 2. PREPARAZIONE FEATURE (SLOPE E RESIDUI) ---
window_slope = 50  # Finestra per la pendenza

print("Calcolo delle pendenze per i residui...")
for esn in res_train["ESN"].unique():
    mask = res_train["ESN"] == esn
    # Calcoliamo lo slope per i sensori principali
    # Usiamo transform per mantenere l'allineamento degli indici
    res_train.loc[mask, 'T45_Slope'] = res_train.loc[mask, 'Sensed_T45'].rolling(window=window_slope, min_periods=10).apply(get_slope)
    res_train.loc[mask, 'T3_Slope'] = res_train.loc[mask, 'Sensed_T3'].rolling(window=window_slope, min_periods=10).apply(get_slope)
    res_train.loc[mask, 'Ps3_Slope'] = res_train.loc[mask, 'Sensed_Ps3'].rolling(window=window_slope, min_periods=10).apply(get_slope)

# Riempiamo i NaN generati dalla rolling window iniziale
res_train[['T45_Slope', 'T3_Slope', 'Ps3_Slope']] = res_train[['T45_Slope', 'T3_Slope', 'Ps3_Slope']].fillna(0)

# --- 3. CONTROLLO CORRELAZIONE (Per rispondere alla tua domanda su T3) ---
print("\nVerifica Correlazione Slope vs RUL WW:")
target_ww = ww_rul_train['Cycles_to_WW'].values
for col in ['T45_Slope', 'T3_Slope', 'Ps3_Slope']:
    corr, _ = pearsonr(res_train[col], target_ww)
    print(f"Correlazione {col}: {corr:.4f}")

# --- 4. PREPARAZIONE DATASET DI TRAINING ---
# Scegliamo le feature in base a quelle che hanno mostrato correlazione nel plot/test
features_ww = ['Sensed_T45', 'T45_Slope', 'Ps3_Slope'] # Esempio: escludiamo T3 se non correla

X_train_ww = res_train[features_ww].copy()
# Applichiamo il clipping alla RUL (Target): ignoriamo tutto ciò che è sopra 150 cicli
# Questo serve a far concentrare il modello sulla fase critica pre-lavaggio
y_train_ww = ww_rul_train['Cycles_to_WW'].clip(upper=150)

# --- 5. ADDESTRAMENTO MODELLO LIGHTGBM ---
print("\nAddestramento LightGBM per Water Wash...")
lgbm_ww = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    num_leaves=31,
    importance_type='gain',
    reg_alpha=0.2,   # Regolarizzazione
    reg_lambda=0.2,
    random_state=42
)

lgbm_ww.fit(X_train_ww, y_train_ww)
print("Modello addestrato con successo!")

# --- 6. TEST E PLOT DI VERIFICA (Su un ESN di training) ---
esn_test = res_train["ESN"].unique()[0]
mask_test = res_train["ESN"] == esn_test

X_test_sample = X_train_ww[mask_test]
y_true_sample = ww_rul_train.loc[mask_test, 'Cycles_to_WW']
y_pred_sample = lgbm_ww.predict(X_test_sample)

plt.figure(figsize=(15, 6))
plt.plot(y_true_sample.values, label='RUL Reale (WW)', color='orange', linestyle='--', linewidth=2)
plt.plot(y_pred_sample, label='RUL Predetta (LGBM)', color='blue', alpha=0.8)
plt.axhline(y=30, color='red', linestyle=':', label='Soglia Allerta (30 cicli)')
plt.title(f"Verifica Predizione Water Wash - ESN {esn_test}")
plt.xlabel("Cicli")
plt.ylabel("RUL (Cicli al prossimo lavaggio)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Store per il notebook di test
# %store lgbm_ww
# %store features_ww

# %%
# PLOTTING TRAINING
for esn in train_data_healthy["ESN"].unique():
  fig, axs = plt.subplots(2,3, figsize=(15,8))
  fig.suptitle(f'ESN - {esn}', fontsize=16)
  for i, ax in enumerate(axs.flat):
    if isinstance(ax, plt.Axes):
        ax.plot(res_train.loc[res_train["ESN"] == esn, degradation_vars[i]], linewidth=1)
        ax.set_title(degradation_vars[i])
        ax.set_ylabel("Residuals")
        ax.set_xlabel(f"{degradation_vars[i]}_res")
        ax.grid()
  fig.subplots_adjust(hspace=0.4, wspace=0.4)
  fig.show()


# %%
# PLOTTING VALIDATION
fig, axs = plt.subplots(2,3, figsize=(15,8))
fig.suptitle(f'ESN - {testing_esn}', fontsize=16)
for i, ax in enumerate(axs.flat):
    if isinstance(ax, plt.Axes):
        ax.plot(res_val.loc[:, degradation_vars[i]], linewidth=1)
        ax.set_title(degradation_vars[i])
        ax.set_ylabel("Residuals")
        ax.set_xlabel(f"{degradation_vars[i]}_res")
        ax.grid()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.show()


# %%
# PLOTTING TEST
for esn in dfv["ESN"].unique():
  fig, axs = plt.subplots(2,3, figsize=(15,8))
  fig.suptitle(f'ESN - {esn}', fontsize=16)
  for i, ax in enumerate(axs.flat):
    if isinstance(ax, plt.Axes):
        ax.plot(res_test.loc[res_test["ESN"] == esn, degradation_vars[i]], linewidth=1)
        ax.set_title(degradation_vars[i])
        ax.set_ylabel("Residuals")
        ax.set_xlabel(f"{degradation_vars[i]}_res")
        ax.grid()
  fig.subplots_adjust(hspace=0.4, wspace=0.4)
  fig.show()

# %% [markdown]
# # HPT e HPC
# ## Ricerca di a,b,c,d,e,f,g globali combinazione lineare di tutti i sensori

# %%
# Standardizzazione dei residui

# Dati di training
train_chunks = []
dict_mins = {var: [] for var in degradation_vars}
dict_maxs = {var: [] for var in degradation_vars}
for esn in res_train["ESN"].unique():
  mask = res_train["ESN"] == esn
  mask_hpt = hpt_rul_train["ESN"] == esn
  hpt_rul_train.loc[mask, "Cycles_to_HPT_SV"] = minmax(hpt_rul_train[mask_hpt], "Cycles_to_HPT_SV")
  mask_hpc = hpc_rul_train["ESN"] == esn
  hpc_rul_train.loc[mask, "Cycles_to_HPC_SV"] = minmax(hpc_rul_train[mask_hpc], "Cycles_to_HPC_SV")
  mask_ww = ww_rul_train["ESN"] == esn
  ww_rul_train.loc[mask, "Cycles_to_WW"] = minmax(ww_rul_train[mask_ww], "Cycles_to_WW")
  for var in degradation_vars:
    dict_mins[var].append(res_train.loc[mask, var].min())
    dict_maxs[var].append(res_train.loc[mask, var].max())
    res_train.loc[mask, var] = minmax(res_train.loc[mask], var)
avg_mins_per_var = {var: np.mean(lst) for var, lst in dict_mins.items()}
avg_maxs_per_var = {var: np.mean(lst) for var, lst in dict_maxs.items()}

print(f'Scaled: {res_train}')

# Dati di validation
for var in degradation_vars:
    res_val.loc[var] = minmax(res_val, var)
hpt_rul_val_scaled = normalize(hpt_rul_val)
hpc_rul_val_scaled = normalize(hpc_rul_val)
ww_rul_val_scaled = normalize(ww_rul_val)


# Dati di test
for esn in dfv["ESN"].unique():
    mask = res_test["ESN"] == esn
    for var in degradation_vars:
        # Usa i limiti medi del training
        m = avg_mins_per_var[var]
        M = avg_maxs_per_var[var]
        res_test.loc[mask, var] = (res_test.loc[mask, var] - m) / (M - m)



# %store res_train
# %store res_val
# %store res_test
# %store hpt_rul_train
# %store hpc_rul_train
# %store ww_rul_train
# %store hpt_rul_val_scaled
# %store hpc_rul_val_scaled
# %store ww_rul_val_scaled
# %store avg_mins_per_var
# %store avg_maxs_per_var


# %%
# Preprocessing per il training dell'ottimizzatore ottimizzatore

all_train_res = []
all_train_hpt_rul = []
all_train_hpc_rul = []
all_train_ww_rul = []

for esn in res_train["ESN"].unique():
    res_esn = res_train.loc[res_train["ESN"] == esn, ["ESN"] + degradation_vars]

    rul_hpt_esn = hpt_rul_train.loc[hpt_rul_train["ESN"] == esn]
    rul_hpc_esn = hpc_rul_train.loc[hpc_rul_train["ESN"] == esn]
    rul_ww_esn = ww_rul_train.loc[ww_rul_train["ESN"] == esn]

    all_train_res.append(res_esn)
    all_train_hpt_rul.append(rul_hpt_esn)
    all_train_hpc_rul.append(rul_hpc_esn)
    all_train_ww_rul.append(rul_ww_esn)


# Impiliamo tutto: diventano quattro matrici uniche
X_train_opt = pd.concat(all_train_res, ignore_index=True)
Y_train_hpt = pd.concat(all_train_hpt_rul, ignore_index=True)
Y_train_hpc = pd.concat(all_train_hpc_rul, ignore_index=True)
Y_train_ww = pd.concat(all_train_ww_rul, ignore_index=True)


# %%
bounds = [
    (-1000, 1000),  # a
    (-1000, 1000),  # b
    (-1000, 1000),  # c
    (-1000, 1000),  # d
    (-1000, 1000),  # e
    (-1000, 1000),  # f
]

all_coefs_hpt = []
all_coefs_hpc = []
# all_coefs_ww = []

for esn in res_train["ESN"].unique():
  result_hpt = differential_evolution(
      objective_deviation,
      bounds=bounds,
      args=(X_train_opt.loc[X_train_opt["ESN"] == esn, degradation_vars], Y_train_hpt.loc[Y_train_hpt["ESN"] == esn, "Cycles_to_HPT_SV"]),
      strategy='best1bin',
      maxiter=1500,                # generazioni
      popsize=5,
      workers=-1,
      tol=0.02,                      # Tolleranza
  )
  all_coefs_hpt.append(result_hpt.x)

  result_hpc = differential_evolution(
      objective_deviation,
      bounds=bounds,
      args=(X_train_opt.loc[X_train_opt["ESN"] == esn, degradation_vars], Y_train_hpc.loc[Y_train_hpc["ESN"] == esn, "Cycles_to_HPC_SV"]),
      strategy='best1bin',
      maxiter=1500,                # generazioni
      popsize=5,
      workers=-1,
      tol=0.02,                      # Tolleranza
  )
  all_coefs_hpc.append(result_hpc.x)

  # result_ww = differential_evolution(
  #     objective_deviation,
  #     bounds=bounds,
  #     args=(X_train_opt.loc[X_train_opt["ESN"] == esn, degradation_vars], Y_train_ww.loc[Y_train_ww["ESN"] == esn, "Cycles_to_WW"]),
  #     strategy='best1bin',
  #     maxiter=400,                # generazioni
  #     popsize=100,
  #     workers=-1,
  #     tol=0,                      # Tolleranza
  # )
  # all_coefs_ww.append(result_ww.x)

coefs_hpt = np.mean(all_coefs_hpt, axis=0)
coefs_hpc = np.mean(all_coefs_hpc, axis=0)
# coefs_ww  = np.mean(all_coefs_ww, axis=0)

# Stampa dei risultati medi
print("\nCOEFFICIENTI MEDI FINALI (Training Set):")
print(f"HPT: {coefs_hpt}")
print(f"HPC: {coefs_hpc}")
# print(f"WW:  {coefs_ww}")

# Store dei coefficienti medi per il testing
# %store coefs_hpt
# %store coefs_hpc
# # %store coefs_ww


# %%
# PARAMETRI DI TEST SOLO PER HPT E HPC

coefs_hpt = [276.19439386,
             -343.90020315,
             781.45341037,
             -593.41135514,
             -190.36846837,
             118.5466599]

coefs_hpc = [354.11354513,
             295.34981181,
             -386.49672567,
             790.99844931,
             -679.69393928,
             487.61887153]

# %store coefs_hpt
# %store coefs_hpc

# %%
# PLOTTING SU DATI DI TRAINING

hpt_limits, hpc_limits, ww_limits = [], [], []
for esn in res_train["ESN"].unique():
  temp = res_train[res_train["ESN"] == esn].copy()
  
  # Calcolo e standardizzazione degli health index
  hi_hpt = normalize(HIE(coefs_hpt, temp[degradation_vars]))
  hi_hpc = normalize(HIE(coefs_hpc, temp[degradation_vars]))
  # hi_ww = normalize(HIE(coefs_ww, temp[degradation_vars]))

  hpt_limits.append((hi_hpt.min(), hi_hpt.max()))
  hpc_limits.append((hi_hpc.min(), hi_hpc.max()))
  # ww_limits.append((hi_ww.min(), hi_ww.max()))

  # Definizione delle RUL efettive
  hpt_rul_esn = hpt_rul_train[hpt_rul_train["ESN"] == esn].copy()
  hpc_rul_esn = hpc_rul_train[hpc_rul_train["ESN"] == esn].copy()
  ww_rul_esn  = ww_rul_train[ww_rul_train["ESN"] == esn].copy()

  fig, axs = plt.subplots(1, 2, figsize=(30, 6))
  fig.suptitle(f'Training: ESN - {esn}', fontsize=16)
  axs[0].plot(hi_hpt, color='tab:blue', label='Health Index (HPT)')
  axs[0].plot(hpt_rul_esn["Cycles_to_HPT_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  axs[1].plot(hi_hpc, color='tab:green', label='Health Index (HPC)')
  axs[1].plot(hpc_rul_esn["Cycles_to_HPC_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  # axs[2].plot(hi_ww, color='tab:green', label='Health Index (HPC)')
  # axs[2].plot(ww_rul_esn["Cycles_to_WW"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  fig.tight_layout()
  fig.show()


def get_avg_limits(limits_list):
    mins = [l[0] for l in limits_list]
    maxs = [l[1] for l in limits_list]
    return np.mean(mins), np.mean(maxs)

# Calcolo limiti medi finali
avg_min_hpt, avg_max_hpt = get_avg_limits(hpt_limits)
avg_min_hpc, avg_max_hpc = get_avg_limits(hpc_limits)
# avg_min_ww,  avg_max_ww  = get_avg_limits(ww_limits)

print(f"Limiti Medi HPT: Min={avg_min_hpt:.4f}, Max={avg_max_hpt:.4f}")

# Store per usarli nel notebook del Testing
# %store avg_min_hpt
# %store avg_max_hpt
# %store avg_min_hpc
# %store avg_max_hpc
# # %store avg_min_ww
# # %store avg_max_ww

# %%
# PLOTTING SU DATI DI VALIDATION

hi_hpt_val = normalize(HIE(coefs_hpt, res_val[degradation_vars]).dropna())
hi_hpc_val = normalize(HIE(coefs_hpc, res_val[degradation_vars]).dropna())
# hi_ww_val = normalize(HIE(coefs_ww, res_val[degradation_vars]).dropna())

fig, axs = plt.subplots(1, 2, figsize=(30, 6))
fig.suptitle(f'Validation', fontsize=16)
axs[0].plot(hi_hpt_val, color='tab:blue', label='Health Index (HPT)')
axs[0].plot(hpt_rul_val_scaled, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
axs[1].plot(hi_hpc_val, color='tab:green', label='Health Index (HPC)')
axs[1].plot(hpc_rul_val_scaled, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
# axs[2].plot(hi_ww_val, color='tab:green', label='Health Index (HPC)')
# axs[2].plot(ww_rul_val_scaled, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
fig.tight_layout()
fig.show()

# %%
# PLOTTING SU DATI DI TESTING

for esn in dft["ESN"].unique():
  temp = res_test[res_test["ESN"] == esn].copy()

  # Calcolo e standardizzazione degli health index
  hi_hpt_test = normalize(HIE(coefs_hpt, temp[degradation_vars]).dropna())
  hi_hpc_test = normalize(HIE(coefs_hpc, temp[degradation_vars]).dropna())
  # hi_ww_test  = normalize(HIE(coefs_ww, temp[degradation_vars]).dropna())

  fig, axs = plt.subplots(1, 2, figsize=(30, 6))
  fig.suptitle(f'Test: ESN - {esn}', fontsize=16)
  axs[0].plot(hi_hpt_test, color='tab:blue', label='Health Index (HPT)')
  axs[1].plot(hi_hpc_test, color='tab:green', label='Health Index (HPC)')
  # axs[2].plot(hi_ww_test, color='tab:red', label='Health Index (HPC)')
  fig.tight_layout()
  fig.show()

# %% [markdown]
# ### Classificazione dell'errore con LightGBM per HPC, HPT e WW

# %%
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. Recupero gli Eventi (Indici)
# ---------------------------------------------------------
# Assumiamo che i dataframe hpc_rul_train, hpt_rul_train, ww_rul_train e res_train siano già caricati
target_esns = [101, 102, 103, 104]

def get_events(df_rul, col_name):
    events = {}
    for esn in target_esns:
        # Troviamo gli indici dove il countdown è 0
        mask = (df_rul["ESN"] == esn) & (df_rul[col_name] == 0)
        events[esn] = sorted(df_rul[mask].index.tolist())
    return events

hpc_events = get_events(hpc_rul_train, "Cycles_to_HPC_SV")
hpt_events = get_events(hpt_rul_train, "Cycles_to_HPT_SV")
ww_events  = get_events(ww_rul_train,  "Cycles_to_WW")

# ---------------------------------------------------------
# 2. Logica di Reset dei Contatori
# ---------------------------------------------------------

# Inizializziamo le nuove colonne a 0 (o NaN se preferisci vederlo vuoto)
res_train["Cycle_count_HPC"] = 0
res_train["Cycle_count_HPT"] = 0
res_train["Cycle_count_WW"] = 0

# Dizionario per mappare colonna -> dizionario eventi corrispondente
config = {
    "Cycle_count_HPC": hpc_events,
    "Cycle_count_HPT": hpt_events,
    "Cycle_count_WW":  ww_events
}

print("Inizio calcolo contatori con reset...")

for esn in target_esns:
    # Filtriamo il dataframe principale per il motore corrente
    # Troviamo l'indice iniziale e finale di questo ESN in res_train
    mask_esn = res_train["ESN"] == esn
    if not mask_esn.any():
        continue
    
    esn_indices = res_train[mask_esn].index
    start_global = esn_indices[0] # Primo indice assoluto dell'ESN
    end_global = esn_indices[-1]  # Ultimo indice assoluto dell'ESN

    # Per ogni tipo di evento (HPC, HPT, WW)
    for col_name, event_dict in config.items():
        breakpoints = event_dict.get(esn, [])
        
        current_start = start_global
        
        # Iteriamo attraverso gli eventi di rottura
        for event_idx in breakpoints:
            # Assicuriamoci che l'evento sia dentro il range attuale dell'ESN
            if event_idx < current_start or event_idx > end_global:
                continue
            
            # Calcoliamo la lunghezza del segmento
            # Da current_start fino a event_idx (incluso)
            length = (event_idx - current_start) + 1
            
            # Creiamo la sequenza 0, 1, 2... fino a length-1
            counter_values = np.arange(length)
            
            # Assegniamo alla colonna specifica
            res_train.loc[current_start:event_idx, col_name] = counter_values
            
            # Il prossimo conteggio ripartirà dall'indice successivo all'evento
            current_start = event_idx + 1
            
        # GESTIONE CODA:
        # Se dopo l'ultimo evento ci sono ancora dati fino alla fine dell'ESN
        if current_start <= end_global:
            length = (end_global - current_start) + 1
            counter_values = np.arange(length)
            res_train.loc[current_start:end_global, col_name] = counter_values

print("Calcolo completato.")

# ---------------------------------------------------------
# 3. Controllo
# ---------------------------------------------------------
# Stampiamo un esempio per vedere se il contatore si resetta
# Prendiamo un ESN e una colonna (es. HPC) intorno a un evento
sample_esn = 101
if hpc_events[sample_esn]:
    evt = hpc_events[sample_esn][0] # Primo evento HPC
    print(f"\nVerifica reset HPC per ESN {sample_esn} all'evento {evt}:")
    # Mostriamo 5 righe prima e 5 dopo l'evento
    print(res_train.loc[evt-5 : evt+5, ["ESN", "Cycle_count_HPC"]])

# %%
# NON USARE QUESTO

# TRAINING REGRESSORE PER ERRORE GAP
# lightgbm trainato su spezzoni di eventi
# un modello per hpt e hpc e uno per ww???????????

# Regressori per il calcolo dell'errore (tra predizione e RUL reale)
# NON CREDO CHE SERVANO
# regr_hpc = LinearRegression()
# regr_hpt = LinearRegression()
# regr_ww = LinearRegression()

# Liste per il training dei regressori
all_train_hi_hpt = []
all_train_hpt_rul = []
all_train_hi_hpc = []
all_train_hpc_rul = []
# all_train_hi_ww = []
# all_train_ww_rul = []

# HI di training
for esn in res_train["ESN"].unique():
  temp = res_train[res_train["ESN"] == esn].copy()
  hi_hpt = normalize(HIE(coefs_hpt, temp[degradation_vars]).dropna())
  hi_hpc = normalize(HIE(coefs_hpc, temp[degradation_vars]).dropna())
  # hi_ww = normalize(HIE(coefs_ww, temp[degradation_vars]).dropna())
  print(f'SHAPE: {hi_hpt.shape}')

  hpt_rul_esn = hpt_rul_train[hpt_rul_train["ESN"] == esn].copy()
  hpc_rul_esn = hpc_rul_train[hpc_rul_train["ESN"] == esn].copy()
  # ww_rul_esn  = ww_rul_train[ww_rul_train["ESN"] == esn].copy()
  rul_hpt_esn = hpt_rul_esn.iloc[:, 1]
  rul_hpc_esn = hpc_rul_esn.iloc[:, 1]
  # rul_ww_esn  = ww_rul_esn.iloc[:, 1]

  all_train_hi_hpt.append(hi_hpt)
  all_train_hi_hpc.append(hi_hpc)
  # all_train_hi_ww.append(hi_ww)
  all_train_hpt_rul.append(rul_hpt_esn)
  all_train_hpc_rul.append(rul_hpc_esn)
  # all_train_ww_rul.append(rul_ww_esn)


# Impiliamo tutto: diventano quattro matrici uniche
X_train_hpt = pd.concat(all_train_hi_hpt, ignore_index=True).dropna()
X_train_hpc = pd.concat(all_train_hi_hpc, ignore_index=True).dropna()
# X_train_ww = pd.concat(all_train_hi_ww, ignore_index=True).dropna()

Y_train_hpt = pd.concat(all_train_hpt_rul, ignore_index=True).dropna()
Y_train_hpc = pd.concat(all_train_hpc_rul, ignore_index=True).dropna()
# Y_train_ww = pd.concat(all_train_ww_rul, ignore_index=True).dropna()


# TRAINING dei regressori per l'errore
print(f'X TRAIN: {X_train_hpt.shape}')
print(f'Y TRAIN: {Y_train_hpt.shape}')

regr_hpt.fit(X_train_hpt, Y_train_hpt)
regr_hpc.fit(X_train_hpc, Y_train_hpc)
# regr_ww.fit(X_train_ww, X_train_ww)


test_hi_hpt = normalize_hi(HIE(coefs_hpt, res_test[res_test["ESN"] == 106][degradation_vars]))
test_hi_hpc = normalize_hi(HIE(coefs_hpc, res_test[res_test["ESN"] == 106][degradation_vars]))
test_hi_ww = normalize_hi(HIE(coefs_ww, res_test[res_test["ESN"] == 106][degradation_vars]))

val_hi_hpt = normalize_hi(HIE(coefs_hpt, res_val[degradation_vars])).dropna()
val_hi_hpc = normalize_hi(HIE(coefs_hpc, res_val[degradation_vars])).dropna()
val_hi_ww = normalize_hi(HIE(coefs_ww, res_val[degradation_vars])).dropna()

# %store X_train_hpt
# %store X_train_hpc
# %store X_train_ww
# %store Y_train_hpt
# %store Y_train_hpc
# %store Y_train_ww
# %store regr_hpt
# %store regr_hpc
# %store regr_ww
# %store all_train_hpt_rul
# %store all_train_hpc_rul
# %store all_train_ww_rul



# %%
# LIGHTGBM

X_hpc_list, y_hpc_list = [], []
X_hpt_list, y_hpt_list = [], []
X_ww_list, y_ww_list = [], []

# Sui dati di training
for esn in res_train["ESN"].unique():

    temp = res_train[res_train["ESN"] == esn].reset_index().copy()

    # Calcolo e standardizzazione degli health index
    hi_hpt = normalize(HIE(coefs_hpt, temp[degradation_vars])).dropna()
    hi_hpc = normalize(HIE(coefs_hpc, temp[degradation_vars])).dropna()
    # hi_ww = normalize(HIE(coefs_ww, temp[degradation_vars])).dropna()
    print(f'SHAPE: {hi_hpt.shape}')

    # Valori predetti
    # pred_rul_hpt = regr_hpt.predict(hi_hpt)
    # pred_rul_hpc = regr_hpc.predict(hi_hpc)
    # pred_rul_ww = regr_hpc.predict(hi_ww)

    hpt_rul = hpt_rul_train[hpt_rul_train["ESN"] == esn].copy()
    hpc_rul = hpc_rul_train[hpc_rul_train["ESN"] == esn].copy()
    # ww_rul = ww_rul_train[ww_rul_train["ESN"] == esn].copy()

    # Plot intermedio
    # fig, axs = plt.subplots(1, 3, figsize=(30, 6))
    # fig.suptitle(f'Training: ESN - {esn}', fontsize=16)
    # axs[0].plot(hi_hpt, color='tab:blue', label='Health Index (HPT)')
    # axs[0].plot(pred_rul_hpt, color='tab:orange', linewidth=2, linestyle='--', label='Predicted')
    # axs[1].plot(hi_hpc, color='tab:blue', label='Health Index (HPC)')
    # axs[1].plot(pred_rul_hpc, color='tab:orange', linewidth=2, linestyle='--', label='Predicted')
    # axs[2].plot(hi_ww, color='tab:blue', label='Health Index (HPC)')
    # axs[2].plot(pred_rul_ww, color='tab:orange', linewidth=2, linestyle='--', label='Predicted')
    # fig.tight_layout()
    # fig.show()

    # Calcolo errore gap
    gap_true_hpt = hpt_rul["Cycles_to_HPT_SV"].values.flatten() - np.asarray(hi_hpt).flatten()
    gap_true_hpc = hpc_rul["Cycles_to_HPC_SV"].values.flatten() - np.asarray(hi_hpc).flatten()
    # gap_true_ww = ww_rul["Cycles_to_WW"].values.flatten() - np.asarray(hi_ww).flatten()

    window_size = 100

    feat_slope_hpt, feat_intercept_hpt = get_rolling_slope_intercept(hi_hpt, window_size)
    feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc, window_size)
    # feat_slope_ww, feat_intercept_ww = get_rolling_slope_intercept(hi_ww, window_size)

    # Accumulo dati HPT
    X_hpt_list.append(pd.DataFrame({'HI': np.asarray(hi_hpt).flatten(), 'Slope': feat_slope_hpt, 'Intercept': feat_intercept_hpt}))
    y_hpt_list.append(gap_true_hpt)

    # Accumulo dati HPC
    X_hpc_list.append(pd.DataFrame({'HI': np.asarray(hi_hpc).flatten(), 'Slope': feat_slope_hpc, 'Intercept': feat_intercept_hpc}))
    y_hpc_list.append(gap_true_hpc)

    # Accumulo dati WW
    # X_ww_list.append(pd.DataFrame({'HI': np.asarray(hi_ww).flatten(), 'Slope': feat_slope_ww, 'Intercept': feat_intercept_ww}))
    # y_ww_list.append(gap_true_ww)

    
X_train_hpc = pd.concat(X_hpc_list, ignore_index=True)
y_train_hpc = np.concatenate(y_hpc_list)

X_train_hpt = pd.concat(X_hpt_list, ignore_index=True)
y_train_hpt = np.concatenate(y_hpt_list)

# X_train_ww = pd.concat(X_ww_list, ignore_index=True)
# y_train_ww = np.concatenate(y_hpt_list)

# Training dei modelli
print("Training LGBM HPC...")
lgbm_hpc = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03)
lgbm_hpc.fit(X_train_hpc, y_train_hpc)

print("Training LGBM HPT...")
lgbm_hpt = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03)
lgbm_hpt.fit(X_train_hpt, y_train_hpt)

# print("Training LGBM WW...")
# lgbm_ww = lgb.LGBMRegressor(n_estimators=10000, learning_rate=0.01)
# lgbm_ww.fit(X_train_ww, y_train_ww)


# %store lgbm_hpc
# %store lgbm_hpt
# # %store lgbm_ww



# %%
# Test sui dati di training

for esn in res_train["ESN"].unique():
    temp = res_train[res_train["ESN"] == esn].reset_index().copy()

    # Calcolo e standardizzazione degli health index
    hi_hpt = normalize(HIE(coefs_hpt, temp[degradation_vars])).dropna()
    hi_hpc = normalize(HIE(coefs_hpc, temp[degradation_vars])).dropna()
    # hi_ww = normalize(HIE(coefs_ww, temp[degradation_vars])).dropna()
    print(f'SHAPE: {hi_hpt.shape}')

    # RUL effettiva
    hpt_rul = hpt_rul_train.loc[hpt_rul_train["ESN"] == esn, "Cycles_to_HPT_SV"].values.copy()
    hpc_rul = hpc_rul_train.loc[hpc_rul_train["ESN"] == esn, "Cycles_to_HPC_SV"].values.copy()
    # ww_rul = ww_rul_train.loc[ww_rul_train["ESN"] == esn, "Cycles_to_WW"].values.copy()

    # HPT
    feat_slope_hpt, feat_intercept_hpt = get_rolling_slope_intercept(hi_hpt, window_size)
    X_lgbm_hpt = pd.DataFrame({
        'HI': hi_hpt.values.flatten(),                # Valore attuale HI
        'Slope': feat_slope_hpt,            # Pendenza locale
        'Intercept': feat_intercept_hpt     # Intercetta locale
    })
    pred_gap_hpt = lgbm_hpt.predict(X_lgbm_hpt)
    pred_rul_hpt = hi_hpt.values.flatten() + pred_gap_hpt

    # HPC
    feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc, window_size)
    X_lgbm_hpc = pd.DataFrame({
        'HI': hi_hpc.values.flatten(),                # Valore attuale HI
        'Slope': feat_slope_hpc,            # Pendenza locale
        'Intercept': feat_intercept_hpc     # Intercetta locale
    })
    pred_gap_hpc = lgbm_hpc.predict(X_lgbm_hpc)
    pred_rul_hpc = hi_hpc.values.flatten() + pred_gap_hpc

    # # WW
    # feat_slope_ww, feat_intercept_ww = get_rolling_slope_intercept(hi_ww, window_size)
    # X_lgbm_ww = pd.DataFrame({
    #     'HI': hi_ww.values.flatten(),                 # Valore attuale HI
    #     'Slope': feat_slope_ww,             # Pendenza locale
    #     'Intercept': feat_intercept_ww      # Intercetta locale
    # })
    # pred_gap_ww = lgbm_ww.predict(X_lgbm_ww)
    # pred_rul_ww = hi_ww.values.flatten() + pred_gap_ww


    fig, axs = plt.subplots(1, 2, figsize=(30, 6))
    fig.suptitle(f'Training: ESN - {esn}', fontsize=16)
    axs[0].plot(pred_rul_hpt, color='tab:blue', label='Predicted - HPT')
    axs[0].plot(hpt_rul, color='tab:orange', linewidth=2, linestyle='--', label='Real')
    axs[1].plot(pred_rul_hpc, color='tab:blue', label='Predicted - HPC')
    axs[1].plot(hpc_rul, color='tab:orange', linewidth=2, linestyle='--', label='Real')
    # axs[2].plot(pred_rul_ww, color='tab:blue', label='Predicted - WW')
    # axs[2].plot(ww_rul, color='tab:orange', linewidth=2, linestyle='--', label='Real')
    fig.tight_layout()
    fig.show()

    
    

# %%
# Test sul motore di VALIDATION

# Calcolo e standardizzazione degli health index
val_hi_hpt = normalize(HIE(coefs_hpt, res_val[degradation_vars])).dropna()
val_hi_hpc = normalize(HIE(coefs_hpc, res_val[degradation_vars])).dropna()
# val_hi_ww = normalize(HIE(coefs_ww, res_val[degradation_vars])).dropna()
print(f'SHAPE: {val_hi_hpt.shape}')

# RUL effettiva
val_hpt_rul = hpt_rul_val_scaled.values.copy()
val_hpc_rul = hpc_rul_val_scaled.values.copy()
# val_ww_rul = ww_rul_val_scaled.values.copy()


# HPT
feat_slope_hpt, feat_intercept_hpt = get_rolling_slope_intercept(val_hi_hpt, window_size)
X_lgbm_hpt_val = pd.DataFrame({
    'HI': val_hi_hpt.values.flatten(),                # Valore attuale HI
    'Slope': feat_slope_hpt,            # Pendenza locale
    'Intercept': feat_intercept_hpt     # Intercetta locale
})
pred_gap_hpt_val = lgbm_hpt.predict(X_lgbm_hpt_val)
pred_rul_hpt_val = val_hi_hpt.values.flatten() + pred_gap_hpt_val

# HPC
feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(val_hi_hpc, window_size)
X_lgbm_hpc_val = pd.DataFrame({
    'HI': val_hi_hpc.values.flatten(),                # Valore attuale HI
    'Slope': feat_slope_hpc,            # Pendenza locale
    'Intercept': feat_intercept_hpc     # Intercetta locale
})
pred_gap_hpc_val = lgbm_hpc.predict(X_lgbm_hpc_val)
pred_rul_hpc_val = val_hi_hpc.values.flatten() + pred_gap_hpc_val

# # WW
# feat_slope_ww, feat_intercept_ww = get_rolling_slope_intercept(val_hi_ww, window_size)
# X_lgbm_ww_val = pd.DataFrame({
#     'HI': val_hi_ww.values.flatten(),                 # Valore attuale HI
#     'Slope': feat_slope_ww,             # Pendenza locale
#     'Intercept': feat_intercept_ww      # Intercetta locale
# })
# pred_gap_ww_val = lgbm_ww.predict(X_lgbm_ww_val)
# pred_rul_ww_val = val_hi_ww.values.flatten() + pred_gap_ww_val


fig, axs = plt.subplots(1, 2, figsize=(30, 6))
fig.suptitle(f'Validation: ESN - {testing_esn}', fontsize=16)
axs[0].plot(pred_rul_hpt_val, color='tab:blue', label='Predicted - HPT')
axs[0].plot(val_hpt_rul, color='tab:orange', linewidth=2, linestyle='--', label='Real')
axs[1].plot(pred_rul_hpc_val, color='tab:blue', label='Predicted - HPC')
axs[1].plot(val_hpc_rul, color='tab:orange', linewidth=2, linestyle='--', label='Real')
# axs[2].plot(pred_rul_ww_val, color='tab:blue', label='Predicted - WW')
# axs[2].plot(val_ww_rul, color='tab:orange', linewidth=2, linestyle='--', label='Real')
fig.tight_layout()
fig.show()


# %% [markdown]
# # WW

# %%
# TENTATIVO PER WW: CONTROLLO SE LA PENDENZA DEL RESIDUO DI T45 AUMENTA ALL'AVVICINARSI DEL WW

# Calcolo della slope del residuo di T3
window_slope = 50
res_train['T3_Slope'] = res_train.groupby('ESN')['Sensed_Ps3'].transform(
    lambda x: x.rolling(window=window_slope).apply(get_slope)
)

# Riempimento dei NaN con 0
res_train['T3_Slope'] = normalize(res_train['T3_Slope'].fillna(0))


# Iterazione per ogni ESN di train
for esn in res_train["ESN"].unique():
    data_plot = res_train[res_train["ESN"] == esn]
    target_plot = ww_rul_train[ww_rul_train["ESN"] == esn]

    # PLOT
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlabel('Cicli')
    ax.set_ylabel('Residuo Sensed_T3', color='tab:blue')
    ax.plot(data_plot.index, data_plot['Sensed_Ps3'].rolling(20).mean(), color='tab:blue', linewidth=2, label='Residuo T3 (Smooth)')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_ylabel('Slope (Trend di Degrado)', color='tab:red')
    ax.plot(data_plot.index, data_plot['T3_Slope'], color='tab:red', linewidth=2, label='T3 Slope (Window 50)')
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax.spines.right.set_position(("axes", 1.1))
    ax.plot(data_plot.index, target_plot['Cycles_to_WW'], color='gray', linestyle='--', alpha=0.5, label='RUL WW Reale')
    ax.set_ylabel('RUL (Cicli mancanti al WW)', color='gray')

    plt.title(f'Analisi del Degrado per ESN {esn}: Residuo vs Slope vs RUL', fontsize=16)
    fig.tight_layout()

    # Uniamo le legende di tutti gli assi
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax.get_legend_handles_labels()
    lines3, labels3 = ax.get_legend_handles_labels()
    ax.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')

    plt.grid(True, alpha=0.3)
    plt.show()



# %%
# TRAINING 

window_slope = 50
sensors_to_use = ['Sensed_T3', 'Sensed_T45']

print("Calcolo degli Slope per i sensori selezionati...")
for sensor in sensors_to_use:
    col_name = f'{sensor}_Slope'
    # Calcoliamo lo slope mobile per ogni motore
    res_train[col_name] = normalize(res_train.groupby('ESN')[sensor].transform(
        lambda x: x.rolling(window=window_slope, min_periods=window_slope).apply(get_slope)
    ).fillna(0))

# --- 3. PREPARAZIONE DATASET DI TRAINING ---
# Selezioniamo i residui originali + gli slope appena calcolati
features = [f'{s}_Slope' for s in sensors_to_use]

X_train_ww = res_train[features]
y_train_ww = ww_rul_train['Cycles_to_WW']

# Trucco: Clippiamo la RUL a 150. Oltre i 150 cicli, non ci interessa la precisione
# y_train_ww_clipped = y_train_ww.clip(upper=150)

# --- 4. TRAINING DEL MODELLO LIGHTGBM ---
print("Inizio addestramento LightGBM per Water Wash...")
lgbm_ww = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    importance_type='gain',
    reg_alpha=0.2,   # L1 regularization
    reg_lambda=0.2,  # L2 regularization
    random_state=42
)

lgbm_ww.fit(X_train_ww, y_train_ww)
print("Addestramento completato.")

# --- 5. TEST E PLOT DEI RISULTATI ---
# Proviamo la predizione su un motore di training per vedere se "fitta"
esn_test = res_train["ESN"].unique()[0]
mask = res_train["ESN"] == esn_test

X_example = X_train_ww[mask]
print(f'COSE: {X_example}')
y_real = y_train_ww[mask]
y_pred = lgbm_ww.predict(X_example)

plt.figure(figsize=(15, 6))
plt.plot(y_real.values, label='RUL Reale (WW)', color='orange', linewidth=2)
plt.plot(y_pred, label='RUL Predetta (LGBM)', color='blue')
# plt.title(f"Verifica Modello WW - ESN {esn_test}")
plt.xlabel("Cicli")
plt.ylabel("RUL (Cicli)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# %store lgbm_ww

# %%
sensors_to_use = ['Sensed_T3', 'Sensed_T45']

# Creiamo una copia per non sporcare i dati originali
X_test_ww = res_val[sensors_to_use].copy()

# Calcoliamo le pendenze (Slope) anche per il set di validazione
# Usiamo la stessa finestra (window_slope = 50) del training
for sensor in sensors_to_use:
    col_name = f'{sensor}_Slope'
    X_test_ww[col_name] = normalize(X_test_ww[sensor].rolling(window=50, min_periods=50).apply(get_slope).fillna(0))

# Selezioniamo solo le colonne usate durante il training (ordine corretto)
features = [f'{s}_Slope' for s in sensors_to_use]
X_test_input = X_test_ww[features]


# Eseguiamo la predizione
y_pred_val = lgbm_ww.predict(X_test_input)


# Recuperiamo la RUL reale per il confronto
y_true_val = ww_rul_val_scaled.values 

# --- PLOT DI VALIDAZIONE ---
plt.figure(figsize=(15, 6))
plt.plot(y_true_val, label='RUL Reale (Water Wash)', color='orange', linewidth=2)
plt.plot(y_pred_val, label='RUL Predetta (Modello)', color='blue')
plt.title(f'Validazione su ESN {testing_esn}: Predizione Eventi Water Wash', fontsize=16)
plt.xlabel('Cicli')
plt.ylabel('RUL (Cicli mancanti)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
