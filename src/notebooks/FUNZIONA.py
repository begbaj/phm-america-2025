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
from datetime import datetime
from enum import Enum
from numpy import sign
from pandas import DataFrame, Series
from plotly.graph_objs import Data
from pyparsing import line
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sympy import O, deg
from types import FunctionType
from xgboost import XGBRegressor
from xgboost import train
import glob
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
import pandas as pd
import pulp
import pwlf
import random
import scipy.optimize as optimize
import scipy.stats as stats


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# from tools import utils as u, config as cfg, plotting as up, preprocessing as pp
# %load_ext autoreload
# %autoreload 2

from tools import utils as u, config as cfg, plotting as up, preprocessing as pp


# %%
# CONFIG

DATA_BASE_PATH = f"../../Data/"
DATA_TESTING_PATH= f"{DATA_BASE_PATH}/PHM2025_test_data/"
DATA_TRAINING_PATH = f"{DATA_BASE_PATH}/PHM2025_training_data/"
DATA_VALIDATION_PATH = f"{DATA_BASE_PATH}/PHM2025_validation_data/"
def DATA_TEST_DATA(num):
    return f"{DATA_TESTING_PATH}/test_{num}.csv"
def DATA_VALIDATION_DATA(num):
    return f"{DATA_TESTING_PATH}/val_{num}.csv"
DATA_TRAINING_DATA = f"{DATA_TESTING_PATH}/training_data.csv"
PLOT_PATH = f"./img/"

# %%
SENSORS = u.ESENSORS.values()

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
testing_esn = 102
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
degradation_vars = [s for s in SENSORS if s not in operating_vars and s != "Sensed_P25" and s != "Sensed_T5"]
managed_cols = set(degradation_vars) | set(operating_vars)

# %%
# PREPROCESSING DATI DI TRAINING
df = u.load_training()()
df = pp.remove_outliers(df, SENSORS)
df = pp.missingfill(df).dropna()

# PREPROCESSING DATI DI VALIDATION
dfv = u.load_validation(range(0,48))
dfv = pp.remove_outliers(dfv, SENSORS)
dfv = pp.missingfill(dfv, align_cols=["Snapshot", "Cycles"]).dropna()

# PREPROCESSING DATI DI TESTING
dft = u.load_testing(range(0,52))
dft = pp.remove_outliers(dft, SENSORS)
dft = pp.missingfill(dft, align_cols=["Snapshot", "Cycles"]).dropna()


# %store df
# %store dfv
# %store dft


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

cycles_healthy = 50

testing_esn = 102
val_data = df[df["ESN"] == testing_esn].reset_index().copy()
X_val = val_data[operating_vars]
Y_val = val_data[degradation_vars]

train_data_full = df[df["ESN"] != testing_esn].copy()
train_data_healthy = train_data_full.groupby("ESN").head(cycles_healthy).reset_index(drop=True)

base_df = train_data_healthy.copy()

# Data augmentation
new_synthetic_units = []
for i in range(400):
    aug_df = base_df.copy()
    noise = np.random.normal(loc=0, scale=10, size=aug_df[degradation_vars + operating_vars].shape)
    aug_df[degradation_vars + operating_vars] += noise
    aug_df['ESN'] = f"aug_{i}" 
    new_synthetic_units.append(aug_df)

train_data_healthy_augmented = pd.concat([train_data_healthy] + new_synthetic_units, ignore_index=True)

X_train = train_data_healthy_augmented[operating_vars]
Y_train = train_data_healthy_augmented[degradation_vars]

print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)

# training regressore lineare
model = train_model(X_train, Y_train)
# %store model

# Predict dei valori e calcolo dei residui per gli esn di training
# Mi porto dietro gli hpt cumulativi per il lightgbm
res_list = []
for esn in train_data_full["ESN"].unique():
  mask = train_data_full["ESN"] == esn
  X_train = train_data_full.loc[mask, operating_vars]
  Y_train = train_data_full.loc[mask, degradation_vars]
  Y_pred = model.predict(X_train)
  res_temp = Y_train - Y_pred
  res_temp["ESN"] = esn
  res_temp["Cumulative_HPC_SVs"] = train_data_full.loc[mask, "Cumulative_HPC_SVs"]
  res_list.append(res_temp)

res_train = pd.concat(res_list)

# Predict dei valori e calcolo dei residui per gli esn di "validation"
# Mi porto dietro gli hpt cumulativi per il lightgbm
res_val_list = []
Y_pred = model.predict(X_val)
res_val_temp = Y_val - Y_pred
res_val_temp["Cumulative_HPC_SVs"] = val_data["Cumulative_HPC_SVs"]
res_val_list.append(res_val_temp)
res_val = pd.concat(res_val_list)

# Integrazione residui sul dataset originale
cleaned_chunks = []
for esn in train_data_full["ESN"].unique():
  temp = res_train[res_train["ESN"] == esn].copy()
  temp[degradation_vars] = remove_outliers(temp[degradation_vars], SENSORS, threshold=0.8)
  temp[degradation_vars] = temp[degradation_vars].rolling(window=50, min_periods=1).median()
  temp = temp.dropna()
  cleaned_chunks.append(temp)

res_train = pd.concat(cleaned_chunks).dropna()
train_data_full.update(res_train)


# Integrazione residui sul dataset originale
res_val[degradation_vars] = remove_outliers(res_val[degradation_vars], SENSORS, threshold=0.8)
res_val[degradation_vars] = res_val[degradation_vars].rolling(window=50, min_periods=1).mean()
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

# PLOTTING TRAINING
fig, axs = plt.subplots(2,3, figsize=(15,8))
for esn in res_train["ESN"].unique():
  fig.suptitle(f'ESN - {esn}', fontsize=16)
  for i, ax in enumerate(axs.flat):
    if isinstance(ax, plt.Axes):
        degrad = res_train.loc[res_train["ESN"] == esn, degradation_vars[i]].reset_index(drop=True)
        ax.plot(degrad, linewidth=0.5, label=esn)
        ax.set_title(degradation_vars[i])
        ax.set_ylabel("Residuals")
        ax.set_xlabel(f"{degradation_vars[i]}_res")
        ax.legend()
        ax.grid()
  fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.show()

plt.figure()
for esn in res_train["ESN"].unique():
  prova = res_train[res_train["ESN"] ==  esn][degradation_vars].reset_index(drop=True).sum(axis=1)
  prova.plot(linewidth=0.3)
plt.legend()


# %%
# Calcolo dei residui per i dati di test

res_test = pd.DataFrame()
res_list = []


plt.figure(figsize=(16,10))
for i, esn in enumerate(dfv["ESN"].unique()):
  plt.subplot(2,2,i+1)
  mask = dfv["ESN"] == esn
  X_test = dfv.loc[mask, operating_vars]
  Y_test = dfv.loc[mask, degradation_vars]
  # Predict
  Y_pred = model.predict(X_test)

  plt.plot(Y_pred)
  plt.legend(degradation_vars)

  res_temp = Y_test - Y_pred
  res_temp = remove_outliers(res_temp, res_temp.columns, threshold=0.8)
  res_temp = res_temp.ffill()
  res_temp = res_temp.bfill()

  for col in res_temp.columns:
    res_temp[f"raw_{col}"] = res_temp[col]

  res_temp[degradation_vars] = res_temp[degradation_vars].cumsum()
  
  res_temp["ESN"] = esn
  res_list.append(res_temp)

plt.show()

res_test = pd.concat(res_list)

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
for esn in res_train["ESN"].unique():
  data = res_train.loc[res_train["ESN"] == esn, "Sensed_T45"].reset_index(drop=True)
  plt.plot(data, linewidth=1, label=f"T45 {esn} cumsum")
  plt.legend()

plt.subplot(3,1,2)
for esn in res_train["ESN"].unique():
  data = res_train.loc[res_train["ESN"] == esn, "Sensed_T45"].reset_index(drop=True)
  plt.plot(data.cumsum(), linewidth=1, label=f"T45 {esn} cumsum")
  plt.legend()

plt.subplot(3,1,3)
for esn in res_train["ESN"].unique():
  data = res_train.loc[res_train["ESN"] == esn, "Sensed_T45"].reset_index(drop=True)
  plt.plot(data.diff(), linewidth=1, label=f"T45 {esn} cumsum")
  plt.legend()
plt.show()


# PULIZIA E ROLLING
cleaned_chunks = []
for esn in dfv["ESN"].unique():
  temp = res_test[res_test["ESN"] == esn].copy()
  temp = remove_outliers(temp, SENSORS, threshold=0.9)
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
# PLOTTING TRAINING
fig, axs = plt.subplots(2,3, figsize=(15,8))
for esn in res_train["ESN"].unique():
  fig.suptitle(f'ESN - {esn}', fontsize=16)
  for i, ax in enumerate(axs.flat):
    if isinstance(ax, plt.Axes):
        degrad = res_train.loc[res_train["ESN"] == esn, degradation_vars[i]].reset_index(drop=True)
        ax.plot(degrad, linewidth=0.5, label=esn)
        ax.set_title(degradation_vars[i])
        ax.set_ylabel("Residuals")
        ax.set_xlabel(f"{degradation_vars[i]}_res")
        ax.legend()
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
# Preprocessing per il training dell'ottimizzatore

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

coefs_hpt = np.mean(all_coefs_hpt, axis=0)
coefs_hpc = np.mean(all_coefs_hpc, axis=0)

# Stampa dei risultati medi
print("\nCOEFFICIENTI MEDI FINALI (Training Set):")
print(f"HPT: {coefs_hpt}")
print(f"HPC: {coefs_hpc}")

# Store dei coefficienti medi per il testing
# %store coefs_hpt
# %store coefs_hpc


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

  hpt_limits.append((hi_hpt.min(), hi_hpt.max()))
  hpc_limits.append((hi_hpc.min(), hi_hpc.max()))

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
  fig.tight_layout()
  fig.show()


def get_avg_limits(limits_list):
    mins = [l[0] for l in limits_list]
    maxs = [l[1] for l in limits_list]
    return np.mean(mins), np.mean(maxs)

# Calcolo limiti medi finali
avg_min_hpt, avg_max_hpt = get_avg_limits(hpt_limits)
avg_min_hpc, avg_max_hpc = get_avg_limits(hpc_limits)

print(f"Limiti Medi HPT: Min={avg_min_hpt:.4f}, Max={avg_max_hpt:.4f}")

# Store per usarli nel notebook del Testing
# %store avg_min_hpt
# %store avg_max_hpt
# %store avg_min_hpc
# %store avg_max_hpc


# %%
# PLOTTING SU DATI DI VALIDATION

hi_hpt_val = normalize(HIE(coefs_hpt, res_val[degradation_vars]).dropna())
hi_hpc_val = normalize(HIE(coefs_hpc, res_val[degradation_vars]).dropna())

fig, axs = plt.subplots(1, 2, figsize=(30, 6))
fig.suptitle(f'Validation', fontsize=16)
axs[0].plot(hi_hpt_val, color='tab:blue', label='Health Index (HPT)')
axs[0].plot(hpt_rul_val_scaled, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
axs[1].plot(hi_hpc_val, color='tab:green', label='Health Index (HPC)')
axs[1].plot(hpc_rul_val_scaled, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
fig.tight_layout()
fig.show()


# %%
# PLOTTING SU DATI DI TEST

for esn in dft["ESN"].unique():
  temp = res_test[res_test["ESN"] == esn].copy()

  # Calcolo e standardizzazione degli health index
  hi_hpt_test = normalize(HIE(coefs_hpt, temp[degradation_vars]).dropna())
  hi_hpc_test = normalize(HIE(coefs_hpc, temp[degradation_vars]).dropna())

  fig, axs = plt.subplots(1, 2, figsize=(30, 6))
  fig.suptitle(f'Test: ESN - {esn}', fontsize=16)
  axs[0].plot(hi_hpt_test, color='tab:blue', label='Health Index (HPT)')
  axs[1].plot(hi_hpc_test, color='tab:green', label='Health Index (HPC)')
  fig.tight_layout()
  fig.show()
  

# %% [markdown]
# ### Classificazione dell'errore con LightGBM per HPC e HPT
# #### DA SISTEMARE

# %%
# VERSIONE LIGHTGBM DEL PAPER

X_hpc_list, y_hpc_list = [], []

# Sui dati di training
for esn in res_train["ESN"].unique():

    temp = res_train[res_train["ESN"] == esn].reset_index().copy()

    mask = temp["Cumulative_HPC_SVs"] == 2

    # Calcolo e standardizzazione degli health index
    hi_hpc = normalize(HIE(coefs_hpc, temp[degradation_vars])).dropna()
    hpc_rul = hpc_rul_train[hpc_rul_train["ESN"] == esn].copy()

    # Calcolo errore gap
    gap_true_hpc = hpc_rul["Cycles_to_HPC_SV"].values.flatten() - np.asarray(hi_hpc).flatten()

    window_size = 100

    feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc, window_size)

    indices = np.where(mask)[0]

    # Accumulo dati HPC
    X_hpc_list.append(pd.DataFrame({'HI': np.asarray(hi_hpc).flatten(),
                                    'Slope': feat_slope_hpc, 'Intercept': feat_intercept_hpc}).iloc[indices].dropna())
    y_hpc_list.append(gap_true_hpc[indices])
  
X_train_hpc = pd.concat(X_hpc_list, ignore_index=True)
y_train_hpc = np.concatenate(y_hpc_list)

# Training dei modelli
print("Training LGBM HPC...")
lgbm_hpc = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03)
lgbm_hpc.fit(X_train_hpc, y_train_hpc)


# %store lgbm_hpc



# %%
# TEST sui dati di TRAINING

for esn in res_train["ESN"].unique():
    temp = res_train[res_train["ESN"] == esn].reset_index().copy()
    # Maschera per l'applicazione del modello
    mask = temp["Cumulative_HPC_SVs"] == 2

    # Calcolo e standardizzazione degli health index
    hi_hpc = normalize(HIE(coefs_hpc, temp[degradation_vars])).values.flatten()

    # RUL effettiva
    hpc_rul = hpc_rul_train.loc[hpc_rul_train["ESN"] == esn, "Cycles_to_HPC_SV"].values

    # HPC
    feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc, window_size)
    X_lgbm_hpc_all = pd.DataFrame({
        'HI': hi_hpc,                # Valore attuale HI
        'Slope': feat_slope_hpc,            # Pendenza locale
        'Intercept': feat_intercept_hpc     # Intercetta locale
    })

    pred_rul_hpc_full = hi_hpc.copy()
    
    if mask.any():
        X_lgbm_filtered = X_lgbm_hpc_all.loc[mask]
        
        # Predizione del gap solo nei punti desiderati
        pred_gap_hpc = lgbm_hpc.predict(X_lgbm_filtered)
        
        # Calcoliamo la RUL finale sommando HI ai gap predetti
        hi_values_filtered = hi_hpc[mask]
        pred_rul_hpc_full[mask] = hi_values_filtered + pred_gap_hpc.flatten()

    # Plot
    fig, ax = plt.subplots(figsize=(22, 7))
    fig.suptitle(f'Training: ESN - {esn} (LGBM attivo sul terzo ciclo di manutenzione HPC)', fontsize=16)
    ax.plot(hpc_rul, color='tab:orange', linewidth=2, linestyle='--', label='Real RUL')
    # Linea Predetta/HI (sarà HI fino a Phase 2, poi diventerà la predizione LGBM)
    ax.plot(pred_rul_hpc_full, color='tab:blue', linewidth=2, label='HI + LGBM Prediction')
    # Evidenziatore zona attiva
    ax.fill_between(range(len(mask)), ax.get_ylim()[0], ax.get_ylim()[1], 
                    where=mask, color='green', alpha=0.05, label='LGBM Overwrite Zone')
    ax.set_xlabel('Cycles')
    ax.set_ylabel('Health Index / RUL')
    ax.legend()
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.show()
    

# %%
# Test sul motore di VALIDATION

# Calcolo e standardizzazione degli health index
mask = (res_val["Cumulative_HPC_SVs"] == 2).values

val_hi_hpc = normalize(HIE(coefs_hpc, res_val[degradation_vars])).values.flatten()

# RUL effettiva
val_hpc_rul = hpc_rul_val_scaled.values

# HPC
feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(val_hi_hpc, window_size)
X_lgbm_hpc_val = pd.DataFrame({
    'HI': val_hi_hpc,                # Valore attuale HI
    'Slope': feat_slope_hpc,            # Pendenza locale
    'Intercept': feat_intercept_hpc     # Intercetta locale
})

pred_rul_hpc_full_val = val_hi_hpc.copy()
    
if mask.any():
    X_lgbm_filtered_val = X_lgbm_hpc_val.loc[mask]
        
    # Predizione del gap solo nei punti desiderati
    pred_gap_hpc_val = lgbm_hpc.predict(X_lgbm_filtered_val)
        
    # Calcoliamo la RUL finale sommando HI ai gap predetti
    hi_values_filtered_val = val_hi_hpc[mask]
    pred_rul_hpc_full_val[mask] = hi_values_filtered_val + pred_gap_hpc_val.flatten()

# Plot
fig, ax = plt.subplots(figsize=(22, 7))
fig.suptitle(f'Validation: ESN - {testing_esn} (LGBM attivo sul terzo ciclo di manutenzione HPC)', fontsize=16)
ax.plot(val_hpc_rul, color='tab:orange', linewidth=2, linestyle='--', label='Real RUL')
# Linea Predetta/HI (sarà HI fino a Phase 2, poi diventerà la predizione LGBM)
ax.plot(pred_rul_hpc_full_val, color='tab:blue', linewidth=2, label='HI + LGBM Prediction')
# Evidenziatore zona attiva
ax.fill_between(range(len(mask)), ax.get_ylim()[0], ax.get_ylim()[1], 
                where=mask, color='green', alpha=0.05, label='LGBM Overwrite Zone')
ax.set_xlabel('Cycles')
ax.set_ylabel('Health Index / RUL')
ax.legend()
ax.grid(True, alpha=.3)
plt.tight_layout()
plt.show()




# %% [markdown]
# # WW
