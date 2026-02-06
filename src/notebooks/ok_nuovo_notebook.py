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
#     display_name: phm-america-2025 (3.10.19)
#     language: python
#     name: python3
# ---

# %%
from numpy import sign sdkfjslkfjlaskdjflkajsdfljasdlfkjalsdkfjldskfjlj
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
model_i = 0
testing_esn = 102
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
# degradation_vars = [s for s in u.SENSORS if s not in operating_vars]
degradation_vars = [s for s in SENSORS if s not in operating_vars and s != "Sensed_P25" and s != "Sensed_T5"]
managed_cols = set(degradation_vars) | set(operating_vars)

# %%
# DOWNSAMPLING TRAINING PER AVERE PER TUTTI LO STESSO NUMERO DI DATI
df = load_training()()
df = remove_outliers(df, SENSORS)
df = missingfill(df).dropna()

# Aggregazione dataset di training
other_cols_df = [col for col in df.columns if col not in managed_cols]
agg_logic = {col: 'median' for col in degradation_vars}
agg_logic.update({col: 'median' for col in operating_vars})
agg_logic.update({col: 'first' for col in other_cols_df})
df = df.groupby(['ESN', 'Cycles_Since_New']).agg(agg_logic).reset_index(drop=True)


rows = df.groupby('ESN').size().reset_index(name='rows').copy()
print(rows)

# %%
# DOWNSAMPLING VALIDATION PER AVERE PER TUTTI LO STESSO NUMERO DI DATI
dfv = load_validation()
dfv = remove_outliers(dfv, SENSORS)
dfv = missingfill(dfv).dropna()
# dfv = dfv.ffill()
# dfv = dfv.bfill()

# Aggregazione dataset di validation
other_cols_dfv = [col for col in dfv.columns if col not in managed_cols]
agg_logic_v = {col: 'median' for col in degradation_vars}
agg_logic_v.update({col: 'median' for col in operating_vars})
agg_logic_v.update({col: 'first' for col in other_cols_dfv})
dfv = dfv.groupby(['ESN', 'Cycles']).agg(agg_logic_v).reset_index(drop=True)

resoconto_righe = dfv.groupby('ESN').size().reset_index(name='numero_righe').copy()
print(resoconto_righe)

# %%
# DOWNSAMPLING TESTING PER AVERE PER TUTTI LO STESSO NUMERO DI DATI
dft = load_test()
dft = remove_outliers(dft, SENSORS)
dft = missingfill(dft).dropna()
# dft = dft.ffill()
# dft = dft.bfill()

# Aggregazione dataset di training
other_cols_dft = [col for col in dft.columns if col not in managed_cols]
agg_logic_t = {col: 'median' for col in degradation_vars}
agg_logic_t.update({col: 'median' for col in operating_vars})
agg_logic_t.update({col: 'first' for col in other_cols_dft})
dft = dft.groupby(['ESN', 'Cycles']).agg(agg_logic_t).reset_index(drop=True)

resoconto_righe = dft.groupby('ESN').size().reset_index(name='numero_righe').copy()
print(resoconto_righe)


# %%
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
    for i in range(len(series)):
        if i < window:
            slopes.append(0)
            intercepts.append(0)
        else:
            y = series.iloc[i-window:i].values
            x = np.arange(window)
            # Fit polinomiale di grado 1 (retta) -> ritorna [slope, intercept]
            poly = np.polyfit(x, y, 1)
            slopes.append(poly[0])
            intercepts.append(poly[1])
    return np.array(slopes), np.array(intercepts)


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
  temp[degradation_vars] = temp[degradation_vars].rolling(window=100, min_periods=1).mean()
  temp = temp.dropna()
  cleaned_chunks.append(temp)

res_train = pd.concat(cleaned_chunks).dropna()
train_data_full.update(res_train)
# res_train.index = res_train.groupby("ESN").cumcount()


# integrazione residui sul dataset originale
res_val = remove_outliers(res_val, SENSORS, threshold=0.8)
res_val = res_val.rolling(window=100, min_periods=1).mean()
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

for esn in dft["ESN"].unique():
  mask = dft["ESN"] == esn
  X_test = dft.loc[mask, operating_vars]
  Y_test = dft.loc[mask, degradation_vars]
  # Predict
  Y_pred = model.predict(X_test)
  res_temp = Y_test - Y_pred
  res_temp["ESN"] = esn
  res_list.append(res_temp)

res_test = pd.concat(res_list)


# PULIZIA E ROLLING
cleaned_chunks = []
for esn in dft["ESN"].unique():
  temp = res_test[res_test["ESN"] == esn].copy()
  temp = remove_outliers(temp, SENSORS, threshold=0.8)
  temp[degradation_vars] = temp[degradation_vars].rolling(window=100, min_periods=1).mean()
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
for esn in dft["ESN"].unique():
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
# # Ricerca di a,b,c,d,e,f,g globali combinazione lineare di tutti i sensori

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
hpt_rul_val_scaled = hpt_rul_val / hpt_rul_val.max()
hpc_rul_val_scaled = hpc_rul_val / hpc_rul_val.max()
ww_rul_val_scaled = ww_rul_val / ww_rul_val.max()


# Dati di test
for esn in dft["ESN"].unique():
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
all_coefs_ww = []

for esn in res_train["ESN"].unique():
  result_hpt = differential_evolution(
      objective_deviation,
      bounds=bounds,
      args=(X_train_opt.loc[X_train_opt["ESN"] == esn, degradation_vars], Y_train_hpt.loc[Y_train_hpt["ESN"] == esn, "Cycles_to_HPT_SV"]),
      strategy='best1bin',
      maxiter=400,                # generazioni
      popsize=100,
      workers=-1,
      tol=0,                      # Tolleranza
  )
  all_coefs_hpt.append(result_hpt.x)

  result_hpc = differential_evolution(
      objective_deviation,
      bounds=bounds,
      args=(X_train_opt.loc[X_train_opt["ESN"] == esn, degradation_vars], Y_train_hpc.loc[Y_train_hpc["ESN"] == esn, "Cycles_to_HPC_SV"]),
      strategy='best1bin',
      maxiter=400,                # generazioni
      popsize=100,
      workers=-1,
      tol=0,                      # Tolleranza
  )
  all_coefs_hpc.append(result_hpc.x)

  result_ww = differential_evolution(
      objective_deviation,
      bounds=bounds,
      args=(X_train_opt.loc[X_train_opt["ESN"] == esn, degradation_vars], Y_train_ww.loc[Y_train_ww["ESN"] == esn, "Cycles_to_WW"]),
      strategy='best1bin',
      maxiter=400,                # generazioni
      popsize=100,
      workers=-1,
      tol=0,                      # Tolleranza
  )
  all_coefs_ww.append(result_ww.x)

coefs_hpt = np.mean(all_coefs_hpt, axis=0)
coefs_hpc = np.mean(all_coefs_hpc, axis=0)
coefs_ww  = np.mean(all_coefs_ww, axis=0)

# Stampa dei risultati medi
print("\nCOEFFICIENTI MEDI FINALI (Training Set):")
print(f"HPT: {coefs_hpt}")
print(f"HPC: {coefs_hpc}")
print(f"WW:  {coefs_ww}")

# Store dei coefficienti medi per il testing
# %store coefs_hpt
# %store coefs_hpc
# %store coefs_ww



# %%
coefs_hpt = [280.82217546,
             -285.60620346,
             -2.14567969,
             11.85810011,
             -325.74398183,
             108.41160169]

coefs_hpc = [309.71369749,
             107.41789038,
             -199.50020415,
             585.15358714,
             -665.56105852,
             -37.90033531]

coefs_ww = [-143.42443718,
            53.30818271,
            -63.83384532,
            -240.3277819,
            398.75355915,
            572.93332311]

# %%
# PLOTTING SU DATI DI TRAINING

hpt_limits, hpc_limits, ww_limits = [], [], []
for esn in res_train["ESN"].unique():
  temp = res_train[res_train["ESN"] == esn].copy()
  hi_hpt = HIE(coefs_hpt, temp[degradation_vars])
  hi_hpc = HIE(coefs_hpc, temp[degradation_vars])
  hi_ww = HIE(coefs_ww, temp[degradation_vars])

  hpt_limits.append((hi_hpt.min(), hi_hpt.max()))
  hpc_limits.append((hi_hpc.min(), hi_hpc.max()))
  ww_limits.append((hi_ww.min(), hi_ww.max()))

  # Standardizzazione
  hi_min_hpt, hi_max_hpt = hi_hpt.min(), hi_hpt.max()
  hi_hpt = (hi_hpt - hi_min_hpt) / (hi_max_hpt - hi_min_hpt)
  hi_min_hpc, hi_max_hpc = hi_hpc.min(), hi_hpc.max()
  hi_hpc = (hi_hpc - hi_min_hpc) / (hi_max_hpc - hi_min_hpc)
  hi_min_ww, hi_max_ww = hi_ww.min(), hi_ww.max()
  hi_ww = (hi_ww - hi_min_ww) / (hi_max_ww - hi_min_ww)
  hpt_rul_esn = hpt_rul_train[hpt_rul_train["ESN"] == esn].copy()
  hpc_rul_esn = hpc_rul_train[hpc_rul_train["ESN"] == esn].copy()
  ww_rul_esn  = ww_rul_train[ww_rul_train["ESN"] == esn].copy()

  fig, axs = plt.subplots(1, 3, figsize=(30, 6))
  fig.suptitle(f'Training: ESN - {esn}', fontsize=16)
  axs[0].plot(hi_hpt, color='tab:blue', label='Health Index (HPT)')
  # ax0_rul = axs[0].twinx()
  axs[0].plot(hpt_rul_esn["Cycles_to_HPT_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  axs[1].plot(hi_hpc, color='tab:green', label='Health Index (HPC)')
  # ax1_rul = axs[1].twinx()
  axs[1].plot(hpc_rul_esn["Cycles_to_HPC_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  axs[2].plot(hi_ww, color='tab:green', label='Health Index (HPC)')
  # ax2_rul = axs[2].twinx()
  axs[2].plot(ww_rul_esn["Cycles_to_WW"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  fig.tight_layout()
  fig.show()


def get_avg_limits(limits_list):
    mins = [l[0] for l in limits_list]
    maxs = [l[1] for l in limits_list]
    return np.mean(mins), np.mean(maxs)

# Calcolo limiti medi finali
avg_min_hpt, avg_max_hpt = get_avg_limits(hpt_limits)
avg_min_hpc, avg_max_hpc = get_avg_limits(hpc_limits)
avg_min_ww,  avg_max_ww  = get_avg_limits(ww_limits)

print(f"Limiti Medi HPT: Min={avg_min_hpt:.4f}, Max={avg_max_hpt:.4f}")

# Store per usarli nel notebook del Testing
# %store avg_min_hpt
# %store avg_max_hpt
# %store avg_min_hpc
# %store avg_max_hpc
# %store avg_min_ww
# %store avg_max_ww

# %%
# PLOTTING SU DATI DI VALIDATION

hi_hpt_val = HIE(coefs_hpt, res_val[degradation_vars]).dropna()
hi_hpc_val = HIE(coefs_hpc, res_val[degradation_vars]).dropna()
hi_ww_val = HIE(coefs_ww, res_val[degradation_vars]).dropna()

# Standardizzazione
hi_min_hpt, hi_max_hpt = hi_hpt_val.min(), hi_hpt_val.max()
hi_hpt_val = (hi_hpt_val - hi_min_hpt) / (hi_max_hpt - hi_min_hpt)
hi_min_hpc, hi_max_hpc = hi_hpc_val.min(), hi_hpc_val.max()
hi_hpc_val = (hi_hpc_val - hi_min_hpc) / (hi_max_hpc - hi_min_hpc)
hi_min_ww, hi_max_ww = hi_ww_val.min(), hi_ww_val.max()
hi_ww_val = (hi_ww_val - hi_min_ww) / (hi_max_ww - hi_min_ww)

fig, axs = plt.subplots(1, 3, figsize=(30, 6))
fig.suptitle(f'Validation', fontsize=16)
axs[0].plot(hi_hpt_val, color='tab:blue', label='Health Index (HPT)')
# ax0_rul = axs[0].twinx()
axs[0].plot(hpt_rul_val_scaled, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
axs[1].plot(hi_hpc_val, color='tab:green', label='Health Index (HPC)')
# ax1_rul = axs[1].twinx()
axs[1].plot(hpc_rul_val_scaled, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
axs[2].plot(hi_ww_val, color='tab:green', label='Health Index (HPC)')
# ax2_rul = axs[2].twinx()
axs[2].plot(ww_rul_val_scaled, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
fig.tight_layout()
fig.show()

# %%
# PLOTTING SU DATI DI TESTING

for esn in dft["ESN"].unique():
  temp = res_test[res_test["ESN"] == esn].copy()
  hi_hpt_test = HIE(coefs_hpt, temp[degradation_vars])
  hi_hpc_test = HIE(coefs_hpc, temp[degradation_vars])
  hi_ww_test  = HIE(coefs_ww, temp[degradation_vars])

  # Standardizzazione usando i min e max dei dati di training
  hi_hpt_test = (hi_hpt_test - avg_min_hpt) / (avg_max_hpt - avg_min_hpt)
  hi_hpc_test = (hi_hpc_test - avg_min_hpc) / (avg_max_hpc - avg_min_hpc)
  hi_ww_test = (hi_ww_test - avg_min_ww) / (avg_max_ww - avg_min_ww)

  fig, axs = plt.subplots(1, 3, figsize=(30, 6))
  fig.suptitle(f'Test: ESN - {esn}', fontsize=16)
  axs[0].plot(hi_hpt_test, color='tab:blue', label='Health Index (HPT)')
  axs[1].plot(hi_hpc_test, color='tab:green', label='Health Index (HPC)')
  axs[2].plot(hi_ww_test, color='tab:red', label='Health Index (HPC)')
  fig.tight_layout()
  fig.show()

# %% [markdown]
# ### Classificazione dell'errore con LightGBM per HPC, HPT e WW

# %%
# TRAINING REGRESSORE PER ERRORE GAP
# lightgbm trainato su spezzoni di eventi
# un modello per hpt e hpc e uno per ww???????????

# Regressori per il calcolo dell'errore (tra predizione e RUL reale)
regr_hpc = LinearRegression()
regr_hpt = LinearRegression()
regr_ww = LinearRegression()

# Liste per il training dei regressori
all_train_hi_hpt = []
all_train_hi_hpc = []
all_train_hi_ww = []
all_train_hpt_rul = []
all_train_hpc_rul = []
all_train_ww_rul = []

# HI di training
for esn in res_train["ESN"].unique():
  temp = res_train[res_train["ESN"] == esn].copy()
  hi_hpt = HIE(coefs_hpt, temp[degradation_vars])
  hi_hpc = HIE(coefs_hpc, temp[degradation_vars])
  hi_ww = HIE(coefs_ww, temp[degradation_vars])
  print(f'SHAPE: {hi_hpt.shape}')

  # Standardizzazione
  hi_min_hpt, hi_max_hpt = hi_hpt.min(), hi_hpt.max()
  hi_hpt = (hi_hpt - hi_min_hpt) / (hi_max_hpt - hi_min_hpt)
  hi_min_hpc, hi_max_hpc = hi_hpc.min(), hi_hpc.max()
  hi_hpc = (hi_hpc - hi_min_hpc) / (hi_max_hpc - hi_min_hpc)
  hi_min_ww, hi_max_ww = hi_ww.min(), hi_ww.max()
  hi_ww = (hi_ww - hi_min_ww) / (hi_max_ww - hi_min_ww)
  hpt_rul_esn = hpt_rul_train[hpt_rul_train["ESN"] == esn].copy()
  hpc_rul_esn = hpc_rul_train[hpc_rul_train["ESN"] == esn].copy()
  ww_rul_esn  = ww_rul_train[ww_rul_train["ESN"] == esn].copy()
  rul_hpt_esn = hpt_rul_esn.iloc[:, 1]
  rul_hpc_esn = hpc_rul_esn.iloc[:, 1]
  rul_ww_esn  = ww_rul_esn.iloc[:, 1]

  all_train_hi_hpt.append(hi_hpt)
  all_train_hi_hpc.append(hi_hpc)
  all_train_hi_ww.append(hi_ww)
  all_train_hpt_rul.append(rul_hpt_esn)
  all_train_hpc_rul.append(rul_hpc_esn)
  all_train_ww_rul.append(rul_ww_esn)


# Impiliamo tutto: diventano quattro matrici uniche
X_train_hpt = pd.concat(all_train_hi_hpt, ignore_index=True).dropna().to_frame()
X_train_hpc = pd.concat(all_train_hi_hpc, ignore_index=True).dropna().to_frame()
X_train_ww = pd.concat(all_train_hi_ww, ignore_index=True).dropna().to_frame()

Y_train_hpt = pd.concat(all_train_hpt_rul, ignore_index=True).dropna()
Y_train_hpc = pd.concat(all_train_hpc_rul, ignore_index=True).dropna()
Y_train_ww = pd.concat(all_train_ww_rul, ignore_index=True).dropna()


# TRAINING dei regressori per l'errore
print(f'X TRAIN: {X_train_hpt.shape}')
print(f'Y TRAIN: {Y_train_hpt.shape}')

regr_hpt.fit(X_train_hpt, Y_train_hpt)
regr_hpc.fit(X_train_hpc, Y_train_hpc)
regr_ww.fit(X_train_ww, X_train_ww)


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
# REGRESSIONE LINEARE PER ERRORE GAP

# Sui dati di training
pred_rul_hpt = regr_hpc.predict(X_train_hpt)
pred_rul_hpc_ = regr_hpc.predict(X_train_hpc)
pred_rul_ww = regr_hpc.predict(X_train_ww)

gap_true_hpt = all_train_hpt_rul - pred_rul_hpt
gap_true_hpc_train = all_train_hpc_rul - pred_rul_hpc
gap_true_ww = ww_rul - pred_rul_ww



# %%

pred_rul_hpc = regr_hpc.predict(X_base_hpc)
pred_rul_hpt = regr_hpc.predict(X_base_hpt)
pred_rul_ww = regr_hpc.predict(X_base_ww)

gap_true_hpc = hpc_rul - pred_rul_hpc
gap_true_hpt = hpt_rul - pred_rul_hpt
gap_true_ww = ww_rul - pred_rul_ww

window_size = 300

feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc, window_size)
feat_slope_hpt, feat_intercept_hpt = get_rolling_slope_intercept(hi_hpt, window_size)
feat_slope_ww, feat_intercept_ww = get_rolling_slope_intercept(hi_ww, window_size)

X_lgbm_hpc = pd.DataFrame({
    'HI': hi_hpc.values,                # Valore attuale HI
    'Slope': feat_slope_hpc,            # Pendenza locale
    'Intercept': feat_intercept_hpc     # Intercetta locale
})

X_lgbm_hpt = pd.DataFrame({
    'HI': hi_hpt.values,                # Valore attuale HI
    'Slope': feat_slope_hpt,            # Pendenza locale
    'Intercept': feat_intercept_hpt     # Intercetta locale
})

X_lgbm_ww = pd.DataFrame({
    'HI': hi_ww.values,                 # Valore attuale HI
    'Slope': feat_slope_ww,             # Pendenza locale
    'Intercept': feat_intercept_ww      # Intercetta locale
})


mask = X_lgbm_hpc['Slope'] != 0
X_lgbm_hpc = X_lgbm_hpc[mask]
gap_true_hpc = gap_true_hpc[mask]
base_pred_hpc = pred_rul_hpc[mask]
rul_target_hpc = hpc_rul[mask]

mask = X_lgbm_hpt['Slope'] != 0
X_lgbm_hpt = X_lgbm_hpc[mask]
gap_true_hpt = gap_true_hpt[mask]
base_pred_hpt = pred_rul_hpt[mask]
rul_target_hpt = hpt_rul[mask]

mask = X_lgbm_ww['Slope'] != 0
X_lgbm_ww = X_lgbm_ww[mask]
gap_true_ww = gap_true_ww[mask]
base_pred_ww = pred_rul_ww[mask]
rul_target_ww = ww_rul[mask]


lgbm_hpc = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05)
lgbm_hpc.fit(X_lgbm_hpc, gap_true_hpc)
# %store lgbm_hpc

lgbm_hpt = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05)
lgbm_hpt.fit(X_lgbm_hpt, gap_true_hpt)
# %store lgbm_hpt

lgbm_ww = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05)
lgbm_ww.fit(X_lgbm_ww, gap_true_ww)
# %store lgbm_ww

# %%
# HPC
pred_gap = lgbm_hpc.predict(X_lgbm_hpc)
pred_rul = base_pred_hpc + pred_gap

plt.figure(figsize=(10, 6))
plt.scatter(gap_true_hpc, pred_gap, alpha=0.6, s=15, color='blue', label='Predicted vs True')
min_val = min(gap_true_hpc.min(), pred_gap.min())
max_val = max(gap_true_hpc.max(), pred_gap.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', label='Perfect Prediction')

plt.title(f'Figure 5 Replication: Gap Prediction Accuracy\n(Correlation between Slope/Intercept and Prediction Error)')
plt.xlabel('True Gap Difference')
plt.ylabel('Predicted Gap Difference')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(15, 6))

plt.plot(rul_target_hpc.index, rul_target_hpc, 'k--', linewidth=2, label='True RUL')
plt.plot(rul_target_hpc.index, base_pred_hpc, color='tab:red', alpha=0.6, linestyle='-.', label='Linear Prediction (Base)')
plt.plot(rul_target_hpc.index, pred_rul, color='tab:green', linewidth=2, label='LightGBM Corrected Prediction')

plt.title(f'Impact of LightGBM Correction on RUL Prediction (ESN {testing_esn})')
plt.xlabel('Cycles')
plt.ylabel('RUL')
plt.legend()
plt.grid(True)
plt.show()

# HPT
pred_gap = lgbm_hpt.predict(X_lgbm_hpt)
pred_rul = base_pred_hpt + pred_gap

plt.figure(figsize=(10, 6))
plt.scatter(gap_true_hpt, pred_gap, alpha=0.6, s=15, color='blue', label='Predicted vs True')
min_val = min(gap_true_hpt.min(), pred_gap.min())
max_val = max(gap_true_hpt.max(), pred_gap.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', label='Perfect Prediction')

plt.title(f'Figure 5 Replication: Gap Prediction Accuracy\n(Correlation between Slope/Intercept and Prediction Error)')
plt.xlabel('True Gap Difference')
plt.ylabel('Predicted Gap Difference')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(15, 6))

plt.plot(rul_target_hpt.index, rul_target_hpt, 'k--', linewidth=2, label='True RUL')
plt.plot(rul_target_hpt.index, base_pred_hpt, color='tab:red', alpha=0.6, linestyle='-.', label='Linear Prediction (Base)')
plt.plot(rul_target_hpt.index, pred_rul, color='tab:green', linewidth=2, label='LightGBM Corrected Prediction')

plt.title(f'Impact of LightGBM Correction on RUL Prediction (ESN {testing_esn})')
plt.xlabel('Cycles')
plt.ylabel('RUL')
plt.legend()
plt.grid(True)
plt.show()

# WW
pred_gap = lgbm_ww.predict(X_lgbm_ww)
pred_rul = base_pred_ww + pred_gap

plt.figure(figsize=(10, 6))
plt.scatter(gap_true_ww, pred_gap, alpha=0.6, s=15, color='blue', label='Predicted vs True')
min_val = min(gap_true_ww.min(), pred_gap.min())
max_val = max(gap_true_ww.max(), pred_gap.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', label='Perfect Prediction')

plt.title(f'Figure 5 Replication: Gap Prediction Accuracy\n(Correlation between Slope/Intercept and Prediction Error)')
plt.xlabel('True Gap Difference')
plt.ylabel('Predicted Gap Difference')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(30, 6))

plt.plot(rul_target_ww.index, rul_target_ww, 'k--', linewidth=2, label='True RUL')
plt.plot(rul_target_ww.index, base_pred_ww, color='tab:red', alpha=0.6, linestyle='-.', label='Linear Prediction (Base)')
plt.plot(rul_target_ww.index, pred_rul, color='tab:green', linewidth=2, label='LightGBM Corrected Prediction')

plt.title(f'Impact of LightGBM Correction on RUL Prediction (ESN {testing_esn})')
plt.xlabel('Cycles')
plt.ylabel('RUL')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# # Testing

# %%
def time_weighted_error(y_true, y_pred, alpha=0.02, beta=1):
  """Returns the weighted squared error for an array of predictions."""

  error = y_pred-y_true

  weight = np.where(
  error >= 0,
  2 / (1 + alpha * y_true),
  1 / (1 + alpha * y_true)
  )
  return weight * (error ** 2)*beta

def score_submitted_result(df_true, df_pred):
  '''Calculate the score for a single team's submission'''

  # Extract the targets
  true_WW = df_true.Cycles_to_WW.values
  true_HPC = df_true.Cycles_to_HPC_SV.values
  true_HPT = df_true.Cycles_to_HPT_SV.values

  pred_WW = df_pred.Cycles_to_WW.values
  pred_HPC = df_pred.Cycles_to_HPC_SV.values
  pred_HPT = df_pred.Cycles_to_HPT_SV.values

  # WW score
  alpha = 0.01
  beta = 1/float(max(true_WW))
  score_WW = time_weighted_error(true_WW, pred_WW, alpha, beta)
  # Take the mean of the array
  score_WW = np.mean(score_WW)

  # HPC score
  alpha = 0.01
  beta = 2/float(max(true_HPC))
  score_HPC = time_weighted_error(true_HPC, pred_HPC, alpha, beta)
  # Take the mean of the array
  score_HPC = np.mean(score_HPC)

  # HTC score
  alpha = 0.01
  beta = 2/float(max(true_HPT))
  score_HPT = time_weighted_error(true_HPT, pred_HPT, alpha, beta)
  # Take the mean of the array
  score_HPT = np.mean(score_HPT)

  # Average score
  score = np.mean([score_WW, score_HPC, score_HPT])

  return score


# %%
# coefs_hpt
# coefs_hpc
# coefs_ww

model_i = 0
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
degradation_vars = [s for s in u.SENSORS if s not in operating_vars and s != "Sensed_P25" and s != "Sensed_T5"]

df = u.load_testing(10)
# df = u.load_training()()
print(df.shape)
df = pp.remove_outliers(df, u.SENSORS)
df = pp.missingfill(df).dropna()

df = df.ffill()
df = df.bfill()
# model = models[model_i]['model']

engines = {}
for eng in df["ESN"].unique():
    engines[eng] = {}
    test_data = df[df["ESN"] == eng].reset_index()
    test_data = test_data.groupby(["Cycles"]).median().reset_index()

    # for var in operating_vars + degradation_vars:
    #     test_data[var] = minmax(test_data, var)

    rolling_size = 30
    step = 2
    X_test = test_data[operating_vars]#.rolling(rolling_size, step=step).median().dropna()
    Y_test = test_data[degradation_vars]#.rolling(rolling_size, step=step).median().dropna()
    Y_pred = model.predict(X_test)

    res = Y_test - Y_pred
    res = pp.remove_outliers(res, u.SENSORS, threshold=3)
    test_data[degradation_vars] = res
    res = test_data.dropna()

    engines[eng]["X_test"] = X_test.copy()
    engines[eng]["Y_test"] = Y_test.copy()
    engines[eng]["Y_pred"] = Y_pred.copy()
    engines[eng]["res"] = res.copy()

    hi_hpt = HIE(coefs_hpt, res[degradation_vars])
    hi_hpc = HIE(coefs_hpc, res[degradation_vars])
    hi_ww  = HIE(coefs_ww, res[degradation_vars])

    engines[eng]["hi_hpt"] = hi_hpt
    engines[eng]["hi_hpc"] = hi_hpc
    engines[eng]["hi_ww"] = hi_ww

    window_size = 30
    feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc, window_size)
    X_lgbm_hpc = pd.DataFrame({
        'HI': hi_hpc.values,                # Valore attuale HI
        'Slope': feat_slope_hpc,            # Pendenza locale
        'Intercept': feat_intercept_hpc     # Intercetta locale
    })
    print(X_lgbm_hpc)
    # gap_hpc = test_data["Cycles_to_HPC_SV"].values.reshape(-1,1) - regr_hpc.predict(hi_hpc.values.reshape(-1,1))
    gap_hpc = lgbm_hpc.predict(X_lgbm_hpc)
    print(gap_hpc)
    pred_hpc = regr_hpc.predict(hi_hpc.values.reshape(-1,1))
    rul_hpc = gap_hpc + pred_hpc.flatten()

    feat_slope_hpt, feat_intercept_hpt = get_rolling_slope_intercept(hi_hpt, window_size)
    X_lgbm_hpt = pd.DataFrame({
        'HI': hi_hpt.values,                # Valore attuale HI
        'Slope': feat_slope_hpt,            # Pendenza locale
        'Intercept': feat_intercept_hpt     # Intercetta locale
    })
    # gap_hpt = test_data["Cycles_to_HPT_SV"].values.reshape(-1,1) - regr_hpt.predict(hi_hpt.values.reshape(-1,1))
    gap_hpt = lgbm_hpt.predict(X_lgbm_hpt)
    pred_hpt = regr_hpt.predict(hi_hpt.values.reshape(-1,1))
    rul_hpt = gap_hpt + pred_hpt.flatten()

    feat_slope_ww, feat_intercept_ww = get_rolling_slope_intercept(hi_ww, window_size)
    X_lgbm_ww = pd.DataFrame({
        'HI': hi_ww.values,                 # Valore attuale HI
        'Slope': feat_slope_ww,             # Pendenza locale
        'Intercept': feat_intercept_ww      # Intercetta locale
    })
    # gap_ww = test_data["Cycles_to_WW"].values.reshape(-1,1) - regr_ww.predict(hi_ww.values.reshape(-1,1))
    gap_ww = lgbm_ww.predict(X_lgbm_ww)
    pred_ww = regr_ww.predict(hi_ww.values.reshape(-1,1))
    rul_ww = gap_ww + pred_ww.flatten()

    window_size = 1
    rul_hpc = np.lib.stride_tricks.sliding_window_view(rul_hpc, window_shape=window_size).mean(axis=1)
    rul_hpt = np.lib.stride_tricks.sliding_window_view(rul_hpt, window_shape=window_size).mean(axis=1)
    rul_ww = np.lib.stride_tricks.sliding_window_view(rul_ww, window_shape=20).mean(axis=1)

    plt.subplots(1,3, figsize=(25,8))
    plt.suptitle(f'Engine ESN {eng} - Health Index and RUL Predictions')
    plt.subplot(1,3,1)
    plt.plot(rul_hpc, label='LGBM Corrected Pred HPC', color='orange')
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(rul_hpt, label='LGBM Corrected Pred HPT', color='orange')
    plt.subplot(1,3,3)
    plt.plot(rul_ww, label='LGBM Corrected Pred WW', color='orange')
    plt.show()

# %store engines


# %%
# "DEBUGGING"
# Previsioni solo con regressione lineare
model_i = 0
model = models[model_i]['model']
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
degradation_vars = [s for s in u.SENSORS if s not in operating_vars]


df = u.load_testing()()
# Da rivedere come rimuovere gli outlier
df = pp.remove_outliers(df, u.SENSORS)
df = pp.missingfill(df).dropna()


# Per lavorare con i dati a livello di ciclo
managed_cols = set(degradation_vars) | set(operating_vars)
other_cols = [col for col in df.columns if col not in managed_cols]
agg_logic = {col: 'median' for col in degradation_vars}
agg_logic.update({col: 'median' for col in operating_vars})
agg_logic.update({col: 'first' for col in other_cols})


engines = {}
for eng in df["ESN"].unique():
    engines[eng] = {}
    test_data = df[df["ESN"] == eng].reset_index().copy()
    # test_data = test_data.groupby('Cycles_Since_New', as_index=False).agg(agg_logic).reset_index(drop=True)
    rolling_size = 300
    step = 1
    X_test = test_data[operating_vars].rolling(rolling_size, step=step, min_periods=1).median().dropna()
    Y_test = test_data[degradation_vars].rolling(rolling_size, step=step, min_periods=1).median().dropna()
    Y_pred = model.predict(X_test)
    res = Y_test - Y_pred
    res = pp.remove_outliers(res, u.SENSORS, threshold=3)
    test_data[degradation_vars] = res
    res = test_data.dropna()
    window = 370
    step = 1
    res = res.rolling(window, step).mean()
    res = median_norm(res)
    res = res.dropna()

    hi_hpt = HIE(coefs_hpt, res[degradation_vars])
    hi_hpc = HIE(coefs_hpc, res[degradation_vars])
    hi_ww = HIE(coefs_ww, res[degradation_vars])


    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    axs[0].plot(hi_hpt, color='tab:blue', label='Health Index (HPT)')
    ax0_rul = axs[0].twinx()
    ax0_rul.plot(test_data["Cycles_to_HPT_SV"].reset_index(drop=True), color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
    axs[1].plot(hi_hpc, color='tab:green', label='Health Index (HPC)')
    ax1_rul = axs[1].twinx()
    ax1_rul.plot(test_data["Cycles_to_HPC_SV"].reset_index(drop=True), color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
    axs[2].plot(hi_ww, color='tab:green', label='Health Index (WW)')
    ax2_rul = axs[2].twinx()
    ax2_rul.plot(test_data["Cycles_to_WW"].reset_index(drop=True), color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
    fig.tight_layout()
    fig.show()

# %%
# Prova di Agni

model_i = 0
model = models[model_i]['model']
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
degradation_vars = [s for s in u.SENSORS if s not in operating_vars]


df = u.load_testing()()
# Da rivedere come rimuovere gli outlier
df = pp.remove_outliers(df, u.SENSORS)
df = pp.missingfill(df).dropna()


# Per lavorare con i dati a livello di ciclo
managed_cols = set(degradation_vars) | set(operating_vars)
other_cols = [col for col in df.columns if col not in managed_cols]
agg_logic = {col: 'median' for col in degradation_vars}
agg_logic.update({col: 'median' for col in operating_vars})
agg_logic.update({col: 'first' for col in other_cols})


engines = {}
for eng in df["ESN"].unique():
    engines[eng] = {}
    test_data = df[df["ESN"] == eng].reset_index().copy()
    # test_data = test_data.groupby('Cycles_Since_New', as_index=False).agg(agg_logic).reset_index(drop=True)
    # test_data = test_data.groupby(["ESN", "Snapshot"]).median().reset_index()
    rolling_size = 370
    step = 1
    X_test = test_data[operating_vars] .rolling(rolling_size, step=step, min_periods=1).median().dropna()
    Y_test = test_data[degradation_vars].rolling(rolling_size, step=step, min_periods=1).median().dropna()
    Y_pred = model.predict(X_test)
    res = Y_test - Y_pred
    res = pp.remove_outliers(res, u.SENSORS, threshold=3)
    test_data[degradation_vars] = res
    res = test_data.dropna()
    window = 250
    step = 1
    res = res.rolling(window, step).mean()
    res = median_norm(res)
    res = res.dropna()

    engines[eng]["X_test"] = X_test.copy()
    engines[eng]["Y_test"] = Y_test.copy()
    engines[eng]["Y_pred"] = Y_pred.copy()
    engines[eng]["res"] = res.copy()

    hi_hpt = HIE(coefs_hpt, res[degradation_vars])
    hi_hpc = HIE(coefs_hpc, res[degradation_vars])
    hi_ww  = HIE(coefs_ww, res[degradation_vars])

    engines[eng]["hi_hpt"] = hi_hpt
    engines[eng]["hi_hpc"] = hi_hpc
    engines[eng]["hi_ww"] = hi_ww

    X_base_hpc = hi_hpc.values.reshape(-1,1)
    X_base_hpt = hi_hpt.values.reshape(-1,1)
    X_base_ww = hi_ww.values.reshape(-1,1)

    # gap_true_hpc = test_data["Cycles_to_HPC_SV"].values.reshape(-1,1) - regr_hpc.predict(hi_hpc.values.reshape(-1,1))

    # HPC
    base_pred_hpc = regr_hpc.predict(X_base_hpc)
    window_size = 800
    feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc, window_size)
    X_lgbm_hpc = pd.DataFrame({
        'HI': hi_hpc.values,                # Valore attuale HI
        'Slope': feat_slope_hpc,            # Pendenza locale
        'Intercept': feat_intercept_hpc     # Intercetta locale
    })
    # mask = X_lgbm_hpc['Slope'] != 0
    # X_lgbm_hpc = X_lgbm_hpc[mask]
    # base_pred_hpc = base_pred_hpc[mask]
    gap_pred_hpc = lgbm_hpc.predict(X_lgbm_hpc)
    pred_rul_hpc = base_pred_hpc + gap_pred_hpc
    pred_rul_hpc = pd.Series(pred_rul_hpc).rolling(window=window, min_periods=1).mean()

    # HPT
    base_pred_hpt = regr_hpt.predict(X_base_hpt)
    window_size = 800
    feat_slope_hpt, feat_intercept_hpt = get_rolling_slope_intercept(hi_hpt, window_size)
    X_lgbm_hpt = pd.DataFrame({
        'HI': hi_hpt.values,                # Valore attuale HI
        'Slope': feat_slope_hpt,            # Pendenza locale
        'Intercept': feat_intercept_hpt     # Intercetta locale
    })
    # mask = X_lgbm_hpt['Slope'] != 0
    # X_lgbm_hpt = X_lgbm_hpt[mask]
    # base_pred_hpt = base_pred_hpt[mask]
    gap_pred_hpt = lgbm_hpt.predict(X_lgbm_hpt)
    pred_rul_hpt = base_pred_hpt + gap_pred_hpt
    pred_rul_hpt = pd.Series(pred_rul_hpt).rolling(window=window, min_periods=1).mean()


    # WW
    base_pred_ww = regr_ww.predict(X_base_ww)
    window_size = 800
    feat_slope_ww, feat_intercept_ww = get_rolling_slope_intercept(hi_ww, window_size)
    X_lgbm_ww = pd.DataFrame({
        'HI': hi_ww.values,                # Valore attuale HI
        'Slope': feat_slope_ww,            # Pendenza locale
        'Intercept': feat_intercept_ww     # Intercetta locale
    })
    # mask = X_lgbm_ww['Slope'] != 0
    # X_lgbm_ww = X_lgbm_ww[mask]
    # base_pred_ww = base_pred_ww[mask]
    gap_pred_ww = lgbm_ww.predict(X_lgbm_ww)
    pred_rul_ww = base_pred_ww + gap_pred_ww
    pred_rul_ww = pd.Series(pred_rul_ww).rolling(window=window, min_periods=1).mean()


    plt.subplots(1,3, figsize=(18,6))
    plt.suptitle(f'Engine ESN {eng} - Health Index and RUL Predictions')
    plt.subplot(1,3,1)
    plt.plot(test_data["Cycles_to_HPC_SV"].reset_index(drop=True), label='True RUL HPC')
    plt.plot(pred_rul_hpc, label='LGBM Corrected Pred HPC')
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(test_data["Cycles_to_HPT_SV"].reset_index(drop=True), label='True RUL HPT')
    plt.plot(pred_rul_hpt, label='LGBM Corrected Pred HPT')
    plt.legend()
    plt.subplot(1,3,3)
    plt.plot(test_data["Cycles_to_WW"].reset_index(drop=True), label='True RUL WW')
    plt.plot(pred_rul_ww, label='LGBM Corrected Pred WW')
    plt.legend()
    plt.show()

    engines[eng]["X_lgbm_hpc"] = X_lgbm_hpc
    #engines[eng]["gap_true_hpc"] = gap_true_hpc
    engines[eng]["base_pred_hpc"] = base_pred_hpc
    engines[eng]["X_lgbm_hpt"] = X_lgbm_hpt
    #engines[eng]["gap_true_hpt"] = gap_true_hpt
    engines[eng]["base_pred_hpt"] = base_pred_hpt
    engines[eng]["X_lgbm_ww"] = X_lgbm_ww
    #engines[eng]["gap_true_ww"] = gap_true_ww
    engines[eng]["base_pred_ww"] = base_pred_ww
# %store engines

# %%
# coefs_hpt
# coefs_hpc
# coefs_ww

model_i = 0
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
degradation_vars = [s for s in u.SENSORS if s not in operating_vars]


df = u.load_testing()()
# Da rivedere come rimuovere gli outlier
df = pp.remove_outliers(df, u.SENSORS)
df = pp.missingfill(df).dropna()

# Per lavorare con i dati a livello di ciclo
# managed_cols = set(degradation_vars) | set(operating_vars)
# other_cols = [col for col in df.columns if col not in managed_cols]
# agg_logic = {col: 'median' for col in degradation_vars}
# agg_logic.update({col: 'median' for col in operating_vars})
# agg_logic.update({col: 'first' for col in other_cols})

model = models[model_i]['model']

engines = {}
for eng in df["ESN"].unique():
    engines[eng] = {}
    test_data = df[df["ESN"] == eng].reset_index()
    #test_data = test_data.groupby(["Cycles_Since_New"]).mean().reset_index()

    rolling_size = 25
    step = 1
    X_test = test_data[operating_vars].rolling(rolling_size, step=step, min_periods=1).median().dropna()
    Y_test = test_data[degradation_vars].rolling(rolling_size, step=step, min_periods=1).median().dropna()
    #df = df.groupby(["Snapshot"]).median().reset_index()
    Y_pred = model.predict(X_test)

    res = Y_test - Y_pred
    res = pp.remove_outliers(res, u.SENSORS)
    test_data[degradation_vars] = res
    res = test_data.dropna()

    window = 370
    step = window//5
    res = res.rolling(window, step).median()
    res = median_norm(res)
    res = res.dropna()

    engines[eng]["X_test"] = X_test.copy()
    engines[eng]["Y_test"] = Y_test.copy()
    engines[eng]["Y_pred"] = Y_pred.copy()
    engines[eng]["res"] = res.copy()

    hi_hpt = HIE(coefs_hpt, res[degradation_vars])
    hi_hpc = HIE(coefs_hpc, res[degradation_vars])
    hi_ww  = HIE(coefs_ww, res[degradation_vars])

    engines[eng]["hi_hpt"] = hi_hpt
    engines[eng]["hi_hpc"] = hi_hpc
    engines[eng]["hi_ww"] = hi_ww

    # gap_true_hpc = test_data["Cycles_to_HPC_SV"].values.reshape(-1,1) - regr_hpc.predict(hi_hpc.values.reshape(-1,1))

    feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc, window_size)
    X_lgbm_hpc = pd.DataFrame({
        'HI': hi_hpc.values,                # Valore attuale HI
        'Slope': feat_slope_hpc,            # Pendenza locale
        'Intercept': feat_intercept_hpc     # Intercetta locale
    })

    base_pred_hpc = regr_hpc.predict(hi_hpc.values.reshape(-1,1))
    gap_pred_hpc = lgbm_hpc.predict(X_lgbm_hpc)

    final_rul_hpc = base_pred_hpc.flatten() + gap_pred_hpc

    feat_slope_hpt, feat_intercept_hpt = get_rolling_slope_intercept(hi_hpt, window_size)
    X_lgbm_hpt = pd.DataFrame({
        'HI': hi_hpt.values,                # Valore attuale HI
        'Slope': feat_slope_hpt,            # Pendenza locale
        'Intercept': feat_intercept_hpt     # Intercetta locale
    })
    #gap_true_hpt = test_data["Cycles_to_HPT_SV"].values.reshape(-1,1) - regr_hpt.predict(hi_hpt.values.reshape(-1,1))
    base_pred_hpt = regr_hpt.predict(hi_hpt.values.reshape(-1,1))
    gap_pred_hpt = lgbm_hpt.predict(X_lgbm_hpt)
    final_rul_hpt = base_pred_hpt.flatten() + gap_pred_hpt

    feat_slope_ww, feat_intercept_ww = get_rolling_slope_intercept(hi_ww, window_size)
    X_lgbm_ww = pd.DataFrame({
        'HI': hi_ww.values,                 # Valore attuale HI
        'Slope': feat_slope_ww,             # Pendenza locale
        'Intercept': feat_intercept_ww      # Intercetta locale
    })
    #gap_true_ww = test_data["Cycles_to_WW"].values.reshape(-1,1) - regr_ww.predict(hi_ww.values.reshape(-1,1))
    base_pred_ww = regr_ww.predict(hi_ww.values.reshape(-1,1))
    gap_pred_ww = lgbm_ww.predict(X_lgbm_ww)
    final_rul_ww = base_pred_ww.flatten() + gap_pred_ww

    plt.subplots(1,3, figsize=(18,6))
    plt.suptitle(f'Engine ESN {eng} - Health Index and RUL Predictions')
    plt.subplot(1,3,1)
    plt.plot(test_data["Cycles_to_HPC_SV"].reset_index(drop=True), label='True RUL HPC')
    plt.plot(final_rul_hpc, label='LGBM Corrected Pred HPC')
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(test_data["Cycles_to_HPT_SV"].reset_index(drop=True), label='True RUL HPT')
    plt.plot(final_rul_hpt, label='LGBM Corrected Pred HPT')
    plt.legend()
    plt.subplot(1,3,3)
    plt.plot(test_data["Cycles_to_WW"].reset_index(drop=True), label='True RUL WW')
    plt.plot(final_rul_ww, label='LGBM Corrected Pred WW')
    plt.legend()
    plt.show()

    engines[eng]["X_lgbm_hpc"] = X_lgbm_hpc
    #engines[eng]["gap_true_hpc"] = gap_true_hpc
    engines[eng]["base_pred_hpc"] = base_pred_hpc
    engines[eng]["X_lgbm_hpt"] = X_lgbm_hpt
    #engines[eng]["gap_true_hpt"] = gap_true_hpt
    engines[eng]["base_pred_hpt"] = base_pred_hpt
    engines[eng]["X_lgbm_ww"] = X_lgbm_ww
    #engines[eng]["gap_true_ww"] = gap_true_ww
    engines[eng]["base_pred_ww"] = base_pred_ww
# %store engines

