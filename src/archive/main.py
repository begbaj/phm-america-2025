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

from tools import utils as u, config as cfg, plotting as up, preprocessing as pp, algorithms as alg
import tools


# %% [markdown]
# # Definizione Costanti

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
TESTING_ESN = 102
SENSORS = tools.types.enums.SENSORS
OPERATING_VARS = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
# degradation_vars = [s for s in u.SENSORS if s not in operating_vars] # scommentare se si vuole considerare anche i sensori non in test o valiation
DEGRADATION_VARS = [s for s in SENSORS if s not in OPERATING_VARS and s != "Sensed_P25" and s != "Sensed_T5"]
ALL_VARS = set(DEGRADATION_VARS) | set(OPERATING_VARS)



# 1 REGRESSORE LINEARE
HEALTHY_CYCLES = -1 # -1 per considerare tutti i cicli
DATA_AUGMENTATION = False
DATA_AUGMENTATION_UNITS = 600



# %% [markdown]
# # Caricamento Dati

# %%

# TRAINING
df = u.load_training()
df = pp.common_pipeline(df).dropna()

# # Aggregazione dataset di training
# other_cols_df = [col for col in df.columns if col not in managed_cols]
# agg_logic = {col: 'median' for col in degradation_vars}
# agg_logic.update({col: 'median' for col in operating_vars})
# agg_logic.update({col: 'first' for col in other_cols_df})
# df = df.groupby(['ESN', 'Cycles_Since_New']).agg(agg_logic).reset_index(drop=True)

# VALIDATION 

dfv = u.load_validation(range(0,48))
temp = []
for esn in dfv["ESN"].unique():
    temp.append(dfv[dfv["ESN"] == esn].reset_index())
dfv = pd.concat(temp)
dfv = pp.common_pipeline(dfv, outlier_sensors=SENSORS, missing_align_cols=["Snapshot", "Cycles"]).dropna()

# Aggregazione dataset di validation
# other_cols_dfv = [col for col in dfv.columns if col not in managed_cols]
# agg_logic_v = {col: 'median' for col in degradation_vars}
# agg_logic_v.update({col: 'median' for col in operating_vars})
# agg_logic_v.update({col: 'first' for col in other_cols_dfv})
# dfv = dfv.groupby(['ESN', 'Cycles']).agg(agg_logic_v).reset_index(drop=True)
# rows_val = dfv.groupby('ESN').size().reset_index(name='numero_righe').copy()
# print(rows_val)

# TESTING
# dft = u.load_testing(range(0,52))
dft = u.load_testing(39)
temp = []
for esn in dft["ESN"].unique():
    temp.append(dft[dft["ESN"] == esn].reset_index())
dft = pd.concat(temp)
dft = pp.common_pipeline(dft, outlier_sensors=SENSORS, missing_align_cols=["Snapshot", "Cycles"]).dropna()

# Aggregazione dataset di training
# other_cols_dft = [col for col in dft.columns if col not in managed_cols]
# agg_logic_t = {col: 'median' for col in degradation_vars}
# agg_logic_t.update({col: 'median' for col in operating_vars})
# agg_logic_t.update({col: 'first' for col in other_cols_dft})
# dft = dft.groupby(['ESN', 'Cycles']).agg(agg_logic_t).reset_index(drop=True)
# rows_test = dft.groupby('ESN').size().reset_index(name='numero_righe').copy()

TRAINING_UNIQUE_ESNS = df["ESN"].unique()
TESTING_UNIQUE_ESNS = dft["ESN"].unique()
VALIDATION_UNIQUE_ESNS = dfv["ESN"].unique()

# %store df
# %store dfv
# %store dft

# %%
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error

training_data = df[df["ESN"] != TESTING_ESN].copy()

if HEALTHY_CYCLES > 0:
    training_data = training_data.groupby("ESN").head(HEALTHY_CYCLES).reset_index(drop=True)
else:
    training_data = training_data.reset_index(drop=True)

if DATA_AUGMENTATION:
    # SMOTE
    cols_smote = OPERATING_VARS + DEGRADATION_VARS
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(training_data[cols_smote])
    distances, indices = nbrs.kneighbors(training_data[cols_smote])
    new_indices = np.random.randint(0, len(training_data), DATA_AUGMENTATION_UNITS)
    neighbor_offsets = np.random.randint(1, 5, DATA_AUGMENTATION_UNITS)

    temp_list = []
    for i, base_idx in enumerate(new_indices):
        neighbor_idx = indices[base_idx][neighbor_offsets[i]]
        diff = training_data.iloc[neighbor_idx][cols_smote].values - training_data.iloc[base_idx][cols_smote].values
        new_vals = training_data.iloc[base_idx][cols_smote].values + diff * np.random.rand()
        temp = training_data.iloc[base_idx].copy()
        temp[cols_smote] = new_vals
        temp["ESN"] = f"smote_{i}"
        temp_list.append(temp)

    training_data = pd.concat([training_data, pd.DataFrame(temp_list)], ignore_index=True)
    del temp_list, nbrs, indices, diff 

X_train = training_data[OPERATING_VARS]
Y_train = training_data[DEGRADATION_VARS]
nominal_value_model = LinearRegression()
nominal_value_model.fit(X_train, Y_train)
# %store nominal_value_model

Y_pred_train = nominal_value_model.predict(X_train)
training_data[DEGRADATION_VARS] = Y_train - Y_pred_train

unique_esns = [esn for esn in training_data["ESN"].unique() if "smote" not in str(esn)]
n_plots = len(unique_esns)
n_cols = 2
n_rows = math.ceil(n_plots / n_cols)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
axs = axs.flat # Appiattiamo per iterare facile

for i, esn in enumerate(unique_esns):
    mask = training_data["ESN"] == esn
    axs[i].plot(training_data.loc[mask, DEGRADATION_VARS].values)
    mae = np.mean(np.abs(training_data.loc[mask, DEGRADATION_VARS]))
    axs[i].set_title(f"{esn} - Mean Residual: {mae:.4f}")

for j in range(i + 1, len(axs)): axs[j].axis('off')
fig.tight_layout()
plt.show()

# --- 5. VALIDAZIONE ---
test_data = df[df["ESN"] == TESTING_ESN].copy()
Y_val_pred = nominal_value_model.predict(test_data[OPERATING_VARS])
res_val = test_data[DEGRADATION_VARS] - Y_val_pred # Residui Validazione

plt.figure(figsize=(10,4))
plt.plot(res_val)
plt.title(f"Residuals for Testing ESN: {TESTING_ESN}")
plt.show()

# --- 6. POST-PROCESSING (Outliers & Smoothing) ---
cleaned_list = []
for esn in training_data["ESN"].unique():
    temp = training_data[training_data["ESN"] == esn].copy()
    temp = pp.remove_outliers(temp, SENSORS, threshold=0.8)
    temp[DEGRADATION_VARS] = temp[DEGRADATION_VARS].rolling(window=15, min_periods=1).median()
    cleaned_list.append(temp.dropna())
plot_res = pd.concat(cleaned_list).reset_index(drop=True)
del cleaned_list, temp

unique_esns = training_data["ESN"].unique()
for esn in unique_esns:
    plt.figure(figsize=(15, 7))
    for i, d in enumerate(DEGRADATION_VARS):
        plt.subplot(2, 3, i + 1)
        subset = plot_res[plot_res["ESN"] == esn]
        plt.plot(subset[d], label=d, linewidth=0.8)
        plt.legend(loc='upper right')
        plt.title(f"{esn} - {d} (Cleaned)")
    plt.tight_layout()
    plt.show()

# %%
# AGGREGAZIONE SNAPSHOT
cols = ["Cycles_to_HPT_SV", "Cycles_to_HPC_SV", "Cycles_to_WW"]
other_cols_df = [col for col in training_data.columns if col not in ALL_VARS]
agg_logic = {col: 'median' for col in DEGRADATION_VARS}
agg_logic.update({col: 'first' for col in other_cols_df + cols})

templ = []
for esn in training_data["ESN"].unique():
  temp = training_data[training_data["ESN"] == esn].copy()
  temp[cols] = df[df["ESN"] == esn][cols]
  temp = temp.groupby('Cycles_Since_New').agg(agg_logic).reset_index(drop=True)
  # temp[degradation_vars] = temp[degradation_vars].rolling(window=10, min_periods=1).mean()
  temp["ESN"] = esn
  templ.append(temp)
training_data = pd.concat(templ)

for esn in training_data.ESN.unique():
  fig, axs = plt.subplots(2,len(DEGRADATION_VARS)//2, figsize=(30,12))
  fig.suptitle(esn)
  for i,ax in enumerate(axs.flat):
    ax.plot(training_data[training_data["ESN"] == esn][DEGRADATION_VARS[i]].rolling(window=12, min_periods=1).mean(), label=d, linewidth=3)
    ax.legend()
  fig.show()

# train_data_full = training_data.copy()
# training_data.index = training_data.groupby("ESN").cumcount()

# %%

# integrazione residui sul dataset originale
res_val = pp.remove_outliers(res_val, SENSORS, threshold=0.8)
res_val = res_val.rolling(window=10, min_periods=1).mean()
res_val = res_val.dropna()
test_data.update(res_val)


# %store training_data
# %store res_val

valid_indices_train = training_data.index
hpt_rul_train = training_data.loc[valid_indices_train, ["ESN", "Cycles_to_HPT_SV"]]
hpc_rul_train = training_data.loc[valid_indices_train, ["ESN", "Cycles_to_HPC_SV"]]
ww_rul_train = training_data.loc[valid_indices_train, ["ESN", "Cycles_to_WW"]]
T3_training_data = training_data.loc[valid_indices_train, ["ESN", "Sensed_T3"]]
T45_training_data = training_data.loc[valid_indices_train, ["ESN", "Sensed_T45"]]

valid_indices_val = res_val.index
hpt_rul_val = test_data.loc[valid_indices_val, "Cycles_to_HPT_SV"]
hpc_rul_val = test_data.loc[valid_indices_val, "Cycles_to_HPC_SV"]
ww_rul_val  = test_data.loc[valid_indices_val, "Cycles_to_WW"]
T3_res_val = res_val["Sensed_T3"]
T45_res_val = res_val["Sensed_T45"]


# %store hpt_rul_train
# %store hpc_rul_train
# %store ww_rul_train
# %store T3_training_data
# %store T45_training_data
# %store hpt_rul_val
# %store hpc_rul_val
# %store ww_rul_val
# %store T3_res_val
# %store T45_res_val

# PLOTTING TRAINING
fig, axs = plt.subplots(2,3, figsize=(15,8))
for esn in training_data["ESN"].unique():
  fig.suptitle(f'ESN - {esn}', fontsize=16)
  for i, ax in enumerate(axs.flat):
    if isinstance(ax, plt.Axes):
        degrad = training_data.loc[training_data["ESN"] == esn, DEGRADATION_VARS[i]].reset_index(drop=True)
        ax.plot(degrad, linewidth=0.5, label=esn)
        ax.set_title(DEGRADATION_VARS[i])
        ax.set_ylabel("Residuals")
        ax.set_xlabel(f"{DEGRADATION_VARS[i]}_res")
        ax.legend()
        ax.grid()
  fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.show()

plt.figure()
for esn in training_data["ESN"].unique():
  prova = training_data[training_data["ESN"] ==  esn][DEGRADATION_VARS].reset_index(drop=True).sum(axis=1)
  prova.plot(linewidth=0.3)
plt.legend()

print(training_data.columns)


# %%

# PLOTTING TRAINING
fig, axs = plt.subplots(2,3, figsize=(15,8))
for esn in training_data["ESN"].unique():
  fig.suptitle(f'ESN - {esn}', fontsize=16)
  for i, ax in enumerate(axs.flat):
    if isinstance(ax, plt.Axes):
        degrad = training_data.loc[training_data["ESN"] == esn, DEGRADATION_VARS[i]].reset_index(drop=True)
        ax.plot(degrad, linewidth=0.5, label=esn)
        ax.set_title(DEGRADATION_VARS[i])
        ax.set_ylabel("Residuals")
        ax.set_xlabel(f"{DEGRADATION_VARS[i]}_res")
        ax.legend()
        ax.grid()
  # fig.subplots_adjust(hspace=0.4, wspace=0.4)
  fig.subplots_adjust()
fig.show()

plt.figure()
for esn in training_data["ESN"].unique():
  prova = training_data[training_data["ESN"] ==  esn][DEGRADATION_VARS].reset_index(drop=True).sum(axis=1)
  prova.plot(linewidth=0.3)
plt.legend()

print(training_data.columns)


# %%
# Calcolo dei residui per i dati di test
res_test = pd.DataFrame()
res_list = []

plt.figure(figsize=(16,10))
for i, esn in enumerate(dfv["ESN"].unique()):
  plt.subplot(2,2,i+1)
  mask = dfv["ESN"] == esn
  X_test = dfv.loc[mask, OPERATING_VARS]
  Y_test = dfv.loc[mask, DEGRADATION_VARS]

  # Predict
  Y_pred = nominal_value_model.predict(X_test)

  plt.plot(Y_pred)
  plt.legend(DEGRADATION_VARS)

  temp = Y_test - Y_pred
  temp["ESN"] = esn
  temp = pp.remove_outliers(temp, temp.columns, threshold=0.8)
  temp = temp.ffill()
  temp = temp.bfill()

  print(temp.columns)

  res_list.append(temp)

plt.show()

res_test = pd.concat(res_list)

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
for esn in training_data["ESN"].unique():
  data = training_data.loc[training_data["ESN"] == esn, "Sensed_T45"].reset_index(drop=True)
  plt.plot(data, linewidth=1, label=f"T45 {esn} cumsum")
  plt.legend()

plt.subplot(3,1,2)
for esn in training_data["ESN"].unique():
  data = training_data.loc[training_data["ESN"] == esn, "Sensed_T45"].reset_index(drop=True)
  plt.plot(data.cumsum(), linewidth=1, label=f"T45 {esn} cumsum")
  plt.legend()

plt.subplot(3,1,3)
for esn in training_data["ESN"].unique():
  data = training_data.loc[training_data["ESN"] == esn, "Sensed_T45"].reset_index(drop=True)
  plt.plot(data.diff(), linewidth=1, label=f"T45 {esn} cumsum")
  plt.legend()
plt.show()


#PULIZIA E ROLLING
cleaned_chunks = []
for esn in dfv["ESN"].unique():
  temp = res_test[res_test["ESN"] == esn].copy()
  temp = pp.remove_outliers(temp, SENSORS, threshold=0.1)
  temp[DEGRADATION_VARS] = temp[DEGRADATION_VARS].rolling(window=50, min_periods=1).mean()
  cleaned_chunks.append(temp)

res_test = pd.concat(cleaned_chunks).dropna()
res_test.index = res_test.groupby("ESN").cumcount()

T3_res_test = res_test[["ESN", "Sensed_T3"]].copy()
T45_res_test = res_test[["ESN", "Sensed_T45"]].copy()

# Store dei risultati
# %store res_test
# %store T3_res_test
# %store T45_res_test

# %%
# PLOTTING TRAINING
fig, axs = plt.subplots(2,3, figsize=(15,8))
for esn in training_data["ESN"].unique():
  fig.suptitle(f'ESN - {esn}', fontsize=16)
  for i, ax in enumerate(axs.flat):
    if isinstance(ax, plt.Axes):
        degrad = training_data.loc[training_data["ESN"] == esn, DEGRADATION_VARS[i]].reset_index(drop=True)
        ax.plot(degrad, linewidth=0.5, label=esn)
        ax.set_title(DEGRADATION_VARS[i])
        ax.set_ylabel("Residuals")
        ax.set_xlabel(f"{DEGRADATION_VARS[i]}_res")
        ax.legend()
        ax.grid()
  fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.show()


# %%
# PLOTTING VALIDATION
fig, axs = plt.subplots(2,3, figsize=(15,8))
fig.suptitle(f'ESN - {TESTING_ESN}', fontsize=16)
for i, ax in enumerate(axs.flat):
    if isinstance(ax, plt.Axes):
        ax.plot(res_val.loc[:, DEGRADATION_VARS[i]].rolling(window=95, min_periods=1).mean(), linewidth=1)
        ax.set_title(DEGRADATION_VARS[i])
        ax.set_ylabel("Residuals")
        ax.set_xlabel(f"{DEGRADATION_VARS[i]}_res")
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
        ax.plot(res_test.loc[res_test["ESN"] == esn, DEGRADATION_VARS[i]], linewidth=1)
        ax.set_title(DEGRADATION_VARS[i])
        ax.set_ylabel("Residuals")
        ax.set_xlabel(f"{DEGRADATION_VARS[i]}_res")
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
dict_mins = {var: [] for var in DEGRADATION_VARS}
dict_maxs = {var: [] for var in DEGRADATION_VARS}
for esn in training_data["ESN"].unique():
    mask = training_data["ESN"] == esn
    # mask_hpt = hpt_rul_train["ESN"] == esn
    # hpt_rul_train.loc[mask, "Cycles_to_HPT_SV"] = minmax(hpt_rul_train[mask_hpt], "Cycles_to_HPT_SV")
    # mask_hpc = hpc_rul_train["ESN"] == esn
    # hpc_rul_train.loc[mask, "Cycles_to_HPC_SV"] = minmax(hpc_rul_train[mask_hpc], "Cycles_to_HPC_SV")
    # mask_ww = ww_rul_train["ESN"] == esn
    # ww_rul_train.loc[mask, "Cycles_to_WW"] = minmax(ww_rul_train[mask_ww], "Cycles_to_WW")
    hpt_rul = training_data[mask]["Cycles_to_HPT_SV"]
    hpc_rul = training_data[mask]["Cycles_to_HPC_SV"]
    ww_rul = training_data[mask]["Cycles_to_WW"]
    for var in DEGRADATION_VARS:
        dict_mins[var].append(training_data.loc[mask, var].min())
        dict_maxs[var].append(training_data.loc[mask, var].max())
        training_data.loc[mask, var] = alg.minmax(training_data.loc[mask], var)
avg_mins_per_var = {var: np.mean(lst) for var, lst in dict_mins.items()}
avg_maxs_per_var = {var: np.mean(lst) for var, lst in dict_maxs.items()}


# Dati di validation
for var in DEGRADATION_VARS:
    res_val.loc[var] = alg.minmax(res_val, var)
hpt_rul_val_scaled = alg.normalize(hpt_rul_val)
hpc_rul_val_scaled = alg.normalize(hpc_rul_val)
ww_rul_val_scaled = alg.normalize(ww_rul_val)


# Dati di test
for esn in dfv["ESN"].unique():
    mask = res_test["ESN"] == esn
    for var in DEGRADATION_VARS:
        # Usa i limiti medi del training
        m = avg_mins_per_var[var]
        M = avg_maxs_per_var[var]
        res_test.loc[mask, var] = (res_test.loc[mask, var] - m) / (M - m)



# %store training_data
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

for esn in training_data["ESN"].unique():
    res_esn = training_data.loc[training_data["ESN"] == esn, ["ESN"] + DEGRADATION_VARS]

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
    (-1, 1),  # a
    (-1, 1),  # b
    (-1, 1),  # c
    (-1, 1),  # d
    (-1, 1),  # e
    (-1, 1),  # f
]

all_coefs_hpt = []
all_coefs_hpc = []
# all_coefs_ww = []

for esn in training_data["ESN"].unique():
  result_hpt = differential_evolution(
      alg.objective_deviation,
      bounds=bounds,
      args=(X_train_opt.loc[X_train_opt["ESN"] == esn, DEGRADATION_VARS], Y_train_hpt.loc[Y_train_hpt["ESN"] == esn, "Cycles_to_HPT_SV"]),
      strategy='best1bin',
      maxiter=600,                # generazioni
      popsize=30,
      workers=-1,
      tol=0.001,                      # Tolleranza
  )
  all_coefs_hpt.append(result_hpt.x)

  result_hpc = differential_evolution(
      alg.objective_deviation,
      bounds=bounds,
      args=(X_train_opt.loc[X_train_opt["ESN"] == esn, DEGRADATION_VARS], Y_train_hpc.loc[Y_train_hpc["ESN"] == esn, "Cycles_to_HPC_SV"]),
      strategy='best1bin',
      maxiter=600,                # generazioni
      popsize=30,
      workers=-1,
      tol=0.001,                      # Tolleranza
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
coefs_hpt = [-3.25051845,-1.30326949,-2.79572657,-1.6546684, 1.28047409,-3.06583629]
coefs_hpc = [6.56571436,-4.15254605,-3.98586221, 2.5092515, -3.09432992,-3.70538354]

# coefs_hpt = [276.19439386,
#               -343.90020315,
#               781.45341037,
#               -593.41135514,
#               -190.36846837,
#               118.5466599]

# coefs_hpc = [354.11354513,
#              295.34981181,
#              -386.49672567,
#              790.99844931,
#              -679.69393928,
#              487.61887153]

# %store coefs_hpt
# %store coefs_hpc

# %%
# PLOTTING SU DATI DI TRAINING

hpt_limits, hpc_limits, ww_limits = [], [], []
for esn in training_data["ESN"].unique():
  temp = training_data[training_data["ESN"] == esn].copy()
  
  # Calcolo e standardizzazione degli health index
  hi_hpt = alg.normalize(alg.HIE(coefs_hpt, temp[DEGRADATION_VARS]))
  hi_hpc = alg.normalize(alg.HIE(coefs_hpc, temp[DEGRADATION_VARS]))
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
  axs[0].plot(temp["Cycles_to_HPT_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  axs[0].legend()

  axs[1].plot(hi_hpc, color='tab:green', label='Health Index (HPC)')
  axs[1].plot(temp["Cycles_to_HPC_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  axs[1].legend()
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

hi_hpt_val = alg.normalize(alg.HIE(coefs_hpt, res_val[DEGRADATION_VARS]).dropna())
hi_hpc_val = alg.normalize(alg.HIE(coefs_hpc, res_val[DEGRADATION_VARS]).dropna())
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
# PLOTTING SU DATI DI TEST

for esn in dft["ESN"].unique():
  temp = res_test[res_test["ESN"] == esn].copy()

  # Calcolo e standardizzazione degli health index
  hi_hpt_test = alg.normalize(alg.HIE(coefs_hpt, temp[DEGRADATION_VARS]).dropna())
  hi_hpc_test = alg.normalize(alg.HIE(coefs_hpc, temp[DEGRADATION_VARS]).dropna())
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
# NON USARE QUESTO

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

hpc_events = get_events(res_train, "Cycles_to_HPC_SV")
hpt_events = get_events(res_train, "Cycles_to_HPT_SV")
ww_events  = get_events(res_train,  "Cycles_to_WW")

# ---------------------------------------------------------
# 2. Logica di Reset dei Contatori
# ---------------------------------------------------------

# Inizializziamo le nuove colonne a 0 (o NaN se preferisci vederlo vuoto)
res_train["Cycle_count_HPC"] = 0
res_train["Cycle_count_HPT"] = 0
res_train["Cycle_count_WW"]  = 0

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
            print(col_name)
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
  hi_hpt = normalize(HIE(coefs_hpt, temp[DEGRADATION_VARS]).dropna())
  hi_hpc = normalize(HIE(coefs_hpc, temp[DEGRADATION_VARS]).dropna())
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


test_hi_hpt = normalize_hi(HIE(coefs_hpt, res_test[res_test["ESN"] == 106][DEGRADATION_VARS]))
test_hi_hpc = normalize_hi(HIE(coefs_hpc, res_test[res_test["ESN"] == 106][DEGRADATION_VARS]))
test_hi_ww = normalize_hi(HIE(coefs_ww, res_test[res_test["ESN"] == 106][DEGRADATION_VARS]))

val_hi_hpt = normalize_hi(HIE(coefs_hpt, res_val[DEGRADATION_VARS])).dropna()
val_hi_hpc = normalize_hi(HIE(coefs_hpc, res_val[DEGRADATION_VARS])).dropna()
val_hi_ww = normalize_hi(HIE(coefs_ww, res_val[DEGRADATION_VARS])).dropna()

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
    hi_hpt = normalize(HIE(coefs_hpt, temp[DEGRADATION_VARS])).dropna()
    hi_hpc = normalize(HIE(coefs_hpc, temp[DEGRADATION_VARS])).dropna()
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
    hi_hpt = normalize(HIE(coefs_hpt, temp[DEGRADATION_VARS])).dropna()
    hi_hpc = normalize(HIE(coefs_hpc, temp[DEGRADATION_VARS])).dropna()
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
val_hi_hpt = normalize(HIE(coefs_hpt, res_val[DEGRADATION_VARS])).dropna()
val_hi_hpc = normalize(HIE(coefs_hpc, res_val[DEGRADATION_VARS])).dropna()
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
fig.suptitle(f'Validation: ESN - {TESTING_ESN}', fontsize=16)
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
# Test con slope dell'hpc_hi

X_ww_list, y_ww_list = [], []

for esn in res_train["ESN"].unique():
    temp = res_train[res_train["ESN"] == esn].copy()
    hpc_hi = normalize(HIE(coefs_hpc, temp[DEGRADATION_VARS]))
    ww_rul = ww_rul_train.loc[ww_rul_train["ESN"] == esn, "Cycles_to_WW"].copy()
    feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hpc_hi, window_size)
    feat_slope_ps3, feat_intercept_ps3 = get_rolling_slope_intercept(temp["Sensed_Ps3"], window_size)
    # Accumulo dati HPC
    X_ww_list.append(pd.DataFrame({'HI': hpc_hi.values.flatten(), 'Slope_hi': feat_slope_hpc, 'Intercept': feat_intercept_hpc, 'Slope_Ps3': feat_slope_ps3}))
    y_ww_list.append(ww_rul)

X_train_ww = pd.concat(X_ww_list, ignore_index=True)
y_train_ww = np.concatenate(y_ww_list)

print("Training LGBM HPT...")
lgbm_ww = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03)
lgbm_ww.fit(X_train_ww, y_train_ww)  

# %store lgbm_ww


# %%
# Test sui dati di training

for esn in res_train["ESN"].unique():
    temp = res_train[res_train["ESN"] == esn].reset_index().copy()

    # Calcolo e standardizzazione degli health index
    hi_hpc = normalize(HIE(coefs_hpc, temp[DEGRADATION_VARS])).dropna()
    print(f'SHAPE: {hi_hpt.shape}')

    # RUL effettiva
    ww_rul = ww_rul_train.loc[ww_rul_train["ESN"] == esn, "Cycles_to_WW"].values.copy()

    # WW
    feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc, window_size)
    feat_slope_ps3, feat_intercept_ps3 = get_rolling_slope_intercept(temp["Sensed_Ps3"], window_size)
    X_lgbm_hpc = pd.DataFrame({
        'HI': hi_hpc.values.flatten(),       # Valore attuale HI
        'Slope_hi': feat_slope_hpc,          # Pendenza dell'hi
        'Intercept': feat_intercept_hpc,     # Intercetta locale
        'Slope_Ps3': feat_slope_ps3          # Pendenza della Ps3
    })
    pred_ww = lgbm_ww.predict(X_lgbm_hpc)

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle(f'Training Performance: ESN {esn}', fontsize=16)
    ax.plot(ww_rul, color='tab:orange', linewidth=2, linestyle='--', label='Real RUL (WW)')
    ax.plot(pred_ww, color='tab:blue', alpha=0.8, label='Predicted RUL (LGBM)')
    ax.set_xlabel('Cicli (dopo finestra di slope)')
    ax.set_ylabel('RUL (Cicli)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
# Test sui dati di validation

# Calcolo e standardizzazione degli health index
hi_hpc = normalize(HIE(coefs_hpc, res_val[DEGRADATION_VARS])).dropna()

feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc, window_size)
feat_slope_ps3, feat_intercept_ps3 = get_rolling_slope_intercept(res_val["Sensed_Ps3"].dropna(), window_size)
print(feat_slope_hpc.shape)
print(feat_slope_ps3.shape)
X_lgbm_hpc = pd.DataFrame({
    'HI': hi_hpc.values.flatten(),       # Valore attuale HI
    'Slope_hi': feat_slope_hpc,          # Pendenza dell'hi
    'Intercept': feat_intercept_hpc,     # Intercetta locale
    'Slope_Ps3': feat_slope_ps3          # Pendenza della Ps3
})
pred_ww_raw = lgbm_ww.predict(X_lgbm_hpc)

# Plot
fig, ax = plt.subplots(figsize=(15, 6))
fig.suptitle(f'Validation ESN - {TESTING_ESN}', fontsize=16)
ax.plot(ww_rul_val_scaled, color='tab:orange', linewidth=2, linestyle='--', label='Real RUL (WW)')
ax.plot(pred_ww_raw, color='tab:blue', alpha=0.8, label='Predicted RUL (LGBM)')
ax.set_xlabel('Cicli (dopo finestra di slope)')
ax.set_ylabel('RUL (Cicli)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# # WW test

# %%
def get_slope(y):
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return slope

window_slope = 100
res_train['T45_Slope'] = res_train.groupby('ESN')['Sensed_T45'].transform(
    lambda x: x.rolling(window=window_slope).apply(get_slope)
)

res_train['T45_Diff_Slope'] = res_train.groupby("ESN")["T45_Slope"].diff()

# Calcola il valore di T45 all'inizio di ogni ciclo di lavaggio (dove Cycle_count_WW è 0)
res_train['T45_Reset'] = res_train.loc[res_train['Cycle_count_WW'] == 0, 'Sensed_T45']
# Riempi i valori per i cicli successivi (forward fill)
res_train['T45_Reset'] = res_train.groupby('ESN')['T45_Reset'].ffill()
# Calcola l'incremento di temperatura attuale rispetto all'inizio del ciclo
res_train['T45_Increment'] = res_train['Sensed_T45'] - res_train['T45_Reset']

print(res_train)
plt.figure()
# plt.plot(res_train[res_train["ESN"] == 101]["Sensed_T45"])
# plt.plot(res_train[res_train["ESN"] == 101]["T45_Slope"])
# plt.plot(res_train[res_train["ESN"] == 101]["T45_Diff_Slope"])
# plt.scatter(res_train[(res_train["ESN"] == 101) & (res_train["Cycle_count_WW"] == 0)].index.values, 0, 0.001)
plt.plot(res_train["T45_Increment"])
plt.show()


