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
# TUTTI GLI IMPORT
from scipy import stats
from scipy.optimize import differential_evolution
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# %load_ext autoreload
# %autoreload 2

from tools import utils as u, config as cfg, plotting as up, preprocessing as pp
import tools

# %store -r


# %% [markdown]
# # Configurazione

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

SENSORS = tools.types.enums.SENSORS

OPERATING_VARS = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
# degradation_vars = [s for s in u.SENSORS if s not in operating_vars] # scommentare se si vuole considerare anche i sensori non in test o valiation
DEGRAD_VARS = [s for s in SENSORS if s not in OPERATING_VARS and s != "Sensed_P25" and s != "Sensed_T5"]
ALL_VARS = OPERATING_VARS + DEGRAD_VARS

## FUNZIONI

## MODEL TRAINING
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

## Health Index
def s_pred(s_o, model):
    return model.predict(s_o)

def residual(s_d, s_o, model):
    return s_d - s_pred(s_o, model)


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

## TWE

def wind(y_p, y, a):
    diff = y - y_p
    num = np.where(diff >= 0, 2.0, 1.0)
    if isinstance(y_p, pd.DataFrame) or isinstance(y_p, pd.Series):
        y_p = y_p.values
    return num / (1 + a * y_p)


def time_weighted_error(y_true, y_pred, alpha=0.02, beta=1):
  """Returns the weighted squared error for an array of predictions."""

  error = y_pred-y_true

  weight = np.where(
  error >= 0,
  2 / (1 + alpha * y_true),
  1 / (1 + alpha * y_true)
  )
  return weight * (error ** 2)*beta

def TWE(y_p, y, a=0.02, b=1):
    # if isinstance(y_p, pd.DataFrame): y_p = y_p.values
    # weight = wind(y_p, y, a)
    # squared_error = (y - y_p) ** 2
    # return weight * squared_error * b
    return time_weighted_error(y_p, y, a, b)

## TARGET FUNCTIONS
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


# %% [markdown]
# ## Caricamento dati

# %%
# TRAINING
df = u.load_training()
df = pp.remove_outliers(df, SENSORS)
df = pp.missingfill(df).dropna()

# # Aggregazione dataset di training
# other_cols_df = [col for col in df.columns if col not in managed_cols]
# agg_logic = {col: 'median' for col in degradation_vars}
# agg_logic.update({col: 'median' for col in operating_vars})
# agg_logic.update({col: 'first' for col in other_cols_df})
# df = df.groupby(['ESN', 'Cycles_Since_New']).agg(agg_logic).reset_index(drop=True)
# rows = df.groupby('ESN').size().reset_index(name='rows').copy()
# print(rows)

# VALIDATION
dfv = u.load_validation(range(0,48))
dfv = pp.remove_outliers(dfv, SENSORS)
dfv = pp.missingfill(dfv, align_cols=["Snapshot", "Cycles"]).dropna()

# Aggregazione dataset di validation
# other_cols_dfv = [col for col in dfv.columns if col not in managed_cols]
# agg_logic_v = {col: 'median' for col in degradation_vars}
# agg_logic_v.update({col: 'median' for col in operating_vars})
# agg_logic_v.update({col: 'first' for col in other_cols_dfv})
# dfv = dfv.groupby(['ESN', 'Cycles']).agg(agg_logic_v).reset_index(drop=True)
# rows_val = dfv.groupby('ESN').size().reset_index(name='numero_righe').copy()
# print(rows_val)

# TESTING
dft = u.load_testing(range(0,52))
dft = pp.remove_outliers(dft, SENSORS)
dft = pp.missingfill(dft, align_cols=["Snapshot", "Cycles"]).dropna()
# Aggregazione dataset di training
# other_cols_dft = [col for col in dft.columns if col not in managed_cols]
# agg_logic_t = {col: 'median' for col in degradation_vars}
# agg_logic_t.update({col: 'median' for col in operating_vars})
# agg_logic_t.update({col: 'first' for col in other_cols_dft})
# dft = dft.groupby(['ESN', 'Cycles']).agg(agg_logic_t).reset_index(drop=True)
# rows_test = dft.groupby('ESN').size().reset_index(name='numero_righe').copy()

# %store df
# %store dfv
# %store dft

# %% [markdown]
# # Regressione Lineare 1 - modello andamento nominale

# %%
from sklearn.neighbors import NearestNeighbors

# Preparazione dati training
cycles_healthy = 3

augmented_data = False
augmented_count = 100
SMOTE = False
SMOOTH = False
SMOOTH_WINDOW = [8,8]
TESTING_ESN = 103
INCLUDE_TEST = False

if INCLUDE_TEST:
    train_data = df.copy().reset_index(drop=True)
else:
    train_data = df[df["ESN"] != TESTING_ESN].copy().reset_index(drop=True)

train_data_testing = df[df["ESN"] != TESTING_ESN].copy().reset_index(drop=True)
test_data = df[df["ESN"] == TESTING_ESN].copy().reset_index(drop=True)

if SMOOTH:
    train_data[ALL_VARS] = train_data.groupby(["ESN", "Snapshot"])[ALL_VARS].transform(
        lambda x: x.rolling(window=SMOOTH_WINDOW[0], min_periods=SMOOTH_WINDOW[1]).mean()
    )
    train_data = train_data.dropna().reset_index(drop=True)

if cycles_healthy > 0:
    train_data = train_data.groupby("ESN").head(cycles_healthy*8).reset_index(drop=True).copy()
else:
    train_data = train_data.sort_values(["ESN", "Cycles_Since_New", "Snapshot"])

if SMOTE:
    newdf = []
    for esn in train_data["ESN"].unique():
        cur = train_data[train_data["ESN"] == esn]
        nbrs = NearestNeighbors(n_neighbors=min(5, len(cur))).fit(cur[ALL_VARS])
        dist, idx = nbrs.kneighbors(cur[ALL_VARS])
        for i in range(augmented_count):
            neighbor_offsets = np.random.randint(1, idx.shape[1], size=len(cur))
            neighbor_indices = idx[np.arange(len(cur)), neighbor_offsets]
            # SMOTE 
            diff = train_data.iloc[neighbor_indices][ALL_VARS].values - cur[ALL_VARS].values
            new_vals = cur[ALL_VARS].values + diff * np.random.rand(len(cur), 1)
            
            aug_df = cur.copy()
            aug_df[ALL_VARS] = new_vals
            aug_df['ESN'] = f"aug_{i}_{esn}"
            
            newdf.append(aug_df)
    print("total data", len(newdf))

    train_data = pd.concat([train_data] + newdf, ignore_index=True)
    del newdf, aug_df, nbrs, idx, diff, new_vals

# if augmented_data:
    # new_synthetic_units = []
    # for esn in train_data["ESN"].unique():
    #     for i in range(30):
    #         aug_df = train_data[train_data["ESN"] == esn].copy()
    #         noise = np.random.normal(loc=0, scale=2, size=aug_df[ALL_VARS].shape)
    #         aug_df[ALL_VARS] += noise
    #         aug_df['ESN'] = f"aug_{i}_{esn}" 
    #         new_synthetic_units.append(aug_df)
    # train_data = pd.concat([train_data] + newdf, ignore_index=True)
    # del newdf, aug_df, noise

# %%
# training regressore lineare
import time
ENSAMBLE = True
SEPARATE_MODELS = True
train_data = train_data.sort_values(["ESN", "Cycles_Since_New", "Snapshot"])
if ENSAMBLE:
  # con ESNAMBLE creiamo un modello per ogni ESN
  # e poi, con la funziona residual, otterremo la media
  # di tutti i valori predetti
  models = {}
  for esn in train_data["ESN"].unique():
      print(f"esn {esn}: ", end="")
      mask = train_data["ESN"] == esn
      X_train = train_data.loc[mask, OPERATING_VARS]
      X_train = X_train.reset_index(drop=True)
      Y_train = train_data.loc[mask, DEGRAD_VARS]
      Y_train = Y_train.reset_index(drop=True)
      start = time.time()
      model = train_model(X_train, Y_train)
      end = time.time()
      print(end - start)
      models[str(esn)] = model
      del model
  print(models)
  # %store models

else:
  # senza, invece, trainiamo un unico modello su tutti 
  # i valori del dataset
  models = {}
  if not INCLUDE_TEST:
    mask = train_data["ESN"] != TESTING_ESN
    X_train = train_data.loc[mask, OPERATING_VARS]
    Y_train = train_data.loc[mask, DEGRAD_VARS]
  else:
    X_train = train_data[OPERATING_VARS]
    Y_train = train_data[DEGRAD_VARS]
  model = train_model(X_train, Y_train)
  models["all"] = model

def ensamble(data):
    predictions = []
    for model in models.values():
      predictions.append(model.predict(data))
    return np.mean(predictions, axis=0) # mean lo applichiamo indistintamente, nel secondo caso la media è pari al valore stesso

def residual_regressor(data, esn=None):
  if not SEPARATE_MODELS:
    return ensamble(data)
  else:
    if esn:
      try:
        return models[str(esn)].predict(data)
      except KeyError:
        print("Non ci sono modelli addestrati per questo motore, usiamo ensamble generico")
        return ensamble(data)
    else:
      print("Dagli un ESN porcone")





# %%
# calcolo residui
def residuals(dfo):
  res_list = []
  df = dfo.copy()
  for esn in df["ESN"].unique():
    res_temp = None
    mask = df["ESN"] == esn
    X_train = df.loc[mask, OPERATING_VARS]
    Y_train = df.loc[mask, DEGRAD_VARS]
    print(X_train.shape)
    print(Y_train.shape)
    if SEPARATE_MODELS:
      Y_pred = residual_regressor(X_train, esn)
    else:
      Y_pred = residual_regressor(X_train)
    if Y_pred is None:
      return
    print(Y_pred.shape)
    print(f'Nulli: {np.isnan(Y_pred).sum()}')
    twe = np.mean(TWE(Y_pred, Y_train))
    res_temp = Y_train - Y_pred
    res_temp = pp.remove_outliers(res_temp, threshold=3)
    res_temp = res_temp.ffill()
    res_temp = res_temp.bfill()
    res_temp["ESN"] = esn

    try:
      res_temp["Cycles"] = df.loc[mask, "Cycles_Since_New"]
    except:
      res_temp["Cycles"] = df.loc[mask, "Cycles"]

    res_list.append(res_temp)
    print(f"TWE for {esn}: {twe}")
  return pd.concat(res_list)

print("TRAINING DATASET")
res_train = residuals(train_data)

res_all_train = residuals(train_data_testing)
print(" testing esn")
res_test = residuals(test_data)
print("DFT")
res_dft = residuals(dft)
print("DFV")
res_dfv = residuals(dfv)

# %%
# PLOTTING TRAINING

GROUP_CYCLES = True
REMOVE_OUTLIERS = True
OUTLIERS_THRESHOLD = 3

def plot(data, window, min):
    fig, axs = plt.subplots(2, 3, figsize=(15,8))
    axs = axs.flatten()
    for esn in data["ESN"].unique():
        if "aug" in str(esn):
            continue
        res_temp = data[data["ESN"] == esn].copy()
        if GROUP_CYCLES:
            res_temp = res_temp.groupby("Cycles").mean()
        if REMOVE_OUTLIERS:
            res_temp = pp.remove_outliers(res_temp, threshold=OUTLIERS_THRESHOLD, method="iqr")
            res_temp = res_temp.ffill()
            res_temp = res_temp.bfill()
        for i, ax in enumerate(axs):
            if i < len(DEGRAD_VARS):
                d_var = DEGRAD_VARS[i]
                t = res_temp[d_var]
                degrad = t.rolling(window=window, min_periods=min).mean().dropna()
                ax.plot(degrad.index, degrad.values, linewidth=1.5, alpha=0.7, label=str(esn))
                ax.set_title(d_var)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("Cycles") # Aggiungi etichetta asse X
                
    axs[0].legend(fontsize='small', loc='upper right')
    plt.tight_layout()
    plt.show()

plot(res_train, 1, 1)
plot(res_all_train, 10, 1)
plot(res_test, 3, 1)
# plot(residuals(dfv), 10, 1)
# plot(residuals(dft), 10, 1)


# %% [markdown]
# # HPT e HPC
# ## Ricerca di a globale per la combinazione lineare di T3 e T45

# %%
USE_ALL_VARS = False
MAXITER = 500
POPSIZE = 100
TOL = 0.00001

USE_ONLY_TRAIN = False # usare solo i motori indicati come train? oppure anche il testing_esn?
USE_CLEAN_DATA = True  # dati preprocessati

OUTLIERS_THRESHOLD = 3
SEPARATE_COEFS = True


target = None
target_vars = []

if not USE_ALL_VARS:
    def _target_1(a, vars, RUL):
        vars.dropna()
        hi = HI(vars["Sensed_T3"], vars["Sensed_T45"], a)
        if max(hi) == min(hi):
            return 1.0
        RUL = RUL.dropna()
        corr = stats.pearsonr(RUL,hi)
        return -corr[0]
    target = _target_1
    target_vars = ["Sensed_T45", "Sensed_T3"]
    bounds = [(-1000, 1000)]
else:
    def _target_2(params, vars, RUL):
        hi = HIE(params, vars)
        hi_min, hi_max = hi.min(), hi.max()
        if hi_max == hi_min:
            return 1.0
        hi_norm = (hi - hi_min) / (hi_max - hi_min)
        mse = np.mean((hi_norm - RUL)**2)
        return mse
    target = _target_2
    target_vars = DEGRAD_VARS
    bounds = [(-1000, 1000)] * 6

if USE_ONLY_TRAIN:
    temp = df.copy().loc[df["ESN"] != TESTING_ESN]
else:
    temp = df.copy()

res = residuals(temp)
temp[DEGRAD_VARS] = res[DEGRAD_VARS]
X_train = temp.copy()

if USE_CLEAN_DATA:
    X_train = X_train.groupby(["ESN", "Cycles_Since_New"], as_index=False).median()
    X_train = pp.remove_outliers(X_train, threshold=OUTLIERS_THRESHOLD)
    X_train = X_train.ffill()
    X_train = X_train.bfill().dropna()

coef_data = X_train.copy()
# %store coef_data

# chpt = {}
# chpc = {}

# for esn in X_train["ESN"].unique():
#   print(esn)
#   tv = X_train.loc[X_train["ESN"] == esn, target_vars]
#   rul = X_train.loc[X_train["ESN"] == esn, "Cycles_to_HPT_SV"]
#   result_hpt = differential_evolution(
#       target,
#       bounds=bounds,
#       args=(tv, rul),
#       strategy='best1bin',
#       maxiter=MAXITER,                # generazioni
#       popsize=POPSIZE,
#       workers=-1,
#       tol=TOL,                      # Tolleranza
#   )

#   chpt[str(esn)] = result_hpt.x

#   rul = X_train.loc[X_train["ESN"] == esn, "Cycles_to_HPC_SV"]
#   result_hpc = differential_evolution(
#       target,
#       bounds=bounds,
#       args=(tv, rul),
#       strategy='best1bin',
#       maxiter=MAXITER,                # generazioni
#       popsize=POPSIZE,
#       workers=-1,
#       tol=TOL,                      # Tolleranza
#   )
#   chpc[str(esn)] = result_hpc.x

# if not SEPARATE_COEFS:
#     chpt = np.median(np.array(chpt.values()))
#     chpc = np.median(np.array(chpc.values()))




# print("\nCOEFFICIENTI MEDI FINALI (Training Set):")
# print(f"HPT: {chpt}")
# print(f"HPC: {chpc}")


# # %store coefs_hpt
# # %store coefs_hpc


# %%
chpt = {'101': -2.21999815, '102': -2.21689538, '103': -1.86943406, '104': -2.43053548}
chpc = {'101': 4.23411867, '102': 3.39890229, '103': 4.70926506, '104': 3.85075143}

# %%
# PLOTTING SU DATI DI TRAINING
hpt_limits, hpc_limits, ww_limits = [], [], []
MEAN_WINDOW_HPT = 45
MEAN_WINDOW_HPC = 100

scaling_coefs_hpt = {}
scaling_coefs_hpc = {}
scaling_coefs_hpt_final = {}
scaling_coefs_hpc_final = {}

def calc_hi(sd, ahpt, ahpc):
  if USE_ALL_VARS:
    hi_hpt = HIE(ahpt, sd[target_vars])
    hi_hpc = HIE(ahpc, sd[target_vars])
  else:
    hi_hpt = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpt)
    hi_hpc = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpc)
  return hi_hpt, hi_hpc

# Funzione per portare l'health index nella scala della rul
# In fase di testing bisogna usare una funzione differente, se no diventa troppo complicato. La scrivo nella cella sotto.
def scale_to_target(source, target, coefs):
    s_min, s_max = source.min(), source.max()
    t_min, t_max = target.min(), target.max()
    # Salvo i coefficienti per scalare in fase di test
    # quando non ho il ground truth
    if isinstance(coefs, dict):
        if "min" not in coefs:
            coefs["min"] = []
        if "max" not in coefs:
            coefs["max"] = []   
        coefs["min"].append(t_min)
        coefs["max"].append(t_max)
    return (source - s_min) / (s_max - s_min) * (t_max - t_min) + t_min

#alias per cambiare facilmente dataset
data = coef_data.copy()
esns = data["ESN"].unique()
fig, axs = plt.subplots(len(esns), 2, figsize=(30, len(esns)*6))
axs = axs.flatten()
i = 0
for esn in esns:
  sd = data[data["ESN"] == esn].copy()

  if SEPARATE_COEFS:
    if isinstance(chpt, dict) and isinstance(chpc, dict):
      ahpt = chpt[str(esn)]
      ahpc = chpc[str(esn)]
    else:
      print("Zi che combini?")
  else:
    ahpt = chpt
    ahpc = chpc

  hi_hpt, hi_hpc = calc_hi(sd, ahpt, ahpc)
  # Smoothing con la media per migliore interpretabilità
  hi_hpt_smooth = pd.Series(hi_hpt).rolling(window=MEAN_WINDOW_HPT, min_periods=1).mean()
  hi_hpc_smooth = pd.Series(hi_hpc).rolling(window=MEAN_WINDOW_HPC, min_periods=1).mean()

  # Porto l'hi alla stessa scala della RUL 
  hi_hpt_final = scale_to_target(hi_hpt_smooth, sd.loc[sd["ESN"] == esn, "Cycles_to_HPT_SV"], scaling_coefs_hpt)
  hi_hpc_final = scale_to_target(hi_hpc_smooth, sd.loc[sd["ESN"] == esn, "Cycles_to_HPC_SV"], scaling_coefs_hpc)

  axs[i].set_title(f'{"Training" if esn != test_data["ESN"].unique()[0] else "TEST" }: ESN - {esn}', fontsize=16)
  axs[i].plot(hi_hpt_final, color='tab:blue', label='Health Index (HPT)')
  axs[i].plot(sd.loc[sd["ESN"] == esn, "Cycles_to_HPT_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  i += 1
  axs[i].set_title(f'{"Training" if esn != test_data["ESN"].unique()[0] else "TEST" }: ESN - {esn}', fontsize=16)
  axs[i].plot(hi_hpc_final, color='tab:green', label='Health Index (HPC)')
  axs[i].plot(sd.loc[sd["ESN"] == esn, "Cycles_to_HPC_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  i += 1
  fig.tight_layout()
  fig.show()

scaling_coefs_hpt_final["min"] = np.median(scaling_coefs_hpt["min"])
scaling_coefs_hpt_final["max"] = np.median(scaling_coefs_hpt["max"])
scaling_coefs_hpc_final["min"] = np.median(scaling_coefs_hpc["min"])
scaling_coefs_hpc_final["max"] = np.median(scaling_coefs_hpc["max"])
print(scaling_coefs_hpt_final)
print(scaling_coefs_hpc_final)


# %%
# Funzione per portare l'health index nella scala della rul
def scale_to_target_test(source, coefs):
    s_min, s_max = source.min(), source.max()
    return (source - s_min) / (s_max - s_min) * (coefs["max"] - coefs["min"]) + coefs["min"]


# I COEFFICIENTI DA UTILIZZARE SONO:
# scaling_coefs_hpt_final
# scaling_coefs_hpc_final


# %% [markdown]
# # WW
# per il ww bisogna fare una cosa diversa. Intanto bisogna "normalizzare" la salita, ovvero eliminare gli effetti delle manutenzioni hpc e hpt sui residui di T45_res

# %%
# def remove_effect(df):
#     heads = df.groupby("Cumulative_HPT_SVs", as_index=False).head(1)
#     tails = df.groupby("Cumulative_HPT_SVs", as_index=False).tail(1)
    
#     heads = heads[1:]
#     tails = tails[:-1]
    
#     print(heads["Cumulative_HPT_SVs"])
#     print(tails["Cumulative_HPT_SVs"])
#     for c in len(heads["Cumulative_HPT_SVs"]):
#         difference = df.iloc[c, "Sensed_T45"]


def remove_effect(df, col, cycles_col="Cycles_Since_New"):
    df = df.sort_values([col, cycles_col, "Snapshot"])
    grp_stats = df.groupby(col)["Sensed_T45"].agg(['first', 'last'])
    grp_stats = grp_stats.sort_index()
    grp_stats['prev_last'] = grp_stats['last'].shift(1)
    grp_stats['jump'] = grp_stats['first'] - grp_stats['prev_last']
    grp_stats['jump'] = grp_stats['jump'].fillna(0)
    grp_stats['cumulative_offset'] = grp_stats['jump'].cumsum()
    offset_map = grp_stats['cumulative_offset'].to_dict()
    df["Sensed_T45"] = df["Sensed_T45"] - df[col].map(offset_map)
    return df

def remove_effect_predict(df, res_col, cycles_col="Cycles_Since_New", threshold=3.0):
    df = df.sort_values([cycles_col, "Snapshot"]).copy()
    df["res"] = res_col
    diffs = df["res"].diff().fillna(0)
    df["jumps"] = (diffs.abs() > threshold).cumsum()
    df = remove_effect(df, "jumps", cycles_col=cycles_col)
    return df

def comedicoio(df, col, cycles_col="Cycles_Since_New"):
    df = df.sort_values([cycles_col, "Snapshot"]).copy()
    try:
        is_new_regime = df[col].diff() > 0
        t45_diff = df["Sensed_T45"] - df["Sensed_T45"].shift(1)
        step_jumps = t45_diff.where(is_new_regime, 0.0)
        gdiff = step_jumps.cumsum()
        df["Sensed_T45"] = df["Sensed_T45"] - gdiff
        return df
    except:
        hi_hpt, hi_hpc = calc_hi(df, ahpt, ahpc)
        df = remove_effect_predict(df, hi_hpt, cycles_col=cycles_col)
        df = remove_effect_predict(df, hi_hpc, cycles_col=cycles_col)
        return df

def plot_t45_no_effect(data, res_data, win=1, mp=1, cycles_col="Cycles_Since_New"):
    plt.figure(figsize=(10, 7))
    unique_esns = data["ESN"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_esns)))
    dfs = []
    for i, esn in enumerate(unique_esns):
        mask = data["ESN"] == esn
        wwdf = data.loc[mask].copy()
        wwdf[DEGRAD_VARS] = res_data.loc[mask, DEGRAD_VARS]
        hi_hpt, hi_hpc = calc_hi(wwdf, ahpt, ahpc)
        wwdf = comedicoio(wwdf, "Cumulative_HPT_SVs", cycles_col=cycles_col)
        wwdf = comedicoio(wwdf, "Cumulative_HPC_SVs", cycles_col=cycles_col)
        # wwdf = remove_effect_predict(wwdf, hi_hpt)
        # wwdf = remove_effect_predict(wwdf, hi_hpc)
        dfs.append(wwdf)
        if win > 1:
            wwdf = wwdf.rolling(window=win, min_periods=mp).mean().dropna()
        X = wwdf[cycles_col].values.reshape(-1, 1)
        Y = wwdf["Sensed_T45"].values
        reg = LinearRegression().fit(X, Y)
        slope = reg.coef_[0]
        y_pred = reg.predict(X)

        plt.plot(wwdf[cycles_col], Y, 
                 label=f"ESN {esn} (Data)", 
                 color=colors[i], 
                 linewidth=0.3, 
                 alpha=0.6)
        
        plt.plot(wwdf[cycles_col], y_pred, 
                 label=f"ESN {esn} Slope: {slope:.5f}", 
                 color=colors[i], 
                 linewidth=2.5, 
                 linestyle='--')
        try: 
            jumps = wwdf[wwdf["Cycles_to_WW"].diff() > 0]
            if not jumps.empty:
                first_jump = True 
                for x_val in jumps["Cycles_Since_New"]:
                    label = f"Jump ESN {esn}" if first_jump else None
                    plt.axvline(x=x_val, color=colors[i], linestyle='--', 
                                linewidth=1.5, alpha=0.6, label=label) 
                    first_jump = False
        except KeyError:
            print("No WW data")

    if len(unique_esns) > 1:
        all_processed_data = pd.concat(dfs)
        X_all = all_processed_data[cycles_col].values.reshape(-1, 1)
        Y_all = all_processed_data["Sensed_T45"].values
        reg_all = LinearRegression().fit(X_all, Y_all)
        slope_all = reg_all.coef_[0]
        x_range = np.array([X_all.min(), X_all.max()]).reshape(-1, 1)
        y_pred_all = reg_all.predict(x_range)
        plt.plot(x_range, y_pred_all,
                 label=f"Overall mean slope: {slope_all:.5f}", 
                 color='purple', 
                 linewidth=2, 
                 linestyle='-')

    plt.title("Slope Analysis: Sensed_T45 vs Cycles")
    plt.xlabel("Cycles Since New")
    plt.ylabel("Sensed_T45 (Residuals)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_t45_no_effect(test_data, res_test, 1,1)
plot_t45_no_effect(train_data_testing, res_all_train, 1, 1)

# %%
# VALIDATION
for i in range(0, 48):
    dfv = u.load_validation(i)
    res_val = residuals(dfv.copy())

    X_train = res_val.groupby(["ESN", "Cycles"], as_index=False).median()
    X_train = pp.remove_outliers(X_train, threshold=OUTLIERS_THRESHOLD)
    X_train = X_train.ffill()
    X_train = X_train.bfill().dropna()

    ahpt = np.mean(list(chpt.values()))
    ahpc = np.mean(list(chpc.values()))

    data = X_train.copy()
    for esn in data["ESN"].unique():
        sd = data[data["ESN"] == esn].copy()

        hi_hpt, hi_hpc = calc_hi(sd, ahpt, ahpc)

        fig, axs = plt.subplots(1, 2, figsize=(30, 6))
        fig.suptitle(f'Training: ESN - {esn}', fontsize=16)
        axs[0].plot(hi_hpt.rolling(window=1,min_periods=1).mean(), color='tab:blue', label='Health Index (HPT)')
        # ax = axs[0].twinx()
        # ax.plot(sd.loc[sd["ESN"] == esn, "Cycles_to_HPT_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
        axs[1].plot(hi_hpc.rolling(window=1,min_periods=1).mean(), color='tab:green', label='Health Index (HPC)')
        # ax = axs[1].twinx()
        # ax.plot(sd.loc[sd["ESN"] == esn, "Cycles_to_HPC_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
        # axs[2].plot(hi_ww, color='tab:green', label='Health Index (HPC)')
        # axs[2].plot(ww_rul_esn["Cycles_to_WW"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
        fig.tight_layout()
        fig.show()
    plot_t45_no_effect(dfv, res_val, cycles_col="Cycles")

# %%
# VALIDATION insomma testing cioè quello che si chiama validation ma è testing
for i in range(0, 52):
    dft = u.load_testing(i)
    res_val = residuals(dft.copy())

    X_train = res_val.groupby(["ESN", "Cycles"], as_index=False).median()
    X_train = pp.remove_outliers(X_train, threshold=OUTLIERS_THRESHOLD)
    X_train = X_train.ffill()
    X_train = X_train.bfill().dropna()

    ahpt = np.mean(list(chpt.values()))
    ahpc = np.mean(list(chpc.values()))

    data = X_train.copy()
    for esn in data["ESN"].unique():
        sd = data[data["ESN"] == esn].copy()
        hi_hpt, hi_hpc = calc_hi(sd, ahpt, ahpc)
        fig, axs = plt.subplots(1, 2, figsize=(30, 6))
        fig.suptitle(f'Training: ESN - {esn}', fontsize=16)
        axs[0].plot(hi_hpt.rolling(window=1,min_periods=1).mean(), color='tab:blue', label='Health Index (HPT)')
        axs[1].plot(hi_hpc.rolling(window=1,min_periods=1).mean(), color='tab:green', label='Health Index (HPC)')
        fig.tight_layout()
        fig.show()
