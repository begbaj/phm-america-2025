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
cycles_healthy = 7

augmented_data = False
augmented_count = 100
SMOTE = False

TESTING_ESN = 102
INCLUDE_TEST = True

if INCLUDE_TEST:
    train_data = df.copy()
else:
    train_data = df[df["ESN"] != TESTING_ESN].copy()

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

if ENSAMBLE:
  # con ESNAMBLE creiamo un modello per ogni ESN
  # e poi, con la funziona residual, otterremo la media
  # di tutti i valori predetti
  models = {}
  for esn in train_data["ESN"].unique():
      start = time.time()
      print(f"esn {esn}: ", end="")
      mask = train_data["ESN"] == esn
      X_train = train_data.loc[mask, OPERATING_VARS]
      Y_train = train_data.loc[mask, DEGRAD_VARS]
      model = train_model(X_train, Y_train)
      end = time.time()
      print(end - start)
      models[str(esn)] = model
  del model
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


def residual_regressor(data, esn=None):
  if not SEPARATE_MODELS:
    predictions = []
    for model in models.values():
      predictions.append(model.predict(data))
    return np.mean(predictions, axis=0) # mean lo applichiamo indistintamente, nel secondo caso la media è pari al valore stesso
  else:
    if esn:
      try:
        return models[str(esn)].predict(data)
      except KeyError:
        print("Non ci sono modelli addestrati per questo motore")
        return None
    else:
      print("Dagli un ESN porcone")





# %%
# calcolo residui
def residuals(df):
  res_list = []
  for esn in df["ESN"].unique():
    mask = df["ESN"] == esn
    X_train = df.loc[mask, OPERATING_VARS]
    Y_train = df.loc[mask, DEGRAD_VARS]
    if SEPARATE_MODELS:
      Y_pred = residual_regressor(X_train, esn)
    else:
      Y_pred = residual_regressor(X_train)
    if Y_pred is None:
      return
    twe = np.mean(TWE(Y_pred, Y_train))
    res_temp = Y_train - Y_pred
    res_temp = pp.remove_outliers(res_temp, threshold=3)
    res_temp.rolling(window=10,min_periods=1).median()
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
    fig.suptitle(f'Residuals Comparison (Window: {window})', fontsize=16)
    axs = axs.flatten()
    for esn in data["ESN"].unique():
        if "aug" in str(esn):
            continue
        res_temp = data[data["ESN"] == esn]
        if GROUP_CYCLES:
            res_temp = res_temp.groupby("Cycles").mean()
        if REMOVE_OUTLIERS:
            res_temp = pp.remove_outliers(res_temp, threshold=OUTLIERS_THRESHOLD)
            res_temp = res_temp.ffill()
            res_temp = res_temp.bfill()
        for i, ax in enumerate(axs):
            if i < len(DEGRAD_VARS): # Safety check
                d_var = DEGRAD_VARS[i]
                t = res_temp[d_var]
                degrad = t.rolling(window=window, min_periods=min).mean().reset_index(drop=True)
                ax.plot(degrad, linewidth=0.6, alpha=0.7, label=str(esn))
                ax.set_title(d_var)
                ax.grid(True, alpha=0.3)
                
    axs[0].legend(fontsize='small', loc='upper right')
    plt.tight_layout()
    plt.show()

plot(res_train, 1, 1)
plot(res_test, 1, 1)
# plot(residuals(dfv), 10, 1)
# plot(residuals(dft), 10, 1)


# %% [markdown]
# # HPT e HPC
# ## Ricerca di a,b,c,d,e,f,g globali combinazione lineare di tutti i sensori

# %%
USE_ALL_VARS = False
MAXITER = 100
POPSIZE = 500
TOL = 0.001
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

chpt = {}
chpc = {}

for esn in X_train["ESN"].unique():
  print(esn)
  tv = X_train.loc[X_train["ESN"] == esn, target_vars]
  rul = X_train.loc[X_train["ESN"] == esn, "Cycles_to_HPT_SV"]
  result_hpt = differential_evolution(
      target,
      bounds=bounds,
      args=(tv, rul),
      strategy='best1bin',
      maxiter=MAXITER,                # generazioni
      popsize=POPSIZE,
      workers=-1,
      tol=TOL,                      # Tolleranza
  )

  chpt[str(esn)] = result_hpt.x

  rul = X_train.loc[X_train["ESN"] == esn, "Cycles_to_HPC_SV"]
  result_hpc = differential_evolution(
      target,
      bounds=bounds,
      args=(tv, rul),
      strategy='best1bin',
      maxiter=MAXITER,                # generazioni
      popsize=POPSIZE,
      workers=-1,
      tol=TOL,                      # Tolleranza
  )
  chpc[str(esn)] = result_hpc.x

if not SEPARATE_COEFS:
    chpt = np.median(np.array(chpt.values()))
    chpc = np.median(np.array(chpc.values()))




print("\nCOEFFICIENTI MEDI FINALI (Training Set):")
print(f"HPT: {chpt}")
print(f"HPC: {chpc}")


# %store coefs_hpt
# %store coefs_hpc


# %%
# PARAMETRI DI TEST SOLO PER HPT E HPC

# coefs_hpt = [276.19439386,
#              -343.90020315,
#              781.45341037,
#              -593.41135514,
#              -190.36846837,
#              118.5466599]

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

#alias per cambiare facilmente dataset
data = coef_data.copy()
for esn in data["ESN"].unique():
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

  if USE_ALL_VARS:
    hi_hpt = HIE(ahpt, sd[target_vars])
    hi_hpc = HIE(ahpc, sd[target_vars])
  else:
    hi_hpt = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpt)
    hi_hpc = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpc)

  fig, axs = plt.subplots(1, 2, figsize=(30, 6))
  fig.suptitle(f'Training: ESN - {esn}', fontsize=16)
  axs[0].plot(hi_hpt, color='tab:blue', label='Health Index (HPT)')
  # ax = axs[0].twinx()
  # ax.plot(sd.loc[sd["ESN"] == esn, "Cycles_to_HPT_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  axs[1].plot(hi_hpc, color='tab:green', label='Health Index (HPC)')
  # ax = axs[1].twinx()
  # ax.plot(sd.loc[sd["ESN"] == esn, "Cycles_to_HPC_SV"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
  # axs[2].plot(hi_ww, color='tab:green', label='Health Index (HPC)')
  # axs[2].plot(ww_rul_esn["Cycles_to_WW"], color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
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
# LIGHTGBM

X_hpc_list, y_hpc_list = [], []
X_hpt_list, y_hpt_list = [], []
X_ww_list, y_ww_list = [], []

# Sui dati di training
for esn in res_train["ESN"].unique():

    sd = res_train[res_train["ESN"] == esn].reset_index().copy()

    # Calcolo e standardizzazione degli health index
    hi_hpt = normalize(HIE(chpt, sd[DEGRAD_VARS])).dropna()
    hi_hpc = normalize(HIE(chpc, sd[DEGRAD_VARS])).dropna()
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
    sd = res_train[res_train["ESN"] == esn].reset_index().copy()

    # Calcolo e standardizzazione degli health index
    hi_hpt = normalize(HIE(chpt, sd[DEGRAD_VARS])).dropna()
    hi_hpc = normalize(HIE(chpc, sd[DEGRAD_VARS])).dropna()
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
val_hi_hpt = normalize(HIE(chpt, res_val[DEGRAD_VARS])).dropna()
val_hi_hpc = normalize(HIE(chpc, res_val[DEGRAD_VARS])).dropna()
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
# per il ww bisogna fare una cosa diversa. Intanto bisogna "normalizzare" la salita, ovvero eliminare gli effetti delle manutenzioni hpc e hpt sui residui di T45_res

# %%
res_test["Sensed_T45"].plot()

