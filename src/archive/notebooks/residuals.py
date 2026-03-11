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
import time
from sklearn.neighbors import NearestNeighbors
import tools
from scipy import stats
from scipy.optimize import differential_evolution
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgbm

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# %load_ext autoreload
# %autoreload 2

from tools import utils as u, config as cfg, plotting as up, preprocessing as pp


# %%
# CONSTANTS

DATA_BASE_PATH = f"../../Data/"
DATA_TESTING_PATH = f"{DATA_BASE_PATH}/PHM2025_test_data/"
DATA_TRAINING_PATH = f"{DATA_BASE_PATH}/PHM2025_training_data/"
DATA_VALIDATION_PATH = f"{DATA_BASE_PATH}/PHM2025_validation_data/"
DATA_TRAINING_DATA = f"{DATA_TESTING_PATH}/training_data.csv"
PLOT_PATH = f"./img/"

OPERATING_VARS = [
    "Sensed_Altitude",
    "Sensed_Mach",
    "Sensed_Pamb",
    "Sensed_TAT",
    "Sensed_VAFN",
    "Sensed_VBV",
    "Sensed_Fan_Speed",
    "Sensed_Pt2",
]

SENSORS = tools.types.enums.SENSORS

# considera solo le variabili presenti anche in test e val
DEGRAD_VARS = [
    s
    for s in SENSORS
    if s not in OPERATING_VARS and s != "Sensed_P25" and s != "Sensed_T5"
]
# scommentare se si vuole considerare anche i sensori non in test o valiation
# DEGRAD_VARS = [s for s in u.SENSORS if s not in operating_vars]
ALL_VARS = OPERATING_VARS + DEGRAD_VARS


def DATA_TEST_DATA(num):
    return f"{DATA_TESTING_PATH}/test_{num}.csv"


def DATA_VALIDATION_DATA(num):
    return f"{DATA_TESTING_PATH}/val_{num}.csv"


# %%
# FUNZIONI


def train_models(
    df, operating_vars, degradation_vars
) -> dict[int, dict[str, LinearRegression]]:
    X_train = df[operating_vars]
    Y_train = df[degradation_vars]
    models = {}
    for i in range(0, 8):
        X_temp = pd.DataFrame(np.roll(X_train, i, axis=1))
        models[i] = {}
        models[i]["model"] = train_model(X_temp, Y_train)
    return models


def train_model(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model


# Health Index


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
    for i in range(0, 7):
        m = df.iloc[:, i].median()
        df.iloc[:, i] -= m
    return df


def objective(alpha, T3, T45, RUL):
    hi = -alpha * T3 - T45
    RUL = RUL.dropna()
    hi = hi.dropna()
    corr = stats.pearsonr(RUL, hi)
    # return np.sqrt(np.mean((hi - RUL)**2)) + 1
    return -corr[0]


def objective_beta(params, T3, T45, RUL):
    alpha, beta = params
    hi = -alpha * T3 - beta * T45
    RUL = RUL.dropna()
    hi = hi.dropna()
    corr = stats.pearsonr(RUL, hi)
    # return np.sqrt(np.mean((hi - RUL)**2)) + 1
    return -corr[0]


def HIE(params, vars):
    # return np.sum([-params[i]*vars.iloc[:,i] for i in range(0, 8)])
    return vars.dot(-np.array(params))


# TWE


def wind(y_p, y, a):
    diff = y - y_p
    num = np.where(diff >= 0, 2.0, 1.0)
    if isinstance(y_p, pd.DataFrame) or isinstance(y_p, pd.Series):
        y_p = y_p.values
    return num / (1 + a * y_p)


def time_weighted_error(y_true, y_pred, alpha=0.02, beta=1):
    """Returns the weighted squared error for an array of predictions."""

    error = y_pred - y_true

    weight = np.where(error >= 0, 2 / (1 + alpha * y_true), 1 / (1 + alpha * y_true))
    return weight * (error**2) * beta


def TWE(y_p, y, a=0.02, b=1):
    # if isinstance(y_p, pd.DataFrame): y_p = y_p.values
    # weight = wind(y_p, y, a)
    # squared_error = (y - y_p) ** 2
    # return weight * squared_error * b
    return time_weighted_error(y_p, y, a, b)


# TARGET FUNCTIONS


def normalize(col):
    col_min, col_max = col.min(), col.max()
    col = (col - col_min) / (col_max - col_min)
    col = col.to_frame()
    return col


# def get_slope(y):
#     """Calcola la pendenza della retta di regressione per una finestra y"""
#     x = np.arange(len(y))
#     # Polyfit di grado 1 restituisce [pendenza, intercetta]
#     slope = np.polyfit(x, y, 1)[0]
#     return slope

def get_slope(y):
    """Calcola la pendenza della retta di regressione per una finestra y"""
    if len(y) < 2:
        return 0.0
    # Controlla se tutti i valori sono uguali o contengono NaN/Inf
    if np.all(y == y[0]) or not np.all(np.isfinite(y)):
        return 0.0
    x = np.arange(len(y))
    # Usa formula diretta invece di polyfit (più stabile numericamente)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return slope


# %% [markdown]
# # CONFIGURAZIONE


# %% [markdown]
# ## Caricamento dati
# %%
# TRAINING
df = u.load_training()
df = pp.remove_outliers(df, SENSORS)
df = pp.missingfill(df).dropna()

# VALIDATION
no_concat = True
dfv = u.load_validation(range(0, 48), no_concat=no_concat)
if no_concat:
    for i, f in enumerate(dfv):
        dfv[i] = pp.remove_outliers(f, SENSORS)
        dfv[i] = pp.missingfill(f, align_cols=["Snapshot", "Cycles"]).dropna()
else:
    dfv = pp.remove_outliers(dfv, SENSORS)
    dfv = pp.missingfill(dfv, align_cols=["Snapshot", "Cycles"]).dropna()

# TESTING
no_concat = True
dft = u.load_testing(range(0, 52), no_concat=no_concat)
if no_concat:
    for i, f in enumerate(dft):
        dft[i] = pp.remove_outliers(f, SENSORS)
        dft[i] = pp.missingfill(f, align_cols=["Snapshot", "Cycles"]).dropna()
else:
    dft = pp.remove_outliers(dft, SENSORS)
    dft = pp.missingfill(dft, align_cols=["Snapshot", "Cycles"]).dropna()

# %% [markdown]
# # Regressione Lineare 1 - modello andamento nominale
# %%

# Quale motore nel training set verrà utilizzato come
# validation (si usa tecnica Leave-One-Out)
TESTING_ESN = 103

# Si vuole comunque utilizzare questo motore training?
INCLUDE_TEST = False

# REGRESSORE LINEARE ANDAMENTO NOMINALE
# indica quanti cicli devono essere considerati healthy nel training
cycles_healthy = 5
# se utilizzare il metodo di data augmentation oppure no
augmented_data = False
# se si, quanti dati aggiungere per ogni motore
augmented_count = 100
# se utilizzare SMOTE oppure no
SMOTE = False

# si vuole utilizzare la tecnica Ensamble?
# questa tecnica effettuerà il training per ogni motore
# e poi quando si utilizzarà il regressore verà fatta
# una media dei valori di output di ogni regressore
ENSAMBLE = True

# se ENSAMBLE è attivo, questo permette di far utilizzare
# il regressore dedicato al motore, se esiste, altrimenti
# esegue ensamble
SEPARATE_MODELS = True


# Preparazione dati training
if INCLUDE_TEST:
    train_data = df.copy()
else:
    train_data = df[df["ESN"] != TESTING_ESN].copy()

test_data = df[df["ESN"] == TESTING_ESN].copy()

if cycles_healthy > 0:
    train_data = (
        train_data.groupby("ESN").head(cycles_healthy * 8).reset_index(drop=True).copy()
    )
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
            diff = (
                train_data.iloc[neighbor_indices][ALL_VARS].values
                - cur[ALL_VARS].values
            )
            new_vals = cur[ALL_VARS].values + diff * np.random.rand(len(cur), 1)

            aug_df = cur.copy()
            aug_df[ALL_VARS] = new_vals
            aug_df["ESN"] = f"aug_{i}_{esn}"

            newdf.append(aug_df)
    print("total data", len(newdf))

    train_data = pd.concat([train_data] + newdf, ignore_index=True)
    del newdf, aug_df, nbrs, idx, diff, new_vals

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
        return ensamble(data)
    else:
      print("Dagli un ESN porcone")


# %% [markdown]
# ### calcolo residui
# crea funzione `residuals(df)` per ottenere i residui utilizzando il regressore
# %%
def _residuals(df):
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
        res_temp = Y_train - Y_pred
        res_temp = pp.remove_outliers(res_temp, threshold=3, method="iqr")
        res_temp.rolling(window=10, min_periods=1).median()
        res_temp = res_temp.ffill()
        res_temp = res_temp.bfill()
        res_temp["ESN"] = esn
        try:
            res_temp["Cycles"] = df.loc[mask, "Cycles_Since_New"]
        except:
            res_temp["Cycles"] = df.loc[mask, "Cycles"]
        res_list.append(res_temp)
    return pd.concat(res_list)

def residuals(df):
    if isinstance(df, list):
        res_list = []
        for i, d in enumerate(df):
            res_list.append(_residuals(d))
    else:
        res_list = [_residuals(df)]
    return pd.concat(res_list)

print("training dataset")
res_train_healthy = residuals(train_data)
res_train = residuals(df[df["ESN"] != TESTING_ESN].copy())
print("leave-one-out testing esn")
res_test = residuals(test_data)

print("DFT - testing dataframe")
res_dft = residuals(dft)
print("DFV - validation dataset")
res_dfv = residuals(dfv)

# %% [markdown]
# # PLOTTING TRAINING
# %%

# nella visualizzazione, raggruppare i cicli per gruppi?
from pyparsing import line


GROUP_CYCLES = True
# rimuovere outliers durante il plot?
REMOVE_OUTLIERS = True
# il metodo usato è "z-score"
OUTLIERS_THRESHOLD = 3


def plot(data, window, min):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Residuals Comparison (Window: {window})", fontsize=16)
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
            if i < len(DEGRAD_VARS):  # Safety check
                d_var = DEGRAD_VARS[i]
                t = res_temp[d_var]
                degrad = (
                    t.rolling(window=window, min_periods=min)
                    .mean()
                    .reset_index(drop=True)
                )
                ax.plot(degrad, linewidth=0.6, alpha=0.6, label=str(esn))
                ax.scatter(range(len(degrad)), degrad, linewidth=0.6, alpha=0.7, label=str(esn), s=2)
                ax.set_title(d_var)
                ax.grid(True, alpha=0.3)

    axs[0].legend(fontsize="small", loc="upper right")
    plt.tight_layout()
    plt.show()


plot(res_train_healthy, 1, 1)
plot(res_train, 1, 1)
plot(res_test, 1, 1)
# plot(residuals(dfv), 10, 1)
# plot(residuals(dft), 10, 1)


# %% [markdown]
# # HPT e HPC
# Ricerca del valore del coefficiente $\alpha$ in base alla correlazione
# Se USE_ALL_VARS è attivo, i coefficienti sono $\alpha$ e $\beta$, altrimenti solo $\alpha$.
# $\alpha$ è applicato a $\text{Sensed\_T3}$, mentre, se esiste, $\beta$ verrà applicato a $\text{Sensed\_T45}$
# %%
# se si vuole evitare la ricerca e usare parametri default
DO_NOT_TRAIN_COEFS = False

USE_ALL_VARS = False
THIS_ALL_VARS = ["Sensed_T3", "Sensed_T45", "Sensed_Core_Speed", "Sensed_T25"]

# Parametri di configurazione di differential_evolution()
MAXITER = 1000
POPSIZE = 500
TOL = 0.0001

# True: usa solo i motori indicati come train, False prende anche TESTING_ESN
USE_ONLY_TRAIN = False

# Preprocessare i dati?
USE_CLEAN_DATA = True
OUTLIERS_THRESHOLD = 3

# Ogni motore avrà i proprio coefficienti?
SEPARATE_COEFS = False

if USE_ONLY_TRAIN:
    esn_data = df[df["ESN"] != TESTING_ESN].copy()
else:
    esn_data = df.copy()

res = residuals(esn_data)
esn_data[DEGRAD_VARS] = res[DEGRAD_VARS]
X_train = esn_data.copy()

if USE_CLEAN_DATA:
    X_train = X_train.groupby(["ESN", "Cycles_Since_New"], as_index=False).median(numeric_only=True)
    X_train = pp.remove_outliers(X_train, threshold=OUTLIERS_THRESHOLD)
    X_train = X_train.ffill()
    X_train = X_train.bfill().dropna()

coef_data = X_train.copy()

# ESECUZIONE
if not DO_NOT_TRAIN_COEFS:
    target = None
    target_vars = []

    if not USE_ALL_VARS:
        def _target_1(a, sensor_data, RUL):
            hi = HI(sensor_data["Sensed_T3"], sensor_data["Sensed_T45"], a)
            # Allinea indici dopo dropna
            valid = hi.dropna().index.intersection(RUL.dropna().index)
            if len(valid) < 3:
                return 1.0
            hi_valid = hi.loc[valid]
            rul_valid = RUL.loc[valid]
            if hi_valid.max() == hi_valid.min():
                return 1.0
            corr = stats.pearsonr(rul_valid, hi_valid)
            return -corr[0]

        target = _target_1
        target_vars = ["Sensed_T45", "Sensed_T3"]
        bounds = [(-1000, 1000)]
    else:
        def _target_2(params, sensor_data, RUL):
            hi = HIE(params, sensor_data)
            hi_min, hi_max = hi.min(), hi.max()
            if hi_max == hi_min:
                return 1.0
            hi_norm = (hi - hi_min) / (hi_max - hi_min)
            valid = hi_norm.dropna().index.intersection(RUL.dropna().index)
            if len(valid) < 3:
                return 1.0
            mse = np.mean((hi_norm.loc[valid] - RUL.loc[valid]) ** 2)
            return mse

        target = _target_2
        target_vars = THIS_ALL_VARS
        bounds = [(-1000, 1000)] * len(target_vars)

    chpt = {}
    chpc = {}

    for esn in X_train["ESN"].unique():
        print(f"Optimizing ESN {esn}...", end=" ")
        esn_mask = X_train["ESN"] == esn
        tv = X_train.loc[esn_mask, target_vars]

        # HPT
        rul_hpt = X_train.loc[esn_mask, "Cycles_to_HPT_SV"]
        result_hpt = differential_evolution(
            target, bounds=bounds, args=(tv, rul_hpt),
            strategy="best1bin", maxiter=MAXITER,
            popsize=POPSIZE, workers=-1, tol=TOL,
        )
        chpt[str(esn)] = result_hpt.x
        print(f"HPT α={result_hpt.x}", end=" | ")

        # HPC
        rul_hpc = X_train.loc[esn_mask, "Cycles_to_HPC_SV"]
        result_hpc = differential_evolution(
            target, bounds=bounds, args=(tv, rul_hpc),
            strategy="best1bin", maxiter=MAXITER,
            popsize=POPSIZE, workers=-1, tol=TOL,
        )
        chpc[str(esn)] = result_hpc.x
        print(f"HPC α={result_hpc.x}")

    if not SEPARATE_COEFS:
        # FIX: list() necessario in Python 3 per dict.values()
        chpt_vals = np.array(list(chpt.values()))
        chpc_vals = np.array(list(chpc.values()))
        chpt = np.median(chpt_vals)
        chpc = np.median(chpc_vals)

    print("\nCOEFFICIENTI FINALI (Training Set):")
    if isinstance(chpt, dict):
        for esn_key in chpt:
            print(f"  ESN {esn_key}: HPT α={chpt[esn_key]}, HPC α={chpc[esn_key]}")
    else:
        print(f"  HPT (mediana): {chpt}")
        print(f"  HPC (mediana): {chpc}")

else:
    chpt = {"101": -2.23102809, "102": -2.23102809, "103": -2.23102809, "104": -2.23102809}
    chpc = {"101": 4.25050266, "102": 4.25050266, "103": 4.25050266, "104": 4.25050266}

# %%
# PLOTTING SU DATI DI TRAINING

# alias per cambiare facilmente dataset
DATA = coef_data.copy()

hpt_limits, hpc_limits, ww_limits = [], [], []


def calc_hi(sd):
    if USE_ALL_VARS:
        hi_hpt = HIE(ahpt, sd[target_vars])
        hi_hpc = HIE(ahpc, sd[target_vars])
    else:
        hi_hpt = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpt)
        hi_hpc = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpc)
    return hi_hpt, hi_hpc


for esn in DATA["ESN"].unique():
    sd = DATA[DATA["ESN"] == esn].copy()

    if SEPARATE_COEFS:
        if isinstance(chpt, dict) and isinstance(chpc, dict):
            ahpt = chpt[str(esn)]
            ahpc = chpc[str(esn)]
        else:
            print("Zi che combini?")
    else:
        ahpt = chpt
        ahpc = chpc

    hi_hpt, hi_hpc = calc_hi(sd)

    fig, axs = plt.subplots(1, 2, figsize=(30, 6))
    fig.suptitle(f"Training: ESN - {esn}", fontsize=16)
    axs[0].plot(hi_hpt, color="tab:blue", label="Health Index (HPT)")
    axs[0].scatter(hi_hpt.index, hi_hpt, color="tab:blue", label="Health Index (HPT)", s=3)
    axs[1].plot(hi_hpc, color="tab:green", label="Health Index (HPC)")
    axs[1].scatter(hi_hpc.index, hi_hpc, color="tab:green", label="Health Index (HPC)", s=3)
    fig.tight_layout()
    fig.show()

# %% [markdown]
# ### Applicazione del LightGBM per HPT per terzo ciclo di manutenzione + correzione il gap

# %%
# ===== PREPARAZIONE DATI PER CLASSIFICATORE =====
from sklearn.model_selection import LeaveOneGroupOut
import lightgbm as lgbm
from sklearn.metrics import accuracy_score


# ===== SCALE TO TARGET =====
# Definite qui perché servono sia per il classificatore sia per il gap correction.

def scale_to_target(source, target, coefs):
    """Scala source nel range di target, salvando i coefficienti in coefs."""
    s_min, s_max = source.min(), source.max()
    t_min, t_max = target.min(), target.max()
    if isinstance(coefs, dict):
        if "min" not in coefs:
            coefs["min"] = []
        if "max" not in coefs:
            coefs["max"] = []
        coefs["min"].append(t_min)
        coefs["max"].append(t_max)
    denom = s_max - s_min
    if denom == 0:
        return pd.Series(np.full(len(source), (t_max + t_min) / 2), index=source.index)
    return (source - s_min) / denom * (t_max - t_min) + t_min


def scale_to_target_test(source, coefs):
    """Scala source usando i coefficienti salvati durante il training."""
    s_min, s_max = source.min(), source.max()
    denom = s_max - s_min
    if denom == 0:
        return pd.Series(np.full(len(source), (coefs["max"] + coefs["min"]) / 2), index=source.index)
    return (source - s_min) / denom * (coefs["max"] - coefs["min"]) + coefs["min"]


# Coefficienti globali per scale_to_target del classificatore
clf_scale_coefs_hpt = {
    "min": float(coef_data["Cycles_to_HPT_SV"].min()),
    "max": float(coef_data["Cycles_to_HPT_SV"].max()),
}
clf_scale_coefs_hpc = {
    "min": float(coef_data["Cycles_to_HPC_SV"].min()),
    "max": float(coef_data["Cycles_to_HPC_SV"].max()),
}


# STEP 1: Classificatore del ciclo di lavoro
# Obiettivo: dato lo stato attuale del motore, predire in quale
# "segmento di manutenzione" (0, 1, 2, ...)
# Features: residui + HI scalato (scale_to_target) + slope e rolling mean dell'HI scalato

def build_classification_features(data, window=20):
    """
    Costruisce features per il classificatore del ciclo di lavoro.
    Per ogni ESN, calcola HI, lo scala con scale_to_target, poi slope e rolling mean.
    """
    feat_list = []

    for esn in data["ESN"].unique():
        sd = data[data["ESN"] == esn].copy()

        if SEPARATE_COEFS:
            ahpt_local = chpt[str(esn)]
            ahpc_local = chpc[str(esn)]
        else:
            ahpt_local = chpt
            ahpc_local = chpc

        # Calcola HI
        if USE_ALL_VARS:
            hi_hpt = HIE(ahpt_local, sd[target_vars])
            hi_hpc = HIE(ahpc_local, sd[target_vars])
        else:
            hi_hpt = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpt_local)
            hi_hpc = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpc_local)

        # Scala HI con scale_to_target (coefficienti globali)
        hi_hpt = scale_to_target_test(hi_hpt, clf_scale_coefs_hpt)
        hi_hpc = scale_to_target_test(hi_hpc, clf_scale_coefs_hpc)

        feat = sd[DEGRAD_VARS].copy()
        feat["HI_HPT"] = hi_hpt.values
        feat["HI_HPC"] = hi_hpc.values
        feat["HI_HPT_slope"] = hi_hpt.rolling(window=window, min_periods=1).apply(get_slope, raw=True).values
        feat["HI_HPC_slope"] = hi_hpc.rolling(window=window, min_periods=1).apply(get_slope, raw=True).values
        feat["HI_HPT_rolling_mean"] = hi_hpt.rolling(window=window, min_periods=1).mean().values
        feat["HI_HPC_rolling_mean"] = hi_hpc.rolling(window=window, min_periods=1).mean().values
        feat["ESN"] = esn

        # Labels: ciclo di lavoro attuale
        feat["label_hpt"] = sd["Cumulative_HPT_SVs"].values
        feat["label_hpc"] = sd["Cumulative_HPC_SVs"].values

        feat_list.append(feat)

    return pd.concat(feat_list, ignore_index=True)


clf_data = build_classification_features(coef_data, window=20)
clf_feature_cols = [c for c in clf_data.columns if c not in ["ESN", "label_hpt", "label_hpc"]]

print(f"Features classificazione: {clf_feature_cols}")
print(f"Classi HPT: {sorted(clf_data['label_hpt'].unique())}")
print(f"Classi HPC: {sorted(clf_data['label_hpc'].unique())}")
print(f"Scale coefs classificatore HPT: {clf_scale_coefs_hpt}")
print(f"Scale coefs classificatore HPC: {clf_scale_coefs_hpc}")

# %%
# ===== TRAINING CLASSIFICATORE CICLO DI LAVORO =====
# Training unico con i parametri migliori (no cross-validation, no grid search)

X_clf = clf_data[clf_feature_cols].values
y_hpt_cycle = clf_data["label_hpt"].values.astype(int)
y_hpc_cycle = clf_data["label_hpc"].values.astype(int)

clf_hpt = lgbm.LGBMClassifier(
    objective="multiclass",
    n_estimators=600,
    learning_rate=0.002,
    max_depth=10,
    num_leaves=63,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)

clf_hpc = lgbm.LGBMClassifier(
    objective="multiclass",
    n_estimators=600,
    learning_rate=0.002,
    max_depth=10,
    num_leaves=63,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)

clf_hpt.fit(X_clf, y_hpt_cycle)
clf_hpc.fit(X_clf, y_hpc_cycle)

pred_hpt = clf_hpt.predict(X_clf)
pred_hpc = clf_hpc.predict(X_clf)
acc_hpt = accuracy_score(y_hpt_cycle, pred_hpt)
acc_hpc = accuracy_score(y_hpc_cycle, pred_hpc)

print(f"Classificatori addestrati (training unico).")
print(f"Accuratezza in-sample: HPT={acc_hpt:.4f}, HPC={acc_hpc:.4f}")


# %%
# ===== STEP 2 + 3: SCALE_TO_TARGET + GAP CORRECTION =====
def regress_features(data, window=20):
    """
    Features per la predizione diretta di Cycles_to_SV.
    """
    feat_list = []
    for esn in data["ESN"].unique():
        sd = data[data["ESN"] == esn].copy().sort_values("Cycles_Since_New")

        if SEPARATE_COEFS:
            ahpt_local = chpt[str(esn)]
            ahpc_local = chpc[str(esn)]
        else:
            ahpt_local = chpt
            ahpc_local = chpc

        if USE_ALL_VARS:
            hi_hpt = HIE(ahpt_local, sd[target_vars])
            hi_hpc = HIE(ahpc_local, sd[target_vars])
        else:
            hi_hpt = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpt_local)
            hi_hpc = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpc_local)

        feat = sd[DEGRAD_VARS].copy()

        # --- Features base ---
        feat["HI_HPT"] = hi_hpt.values
        feat["HI_HPC"] = hi_hpc.values

        # --- Features di trend (multi-scala) ---
        for w in [10, 20, 50]:
            feat[f"HI_HPT_slope_{w}"] = hi_hpt.rolling(window=w, min_periods=1).apply(get_slope, raw=True).values
            feat[f"HI_HPC_slope_{w}"] = hi_hpc.rolling(window=w, min_periods=1).apply(get_slope, raw=True).values
            feat[f"HI_HPT_mean_{w}"] = hi_hpt.rolling(window=w, min_periods=1).mean().values
            feat[f"HI_HPC_mean_{w}"] = hi_hpc.rolling(window=w, min_periods=1).mean().values
            feat[f"HI_HPT_std_{w}"] = hi_hpt.rolling(window=w, min_periods=2).std().fillna(0).values
            feat[f"HI_HPC_std_{w}"] = hi_hpc.rolling(window=w, min_periods=2).std().fillna(0).values

        # --- Features di posizione temporale ---
        # Delta rispetto al valore iniziale (quanto è degradato)
        feat["HI_HPT_delta_from_start"] = (hi_hpt.values - hi_hpt.values[0])
        feat["HI_HPC_delta_from_start"] = (hi_hpc.values - hi_hpc.values[0])

        # Velocità di degradazione cumulativa
        feat["HI_HPT_cumulative_change"] = hi_hpt.diff().fillna(0).cumsum().values
        feat["HI_HPC_cumulative_change"] = hi_hpc.diff().fillna(0).cumsum().values

        # Rapporto attuale/media storica (quanto siamo lontani dal "normale")
        expanding_mean_hpt = hi_hpt.expanding(min_periods=1).mean()
        expanding_mean_hpc = hi_hpc.expanding(min_periods=1).mean()
        feat["HI_HPT_ratio_to_hist"] = (hi_hpt.values / expanding_mean_hpt.values).clip(-10, 10)
        feat["HI_HPC_ratio_to_hist"] = (hi_hpc.values / expanding_mean_hpc.values).clip(-10, 10)

        # Min e max rolling (range di oscillazione recente)
        for w in [20, 50]:
            feat[f"HI_HPT_range_{w}"] = (
                hi_hpt.rolling(w, min_periods=1).max().values -
                hi_hpt.rolling(w, min_periods=1).min().values
            )
            feat[f"HI_HPC_range_{w}"] = (
                hi_hpc.rolling(w, min_periods=1).max().values -
                hi_hpc.rolling(w, min_periods=1).min().values
            )

        # Accelerazione del degradamento (derivata seconda)
        slope_hpt = hi_hpt.rolling(window=20, min_periods=1).apply(get_slope, raw=True)
        slope_hpc = hi_hpc.rolling(window=20, min_periods=1).apply(get_slope, raw=True)
        feat["HI_HPT_acceleration"] = slope_hpt.diff().fillna(0).values
        feat["HI_HPC_acceleration"] = slope_hpc.diff().fillna(0).values

        # Features dai residui dei singoli sensori (trend)
        for var in DEGRAD_VARS:
            feat[f"{var}_slope_20"] = sd[var].rolling(window=20, min_periods=1).apply(get_slope, raw=True).values

        # Ciclo di lavoro (per scale_to_target e gap correction)
        feat["cycle_hpt"] = sd["Cumulative_HPT_SVs"].values
        feat["cycle_hpc"] = sd["Cumulative_HPC_SVs"].values

        feat["ESN"] = esn
        feat["target_hpt"] = sd["Cycles_to_HPT_SV"].values
        feat["target_hpc"] = sd["Cycles_to_HPC_SV"].values

        feat_list.append(feat)

    return pd.concat(feat_list, ignore_index=True)


# ===== PARAMETRI =====
# Finestra per features di trend
GAP_FEATURE_WINDOW = 20

# LightGBM per correzione gap (base = scale_to_target(HI))
GAP_LGBM_PARAMS = dict(
    objective="regression",
    metric="rmse",
    n_estimators=5000,
    learning_rate=0.002,
    max_depth=12,
    num_leaves=63,
    min_child_samples=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)

# Smoothing rolling window per risultato finale
SMOOTHING_WINDOW = 10

# ===== COSTRUZIONE FEATURES =====
regs_data = regress_features(coef_data, window=GAP_FEATURE_WINDOW)
regs_feature_cols = [c for c in regs_data.columns if c not in ["ESN", "target_hpt", "target_hpc"]]
print(f"Features gap correction: {len(regs_feature_cols)}")

y_true_hpt = regs_data["target_hpt"].values
y_true_hpc = regs_data["target_hpc"].values

# ===== SCALE TO TARGET PER CICLO =====
# Scala HI al range di Cycles_to_SV per ogni ciclo di manutenzione
scale_coefs_hpt = {}  # {cycle_id: {"min": t_min, "max": t_max}}
scale_coefs_hpc = {}

base_pred_hpt = np.full(len(regs_data), np.nan)
base_pred_hpc = np.full(len(regs_data), np.nan)

for cycle in sorted(regs_data["cycle_hpt"].unique()):
    mask = regs_data["cycle_hpt"] == cycle
    coefs = {}
    scaled = scale_to_target(
        regs_data.loc[mask, "HI_HPT"],
        regs_data.loc[mask, "target_hpt"],
        coefs,
    )
    base_pred_hpt[mask.values] = scaled.values
    scale_coefs_hpt[int(cycle)] = {"min": coefs["min"][0], "max": coefs["max"][0]}

for cycle in sorted(regs_data["cycle_hpc"].unique()):
    mask = regs_data["cycle_hpc"] == cycle
    coefs = {}
    scaled = scale_to_target(
        regs_data.loc[mask, "HI_HPC"],
        regs_data.loc[mask, "target_hpc"],
        coefs,
    )
    base_pred_hpc[mask.values] = scaled.values
    scale_coefs_hpc[int(cycle)] = {"min": coefs["min"][0], "max": coefs["max"][0]}

print(f"\nBase HPT (scale_to_target) - RMSE: {np.sqrt(np.nanmean((y_true_hpt - base_pred_hpt)**2)):.2f}")
print(f"Base HPC (scale_to_target) - RMSE: {np.sqrt(np.nanmean((y_true_hpc - base_pred_hpc)**2)):.2f}")
print(f"Scale coefs HPT: {scale_coefs_hpt}")
print(f"Scale coefs HPC: {scale_coefs_hpc}")

# ===== GAP: differenza tra scale_to_target e ground truth =====
gap_hpt = y_true_hpt - base_pred_hpt
gap_hpc = y_true_hpc - base_pred_hpc

print(f"\nGap HPT - mean: {np.nanmean(gap_hpt):.2f}, std: {np.nanstd(gap_hpt):.2f}")
print(f"Gap HPC - mean: {np.nanmean(gap_hpc):.2f}, std: {np.nanstd(gap_hpc):.2f}")

# ===== LGBM GAP CORRECTION =====
X_reg = regs_data[regs_feature_cols].values
groups_reg = regs_data["ESN"].values

lgbm_gap_hpt = lgbm.LGBMRegressor(**GAP_LGBM_PARAMS)
lgbm_gap_hpc = lgbm.LGBMRegressor(**GAP_LGBM_PARAMS)

# ===== VALIDAZIONE LEAVE-ONE-ENGINE-OUT =====
logo_reg = LeaveOneGroupOut()

print("\n=== Gap Correction (scale_to_target) — Leave-One-Engine-Out ===")
for train_idx, test_idx in logo_reg.split(X_reg, gap_hpt, groups_reg):
    test_esn = groups_reg[test_idx[0]]

    # HPT
    lgbm_gap_hpt.fit(X_reg[train_idx], gap_hpt[train_idx])
    gap_pred_hpt = lgbm_gap_hpt.predict(X_reg[test_idx])
    final_pred_hpt = base_pred_hpt[test_idx] + gap_pred_hpt
    rmse_hpt = np.sqrt(np.mean((y_true_hpt[test_idx] - final_pred_hpt)**2))

    # HPC
    lgbm_gap_hpc.fit(X_reg[train_idx], gap_hpc[train_idx])
    gap_pred_hpc = lgbm_gap_hpc.predict(X_reg[test_idx])
    final_pred_hpc = base_pred_hpc[test_idx] + gap_pred_hpc
    rmse_hpc = np.sqrt(np.mean((y_true_hpc[test_idx] - final_pred_hpc)**2))

    # Confronto con base
    rmse_base_hpt = np.sqrt(np.mean((y_true_hpt[test_idx] - base_pred_hpt[test_idx])**2))
    rmse_base_hpc = np.sqrt(np.mean((y_true_hpc[test_idx] - base_pred_hpc[test_idx])**2))

    print(f"ESN {test_esn}:")
    print(f"  HPT: base RMSE={rmse_base_hpt:.2f} → corrected RMSE={rmse_hpt:.2f}")
    print(f"  HPC: base RMSE={rmse_base_hpc:.2f} → corrected RMSE={rmse_hpc:.2f}")

# ===== TRAINING FINALE SU TUTTI I DATI =====
lgbm_gap_hpt.fit(X_reg, gap_hpt)
lgbm_gap_hpc.fit(X_reg, gap_hpc)
print("\nLGBM gap regressors addestrati su tutti i motori.")


# ===== AGGIORNA predict_cycles_to_sv PER USARE V2 =====

def predict_cycles_to_sv_v2(engine_df, engine_residuals, esn, window=GAP_FEATURE_WINDOW):
    """
    Pipeline v2: scale_to_target(HI) + LightGBM gap correction + smoothing.
    """
    sd = engine_df.copy()
    sd[DEGRAD_VARS] = engine_residuals[DEGRAD_VARS].values

    if "Cycles" in sd.columns and "Cycles_Since_New" not in sd.columns:
        sd = sd.rename(columns={"Cycles": "Cycles_Since_New"})

    sd = sd.sort_values("Cycles_Since_New")

    # Coefficienti HI
    if SEPARATE_COEFS and isinstance(chpt, dict):
        ahpt_local = chpt.get(str(esn), np.median([v[0] if isinstance(v, np.ndarray) else v for v in chpt.values()]))
        ahpc_local = chpc.get(str(esn), np.median([v[0] if isinstance(v, np.ndarray) else v for v in chpc.values()]))
    else:
        ahpt_local = chpt if not isinstance(chpt, dict) else np.median(list(chpt.values()))
        ahpc_local = chpc if not isinstance(chpc, dict) else np.median(list(chpc.values()))

    if USE_ALL_VARS:
        hi_hpt = HIE(ahpt_local, sd[target_vars])
        hi_hpc = HIE(ahpc_local, sd[target_vars])
    else:
        hi_hpt = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpt_local)
        hi_hpc = HI(sd["Sensed_T3"], sd["Sensed_T45"], ahpc_local)

    # Build features (stesse di regress_features)
    feat = sd[DEGRAD_VARS].copy()
    feat["HI_HPT"] = hi_hpt.values
    feat["HI_HPC"] = hi_hpc.values

    for w in [10, 20, 50]:
        feat[f"HI_HPT_slope_{w}"] = hi_hpt.rolling(window=w, min_periods=1).apply(get_slope, raw=True).values
        feat[f"HI_HPC_slope_{w}"] = hi_hpc.rolling(window=w, min_periods=1).apply(get_slope, raw=True).values
        feat[f"HI_HPT_mean_{w}"] = hi_hpt.rolling(window=w, min_periods=1).mean().values
        feat[f"HI_HPC_mean_{w}"] = hi_hpc.rolling(window=w, min_periods=1).mean().values
        feat[f"HI_HPT_std_{w}"] = hi_hpt.rolling(window=w, min_periods=2).std().fillna(0).values
        feat[f"HI_HPC_std_{w}"] = hi_hpc.rolling(window=w, min_periods=2).std().fillna(0).values

    feat["HI_HPT_delta_from_start"] = hi_hpt.values - hi_hpt.values[0]
    feat["HI_HPC_delta_from_start"] = hi_hpc.values - hi_hpc.values[0]
    feat["HI_HPT_cumulative_change"] = hi_hpt.diff().fillna(0).cumsum().values
    feat["HI_HPC_cumulative_change"] = hi_hpc.diff().fillna(0).cumsum().values

    expanding_mean_hpt = hi_hpt.expanding(min_periods=1).mean()
    expanding_mean_hpc = hi_hpc.expanding(min_periods=1).mean()
    feat["HI_HPT_ratio_to_hist"] = (hi_hpt.values / expanding_mean_hpt.values).clip(-10, 10)
    feat["HI_HPC_ratio_to_hist"] = (hi_hpc.values / expanding_mean_hpc.values).clip(-10, 10)

    for w in [20, 50]:
        feat[f"HI_HPT_range_{w}"] = hi_hpt.rolling(w, min_periods=1).max().values - hi_hpt.rolling(w, min_periods=1).min().values
        feat[f"HI_HPC_range_{w}"] = hi_hpc.rolling(w, min_periods=1).max().values - hi_hpc.rolling(w, min_periods=1).min().values

    slope_hpt = hi_hpt.rolling(window=20, min_periods=1).apply(get_slope, raw=True)
    slope_hpc = hi_hpc.rolling(window=20, min_periods=1).apply(get_slope, raw=True)
    feat["HI_HPT_acceleration"] = slope_hpt.diff().fillna(0).values
    feat["HI_HPC_acceleration"] = slope_hpc.diff().fillna(0).values

    for var in DEGRAD_VARS:
        feat[f"{var}_slope_20"] = sd[var].rolling(window=20, min_periods=1).apply(get_slope, raw=True).values

    # Classificazione ciclo — usa HI scalato (stessa scala del training)
    hi_hpt_clf = scale_to_target_test(hi_hpt, clf_scale_coefs_hpt)
    hi_hpc_clf = scale_to_target_test(hi_hpc, clf_scale_coefs_hpc)

    clf_feat_df = sd[DEGRAD_VARS].copy()
    clf_feat_df["HI_HPT"] = hi_hpt_clf.values
    clf_feat_df["HI_HPC"] = hi_hpc_clf.values
    clf_feat_df["HI_HPT_slope"] = hi_hpt_clf.rolling(window=20, min_periods=1).apply(get_slope, raw=True).values
    clf_feat_df["HI_HPC_slope"] = hi_hpc_clf.rolling(window=20, min_periods=1).apply(get_slope, raw=True).values
    clf_feat_df["HI_HPT_rolling_mean"] = hi_hpt_clf.rolling(window=20, min_periods=1).mean().values
    clf_feat_df["HI_HPC_rolling_mean"] = hi_hpc_clf.rolling(window=20, min_periods=1).mean().values

    try:
        clf_feat = clf_feat_df[clf_feature_cols].values
        cycle_hpt_series = clf_hpt.predict(clf_feat)
        cycle_hpc_series = clf_hpc.predict(clf_feat)
        cycle_hpt = int(cycle_hpt_series[-1])
        cycle_hpc = int(cycle_hpc_series[-1])
    except Exception:
        cycle_hpt_series = np.zeros(len(feat), dtype=int)
        cycle_hpc_series = np.zeros(len(feat), dtype=int)
        cycle_hpt = 0
        cycle_hpc = 0

    # Ciclo come feature per il gap LGBM
    feat["cycle_hpt"] = cycle_hpt_series
    feat["cycle_hpc"] = cycle_hpc_series

    X_feat = feat[regs_feature_cols].values

    # Scale to target usando il ciclo predetto
    hpt_key = cycle_hpt if cycle_hpt in scale_coefs_hpt else min(scale_coefs_hpt.keys())
    hpc_key = cycle_hpc if cycle_hpc in scale_coefs_hpc else min(scale_coefs_hpc.keys())

    base_hpt = scale_to_target_test(hi_hpt, scale_coefs_hpt[hpt_key])
    base_hpc = scale_to_target_test(hi_hpc, scale_coefs_hpc[hpc_key])

    # Gap correction
    gap_pred_hpt = lgbm_gap_hpt.predict(X_feat)
    gap_pred_hpc = lgbm_gap_hpc.predict(X_feat)

    pred_hpt = np.clip(base_hpt.values + gap_pred_hpt, 0, None)
    pred_hpc = np.clip(base_hpc.values + gap_pred_hpc, 0, None)

    # Smoothing
    pred_hpt = pd.Series(pred_hpt).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().values
    pred_hpc = pd.Series(pred_hpc).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().values

    return {
        "ESN": esn,
        "Cycles_to_HPT_SV": pred_hpt[-1],
        "Cycles_to_HPC_SV": pred_hpc[-1],
        "HPT_cycle": cycle_hpt,
        "HPC_cycle": cycle_hpc,
        "pred_series_hpt": pred_hpt,
        "pred_series_hpc": pred_hpc,
    }


# Alias per compatibilità con celle successive
reg_data = regs_data


# %%
def plot_results(results_df, dfv, dft, res_dfv, res_dft):
    """
    Plotta i risultati delle predizioni HPT e HPC.
    - Per validation: confronta predizione vs ground truth (se disponibile)
    - Per test: mostra solo le predizioni
    """

    # Separa validation e test dai risultati
    val_esns = set()
    for v in dfv:
        val_esns.update(v["ESN"].unique())
    test_esns = set()
    for t in dft:
        test_esns.update(t["ESN"].unique())

    val_results = results_df[results_df["ESN"].isin(val_esns)].copy()
    test_results = results_df[results_df["ESN"].isin(test_esns)].copy()

    # =========================================================
    # 1. BARPLOT: Predizioni HPT e HPC per ogni ESN
    # =========================================================
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle("Predizioni Cycles to Service Visit", fontsize=18)

    # Validation - HPT
    ax = axs[0, 0]
    if len(val_results) > 0:
        x = np.arange(len(val_results))
        ax.bar(x, val_results["Cycles_to_HPT_SV"], color="tab:blue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(val_results["ESN"].astype(str), rotation=90, fontsize=6)
        ax.set_title("Validation - Cycles to HPT SV")
        ax.set_ylabel("Cycles")
        ax.grid(True, alpha=0.3)

    # Validation - HPC
    ax = axs[0, 1]
    if len(val_results) > 0:
        x = np.arange(len(val_results))
        ax.bar(x, val_results["Cycles_to_HPC_SV"], color="tab:green", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(val_results["ESN"].astype(str), rotation=90, fontsize=6)
        ax.set_title("Validation - Cycles to HPC SV")
        ax.set_ylabel("Cycles")
        ax.grid(True, alpha=0.3)

    # Test - HPT
    ax = axs[1, 0]
    if len(test_results) > 0:
        x = np.arange(len(test_results))
        ax.bar(x, test_results["Cycles_to_HPT_SV"], color="tab:orange", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(test_results["ESN"].astype(str), rotation=90, fontsize=6)
        ax.set_title("Test - Cycles to HPT SV")
        ax.set_ylabel("Cycles")
        ax.grid(True, alpha=0.3)

    # Test - HPC
    ax = axs[1, 1]
    if len(test_results) > 0:
        x = np.arange(len(test_results))
        ax.bar(x, test_results["Cycles_to_HPC_SV"], color="tab:red", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(test_results["ESN"].astype(str), rotation=90, fontsize=6)
        ax.set_title("Test - Cycles to HPC SV")
        ax.set_ylabel("Cycles")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =========================================================
    # 2. DISTRIBUZIONE: Istogramma delle predizioni
    # =========================================================
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Distribuzione Predizioni", fontsize=16)

    ax = axs[0]
    if len(val_results) > 0:
        ax.hist(val_results["Cycles_to_HPT_SV"], bins=20, alpha=0.6, color="tab:blue", label="Val HPT")
        ax.hist(val_results["Cycles_to_HPC_SV"], bins=20, alpha=0.6, color="tab:green", label="Val HPC")
    if len(test_results) > 0:
        ax.hist(test_results["Cycles_to_HPT_SV"], bins=20, alpha=0.4, color="tab:orange", label="Test HPT", linestyle="--")
        ax.hist(test_results["Cycles_to_HPC_SV"], bins=20, alpha=0.4, color="tab:red", label="Test HPC", linestyle="--")
    ax.set_title("Distribuzione Cycles to SV")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Distribuzione cicli di lavoro predetti
    ax = axs[1]
    all_results = pd.concat([val_results, test_results], ignore_index=True)
    if "HPT_cycle" in all_results.columns and "HPC_cycle" in all_results.columns:
        cycles_hpt = all_results["HPT_cycle"].value_counts().sort_index()
        cycles_hpc = all_results["HPC_cycle"].value_counts().sort_index()
        width = 0.35
        x_hpt = np.arange(len(cycles_hpt))
        x_hpc = np.arange(len(cycles_hpc))
        ax.bar(x_hpt - width/2, cycles_hpt.values, width, label="HPT cycle", color="tab:blue", alpha=0.7)
        ax.bar(x_hpc + width/2, cycles_hpc.values, width, label="HPC cycle", color="tab:green", alpha=0.7)
        ax.set_xticks(np.arange(max(len(cycles_hpt), len(cycles_hpc))))
        ax.set_title("Cicli di Lavoro Predetti")
        ax.set_xlabel("Ciclo")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =========================================================
    # 3. PER-ENGINE DETAIL: HI + predizione per ogni motore
    # =========================================================
    def plot_engine_detail(engine_list, res_data, label_prefix, results_sub):
        """Plotta HI e predizione per ogni motore in una lista di dataframe"""
        for engine_df in engine_list:
            for esn in engine_df["ESN"].unique():
                edf = engine_df[engine_df["ESN"] == esn].copy()

                # Calcola residui
                engine_res = _residuals(edf)
                if engine_res is None:
                    continue
                edf_res = edf.copy()
                edf_res[DEGRAD_VARS] = engine_res[DEGRAD_VARS].values

                # Coefficienti
                if SEPARATE_COEFS and isinstance(chpt, dict):
                    ahpt_local = chpt.get(str(esn), np.median([v[0] if isinstance(v, np.ndarray) else v for v in chpt.values()]))
                    ahpc_local = chpc.get(str(esn), np.median([v[0] if isinstance(v, np.ndarray) else v for v in chpc.values()]))
                else:
                    ahpt_local = chpt if not isinstance(chpt, dict) else np.median(list(chpt.values()))
                    ahpc_local = chpc if not isinstance(chpc, dict) else np.median(list(chpc.values()))

                # Health Index
                if USE_ALL_VARS:
                    hi_hpt = HIE(ahpt_local, edf_res[target_vars])
                    hi_hpc = HIE(ahpc_local, edf_res[target_vars])
                else:
                    hi_hpt = HI(edf_res["Sensed_T3"], edf_res["Sensed_T45"], ahpt_local)
                    hi_hpc = HI(edf_res["Sensed_T3"], edf_res["Sensed_T45"], ahpc_local)

                # Prendi predizione
                esn_pred = results_sub[results_sub["ESN"] == esn]

                fig, axs = plt.subplots(1, 2, figsize=(18, 5))
                fig.suptitle(f"{label_prefix} ESN {esn}", fontsize=14)

                # HPT
                ax = axs[0]
                hi_hpt_smooth = hi_hpt.rolling(window=10, min_periods=1).mean()
                ax.plot(hi_hpt_smooth.values, color="tab:blue", linewidth=0.8, label="HI HPT (smoothed)")
                if len(esn_pred) > 0:
                    pred_val = esn_pred["Cycles_to_HPT_SV"].values[0]
                    cycle_val = esn_pred["HPT_cycle"].values[0]
                    ax.axhline(y=hi_hpt_smooth.values[-1], color="red", linestyle="--", alpha=0.5)
                    ax.set_title(f"HPT — Pred: {pred_val:.0f} cycles (ciclo {cycle_val})")
                else:
                    ax.set_title("HPT — Health Index")
                ax.set_xlabel("Osservazione")
                ax.set_ylabel("HI")
                ax.legend(fontsize="small")
                ax.grid(True, alpha=0.3)

                # HPC
                ax = axs[1]
                hi_hpc_smooth = hi_hpc.rolling(window=10, min_periods=1).mean()
                ax.plot(hi_hpc_smooth.values, color="tab:green", linewidth=0.8, label="HI HPC (smoothed)")
                if len(esn_pred) > 0:
                    pred_val = esn_pred["Cycles_to_HPC_SV"].values[0]
                    cycle_val = esn_pred["HPC_cycle"].values[0]
                    ax.axhline(y=hi_hpc_smooth.values[-1], color="red", linestyle="--", alpha=0.5)
                    ax.set_title(f"HPC — Pred: {pred_val:.0f} cycles (ciclo {cycle_val})")
                else:
                    ax.set_title("HPC — Health Index")
                ax.set_xlabel("Osservazione")
                ax.set_ylabel("HI")
                ax.legend(fontsize="small")
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

    print("\n=== DETTAGLIO VALIDATION ===")
    plot_engine_detail(dfv, res_dfv, "Validation", val_results)

    print("\n=== DETTAGLIO TEST ===")
    plot_engine_detail(dft, res_dft, "Test", test_results)

    # =========================================================
    # 4. SUMMARY TABLE
    # =========================================================
    print("\n=== SUMMARY STATISTICHE ===")
    summary = pd.DataFrame({
        "Dataset": ["Validation", "Validation", "Test", "Test"],
        "Component": ["HPT", "HPC", "HPT", "HPC"],
        "Mean": [
            val_results["Cycles_to_HPT_SV"].mean() if len(val_results) else np.nan,
            val_results["Cycles_to_HPC_SV"].mean() if len(val_results) else np.nan,
            test_results["Cycles_to_HPT_SV"].mean() if len(test_results) else np.nan,
            test_results["Cycles_to_HPC_SV"].mean() if len(test_results) else np.nan,
        ],
        "Std": [
            val_results["Cycles_to_HPT_SV"].std() if len(val_results) else np.nan,
            val_results["Cycles_to_HPC_SV"].std() if len(val_results) else np.nan,
            test_results["Cycles_to_HPT_SV"].std() if len(test_results) else np.nan,
            test_results["Cycles_to_HPC_SV"].std() if len(test_results) else np.nan,
        ],
        "Min": [
            val_results["Cycles_to_HPT_SV"].min() if len(val_results) else np.nan,
            val_results["Cycles_to_HPC_SV"].min() if len(val_results) else np.nan,
            test_results["Cycles_to_HPT_SV"].min() if len(test_results) else np.nan,
            test_results["Cycles_to_HPC_SV"].min() if len(test_results) else np.nan,
        ],
        "Max": [
            val_results["Cycles_to_HPT_SV"].max() if len(val_results) else np.nan,
            val_results["Cycles_to_HPC_SV"].max() if len(val_results) else np.nan,
            test_results["Cycles_to_HPT_SV"].max() if len(test_results) else np.nan,
            test_results["Cycles_to_HPC_SV"].max() if len(test_results) else np.nan,
        ],
    })
    print(summary.to_string(index=False))



# %%
def plot_training_before_after(coef_data, reg_data, base_pred_hpt_all, base_pred_hpc_all, lgbm_gap_hpt, lgbm_gap_hpc, reg_feature_cols):
    """
    Plotta il confronto prima/dopo l'applicazione del modello sui dati di training.
    Per ogni ESN mostra:
    - Cycles_to_SV reale (ground truth)
    - Predizione base (scale_to_target su HI)
    - Predizione corretta (base + LightGBM gap correction)
    """

    for esn in coef_data["ESN"].unique():
        esn_mask = reg_data["ESN"] == esn
        esn_reg = reg_data[esn_mask].copy()

        y_true_hpt = esn_reg["target_hpt"].values
        y_true_hpc = esn_reg["target_hpc"].values

        # Base prediction (scale_to_target, già calcolate)
        base_pred_hpt = pd.DataFrame(base_pred_hpt_all[esn_mask.values]).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().values.reshape(-1)  
        base_pred_hpc = pd.DataFrame(base_pred_hpc_all[esn_mask.values]).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().values.reshape(-1)

        # Gap correction
        X_feat = esn_reg[reg_feature_cols].values
        gap_pred_hpt = lgbm_gap_hpt.predict(X_feat)
        gap_pred_hpc = lgbm_gap_hpc.predict(X_feat)
        final_pred_hpt = base_pred_hpt + gap_pred_hpt
        final_pred_hpc = base_pred_hpc + gap_pred_hpc
        final_pred_hpt = pd.DataFrame(final_pred_hpt).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().values.reshape(-1)  
        final_pred_hpc = pd.DataFrame(final_pred_hpc).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().values.reshape(-1)

        # Metriche
        rmse_base_hpt = np.sqrt(np.mean((y_true_hpt - base_pred_hpt) ** 2))
        rmse_final_hpt = np.sqrt(np.mean((y_true_hpt - final_pred_hpt) ** 2))
        rmse_base_hpc = np.sqrt(np.mean((y_true_hpc - base_pred_hpc) ** 2))
        rmse_final_hpc = np.sqrt(np.mean((y_true_hpc - final_pred_hpc) ** 2))

        mae_base_hpt = np.mean(np.abs(y_true_hpt - base_pred_hpt))
        mae_final_hpt = np.mean(np.abs(y_true_hpt - final_pred_hpt))
        mae_base_hpc = np.mean(np.abs(y_true_hpc - base_pred_hpc))
        mae_final_hpc = np.mean(np.abs(y_true_hpc - final_pred_hpc))

        x_axis = np.arange(len(y_true_hpt))

        fig, axs = plt.subplots(2, 3, figsize=(24, 10))
        fig.suptitle(f"Training ESN {esn} — Prima vs Dopo Correzione", fontsize=16)

        # ---- ROW 1: HPT ----

        # 1a. Time series: ground truth vs base vs corrected
        ax = axs[0, 0]
        ax.plot(x_axis, y_true_hpt, color="black", linewidth=1.0, alpha=0.8, label="Ground Truth")
        ax.plot(x_axis, base_pred_hpt, color="tab:orange", linewidth=0.8, alpha=0.7, linestyle="--", label=f"Base (RMSE={rmse_base_hpt:.1f})")
        ax.plot(x_axis, final_pred_hpt, color="tab:blue", linewidth=0.8, alpha=0.9, label=f"Corrected (RMSE={rmse_final_hpt:.1f})")
        ax.set_title("HPT — Cycles to SV: Predizione vs Realtà")
        ax.set_xlabel("Osservazione")
        ax.set_ylabel("Cycles to HPT SV")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

        # 1b. Errore residuo prima e dopo
        ax = axs[0, 1]
        error_base_hpt = y_true_hpt - base_pred_hpt
        error_final_hpt = y_true_hpt - final_pred_hpt
        ax.plot(x_axis, error_base_hpt, color="tab:orange", linewidth=0.6, alpha=0.6, label=f"Base error (MAE={mae_base_hpt:.1f})")
        ax.plot(x_axis, error_final_hpt, color="tab:blue", linewidth=0.6, alpha=0.8, label=f"Corrected error (MAE={mae_final_hpt:.1f})")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.fill_between(x_axis, error_final_hpt, 0, alpha=0.15, color="tab:blue")
        ax.set_title("HPT — Errore (Truth - Pred)")
        ax.set_xlabel("Osservazione")
        ax.set_ylabel("Errore")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

        # 1c. Scatter: predizione vs ground truth
        ax = axs[0, 2]
        lim_hpt = [min(y_true_hpt.min(), final_pred_hpt.min(), base_pred_hpt.min()) - 10,
                    max(y_true_hpt.max(), final_pred_hpt.max(), base_pred_hpt.max()) + 10]
        ax.scatter(y_true_hpt, base_pred_hpt, color="tab:orange", alpha=0.3, s=8, label="Base")
        ax.scatter(y_true_hpt, final_pred_hpt, color="tab:blue", alpha=0.3, s=8, label="Corrected")
        ax.plot(lim_hpt, lim_hpt, color="red", linestyle="--", linewidth=1, label="Perfetto")
        ax.set_xlim(lim_hpt)
        ax.set_ylim(lim_hpt)
        ax.set_title("HPT — Scatter Pred vs Truth")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predizione")
        ax.legend(fontsize="small")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # ---- ROW 2: HPC ----

        # 2a. Time series
        ax = axs[1, 0]
        ax.plot(x_axis, y_true_hpc, color="black", linewidth=1.0, alpha=0.8, label="Ground Truth")
        ax.plot(x_axis, base_pred_hpc, color="tab:orange", linewidth=0.8, alpha=0.7, linestyle="--", label=f"Base (RMSE={rmse_base_hpc:.1f})")
        ax.plot(x_axis, final_pred_hpc, color="tab:green", linewidth=0.8, alpha=0.9, label=f"Corrected (RMSE={rmse_final_hpc:.1f})")
        ax.set_title("HPC — Cycles to SV: Predizione vs Realtà")
        ax.set_xlabel("Osservazione")
        ax.set_ylabel("Cycles to HPC SV")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

        # 2b. Errore residuo
        ax = axs[1, 1]
        error_base_hpc = y_true_hpc - base_pred_hpc
        error_final_hpc = y_true_hpc - final_pred_hpc
        ax.plot(x_axis, error_base_hpc, color="tab:orange", linewidth=0.6, alpha=0.6, label=f"Base error (MAE={mae_base_hpc:.1f})")
        ax.plot(x_axis, error_final_hpc, color="tab:green", linewidth=0.6, alpha=0.8, label=f"Corrected error (MAE={mae_final_hpc:.1f})")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.fill_between(x_axis, error_final_hpc, 0, alpha=0.15, color="tab:green")
        ax.set_title("HPC — Errore (Truth - Pred)")
        ax.set_xlabel("Osservazione")
        ax.set_ylabel("Errore")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

        # 2c. Scatter
        ax = axs[1, 2]
        lim_hpc = [min(y_true_hpc.min(), final_pred_hpc.min(), base_pred_hpc.min()) - 10,
                    max(y_true_hpc.max(), final_pred_hpc.max(), base_pred_hpc.max()) + 10]
        ax.scatter(y_true_hpc, base_pred_hpc, color="tab:orange", alpha=0.3, s=8, label="Base")
        ax.scatter(y_true_hpc, final_pred_hpc, color="tab:green", alpha=0.3, s=8, label="Corrected")
        ax.plot(lim_hpc, lim_hpc, color="red", linestyle="--", linewidth=1, label="Perfetto")
        ax.set_xlim(lim_hpc)
        ax.set_ylim(lim_hpc)
        ax.set_title("HPC — Scatter Pred vs Truth")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predizione")
        ax.legend(fontsize="small")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Stampa metriche
        print(f"ESN {esn}:")
        print(f"  HPT  Base → RMSE={rmse_base_hpt:.2f}  MAE={mae_base_hpt:.2f}")
        print(f"  HPT  Corr → RMSE={rmse_final_hpt:.2f}  MAE={mae_final_hpt:.2f}  (Δ RMSE={rmse_base_hpt - rmse_final_hpt:+.2f})")
        print(f"  HPC  Base → RMSE={rmse_base_hpc:.2f}  MAE={mae_base_hpc:.2f}")
        print(f"  HPC  Corr → RMSE={rmse_final_hpc:.2f}  MAE={mae_final_hpc:.2f}  (Δ RMSE={rmse_base_hpc - rmse_final_hpc:+.2f})")
        print()

    # =========================================================
    # SUMMARY GLOBALE
    # =========================================================
    y_all_true_hpt = reg_data["target_hpt"].values
    y_all_true_hpc = reg_data["target_hpc"].values
    all_base_hpt = base_pred_hpt_all
    all_base_hpc = base_pred_hpc_all
    all_gap_hpt = lgbm_gap_hpt.predict(reg_data[reg_feature_cols].values)
    all_gap_hpc = lgbm_gap_hpc.predict(reg_data[reg_feature_cols].values)
    all_final_hpt = all_base_hpt + all_gap_hpt
    all_final_hpc = all_base_hpc + all_gap_hpc

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training Set Globale — Distribuzione Errore", fontsize=16)

    ax = axs[0]
    ax.hist(y_all_true_hpt - all_base_hpt, bins=50, alpha=0.5, color="tab:orange", label="Base error")
    ax.hist(y_all_true_hpt - all_final_hpt, bins=50, alpha=0.5, color="tab:blue", label="Corrected error")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title("HPT — Distribuzione Errore")
    ax.set_xlabel("Errore (Truth - Pred)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axs[1]
    ax.hist(y_all_true_hpc - all_base_hpc, bins=50, alpha=0.5, color="tab:orange", label="Base error")
    ax.hist(y_all_true_hpc - all_final_hpc, bins=50, alpha=0.5, color="tab:green", label="Corrected error")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title("HPC — Distribuzione Errore")
    ax.set_xlabel("Errore (Truth - Pred)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("=== METRICHE GLOBALI ===")
    print(f"HPT  Base    RMSE={np.sqrt(np.mean((y_all_true_hpt - all_base_hpt)**2)):.2f}  MAE={np.mean(np.abs(y_all_true_hpt - all_base_hpt)):.2f}")
    print(f"HPT  Corrected RMSE={np.sqrt(np.mean((y_all_true_hpt - all_final_hpt)**2)):.2f}  MAE={np.mean(np.abs(y_all_true_hpt - all_final_hpt)):.2f}")
    print(f"HPC  Base    RMSE={np.sqrt(np.mean((y_all_true_hpc - all_base_hpc)**2)):.2f}  MAE={np.mean(np.abs(y_all_true_hpc - all_base_hpc)):.2f}")
    print(f"HPC  Corrected RMSE={np.sqrt(np.mean((y_all_true_hpc - all_final_hpc)**2)):.2f}  MAE={np.mean(np.abs(y_all_true_hpc - all_final_hpc)):.2f}")


# ===== ESECUZIONE =====
plot_training_before_after(coef_data, regs_data, base_pred_hpt, base_pred_hpc, lgbm_gap_hpt, lgbm_gap_hpc, regs_feature_cols)


# %%
# ===== CELLA NUOVA: INFERENZA ROBUSTA SU VAL/TEST =====

def get_median_coefs():
    """Calcola i coefficienti mediani dal training per motori sconosciuti."""
    if isinstance(chpt, dict):
        hpt_vals = [v[0] if isinstance(v, np.ndarray) else v for v in chpt.values()]
        hpc_vals = [v[0] if isinstance(v, np.ndarray) else v for v in chpc.values()]
        return np.median(hpt_vals), np.median(hpc_vals)
    else:
        # chpt/chpc sono già scalari (SEPARATE_COEFS = False)
        ahpt = chpt[0] if isinstance(chpt, np.ndarray) else float(chpt)
        ahpc = chpc[0] if isinstance(chpc, np.ndarray) else float(chpc)
        return ahpt, ahpc


def predict_all_engines(engine_list, res_data_list=None):
    """
    Predice Cycles_to_SV per tutti i motori in una lista di dataframe.
    Più robusto: gestisce coefficienti mancanti, clipping ragionevole,
    e fallback a statistiche di training.
    """
    # Range ragionevoli dalle statistiche di training
    # (i target nel training hanno un certo range — le predizioni non dovrebbero
    #  uscire troppo da lì)
    train_max_hpt = regs_data["target_hpt"].max()
    train_max_hpc = regs_data["target_hpc"].max()
    train_mean_hpt = regs_data["target_hpt"].mean()
    train_mean_hpc = regs_data["target_hpc"].mean()

    print(f"Training range HPT: 0 — {train_max_hpt:.0f} (mean={train_mean_hpt:.0f})")
    print(f"Training range HPC: 0 — {train_max_hpc:.0f} (mean={train_mean_hpc:.0f})")

    # Margine: le predizioni possono eccedere il max di training del 30%
    clip_max_hpt = train_max_hpt * 1.3
    clip_max_hpc = train_max_hpc * 1.3

    results = []
    median_ahpt, median_ahpc = get_median_coefs()

    for engine_df in engine_list:
        for esn in engine_df["ESN"].unique():
            edf = engine_df[engine_df["ESN"] == esn].copy()

            # Calcola residui
            engine_res = _residuals(edf)
            if engine_res is None:
                print(f"  ESN {esn}: SKIP (residui None)")
                results.append({
                    "ESN": esn,
                    "Cycles_to_HPT_SV": train_mean_hpt,
                    "Cycles_to_HPC_SV": train_mean_hpc,
                    "HPT_cycle": -1,
                    "HPC_cycle": -1,
                    "confidence": "fallback",
                })
                continue

            try:
                pred = predict_cycles_to_sv_v2(edf, engine_res, esn)
                pred_hpt = np.clip(pred["Cycles_to_HPT_SV"], 0, clip_max_hpt)
                pred_hpc = np.clip(pred["Cycles_to_HPC_SV"], 0, clip_max_hpc)

                # # se la predizione è esattamente 0, probabilmente
                # # il modello ha fallito — usa la mediana della serie predetta
                # if pred_hpt == 0 and len(pred["pred_series_hpt"]) > 10:
                #     # Usa la mediana degli ultimi 20 punti della serie
                #     last_preds = pred["pred_series_hpt"][-20:]
                #     pred_hpt = np.clip(np.median(last_preds[last_preds > 0]) if np.any(last_preds > 0) else train_mean_hpt, 0, clip_max_hpt)

                # if pred_hpc == 0 and len(pred["pred_series_hpc"]) > 10:
                #     last_preds = pred["pred_series_hpc"][-20:]
                #     pred_hpc = np.clip(np.median(last_preds[last_preds > 0]) if np.any(last_preds > 0) else train_mean_hpc, 0, clip_max_hpc)

                results.append({
                    "ESN": esn,
                    "Cycles_to_HPT_SV": pred_hpt,
                    "Cycles_to_HPC_SV": pred_hpc,
                    "HPT_cycle": pred["HPT_cycle"],
                    "HPC_cycle": pred["HPC_cycle"],
                    "confidence": "ok",
                })

            except Exception as ex:
                print(f"  ESN {esn}: ERRORE ({ex}) — fallback a media training")
                results.append({
                    "ESN": esn,
                    "Cycles_to_HPT_SV": train_mean_hpt,
                    "Cycles_to_HPC_SV": train_mean_hpc,
                    "HPT_cycle": -1,
                    "HPC_cycle": -1,
                    "confidence": "error_fallback",
                })

    return pd.DataFrame(results)


print("=== PREDIZIONE VALIDATION ===")
results_val = predict_all_engines(dfv)
print("\n=== PREDIZIONE TEST ===")
results_test = predict_all_engines(dft)
results_df = pd.concat([results_val, results_test], ignore_index=True)
print("\n=== QUALITY CHECK ===")
for label, rdf in [("Validation", results_val), ("Test", results_test)]:
    n_ok = (rdf["confidence"] == "ok").sum()
    n_fallback = (rdf["confidence"] != "ok").sum()
    print(f"{label}: {n_ok} ok, {n_fallback} fallback")
    print(f"  HPT: mean={rdf['Cycles_to_HPT_SV'].mean():.0f}, std={rdf['Cycles_to_HPT_SV'].std():.0f}, "
          f"min={rdf['Cycles_to_HPT_SV'].min():.0f}, max={rdf['Cycles_to_HPT_SV'].max():.0f}")
    print(f"  HPC: mean={rdf['Cycles_to_HPC_SV'].mean():.0f}, std={rdf['Cycles_to_HPC_SV'].std():.0f}, "
          f"min={rdf['Cycles_to_HPC_SV'].min():.0f}, max={rdf['Cycles_to_HPC_SV'].max():.0f}")

plot_results(results_df, dfv, dft, res_dfv, res_dft)

# %% [markdown]
# # WW
# per il ww bisogna fare una cosa diversa. Intanto bisogna "normalizzare" la salita, ovvero eliminare gli effetti delle manutenzioni hpc e hpt sui residui di T45_res

# %%
# WW - Predizione prossimVGo evento Water Wash

from math import e


def remove_effect(df, col):
    """Rimuove i jump di Sensed_T45 causati da manutenzioni (HPT/HPC SV)
    usando le colonne Cumulative_*_SVs disponibili nel training."""
    df = df.sort_values([col, "Cycles_Since_New", "Snapshot"]).copy()
    df = pp.remove_outliers(df, threshold=2.6, sensor_cols=["Sensed_T45"])
    df = df.groupby(["Cycles_Since_New"], as_index=False).median(numeric_only=True)
    grp_stats = df.groupby(col)["Sensed_T45"].agg(["first", "last"])
    grp_stats = grp_stats.sort_index()
    grp_stats["prev_last"] = grp_stats["last"].shift(1)
    grp_stats["jump"] = grp_stats["first"] - grp_stats["prev_last"]
    grp_stats["jump"] = grp_stats["jump"].fillna(0)
    grp_stats["cumulative_offset"] = grp_stats["jump"].cumsum()
    offset_map = grp_stats["cumulative_offset"].to_dict()
    df["Sensed_T45"] = df["Sensed_T45"] - df[col].map(offset_map)
    return df


def get_events(df, soglia=10):
    """Rileva eventi di manutenzione come salti improvvisi nell'HI."""
    hi_hpt, hi_hpc = calc_hi(df)
    cond_hpt = (hi_hpt.diff(1) > soglia) & (hi_hpt.diff(2) > soglia) & (hi_hpt.diff(3) > soglia)
    eventi_hpt = hi_hpt[cond_hpt]

    cond_hpc = (hi_hpc.diff(1) > soglia) & (hi_hpc.diff(2) > soglia) & (hi_hpc.diff(3) > soglia)
    eventi_hpc = hi_hpc[cond_hpc]

    return eventi_hpt, eventi_hpc


def remove_effect_hard_core(df):
    """Rimuove i jump di Sensed_T45 per dati val/test dove non abbiamo
    le colonne Cumulative_*_SVs. Rileva eventi tramite HI."""
    df = df.sort_values(["Cycles_Since_New", "Snapshot"]).copy()
    df = pp.remove_outliers(df, threshold=2.6, sensor_cols=["Sensed_T45"])

    # Rolling solo su colonne numeriche
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].rolling(window=5, min_periods=1).median()

    ehpt, ehpc = get_events(df)
    df["Event_Group"] = 0
    df.loc[ehpt.index, "Event_Group"] = 1
    df["Event_Group"] = df["Event_Group"].cumsum()

    grp_stats = df.groupby("Event_Group")["Sensed_T45"].agg(["first", "last"])
    grp_stats["prev_last"] = grp_stats["last"].shift(1)
    grp_stats["jump"] = grp_stats["first"] - grp_stats["prev_last"]
    grp_stats["jump"] = grp_stats["jump"].fillna(0)
    grp_stats["cumulative_offset"] = grp_stats["jump"].cumsum()
    offset_map = grp_stats["cumulative_offset"].to_dict()
    df["Sensed_T45"] = df["Sensed_T45"] - df["Event_Group"].map(offset_map)
    df = df.drop(columns=["Event_Group"])
    return df


def detect_ww_events(t45_series, slope, window=11, factor_mult=80):
    """
    Rileva eventi WW basandosi sulla deviazione della media mobile
    rispetto al trend lineare atteso.

    Args:
        t45_series: pd.Series di Sensed_T45
        slope: pendenza del trend lineare globale
        window: finestra per media mobile
        factor_mult: moltiplicatore per la soglia (slope * e^2 * factor_mult)

    Returns:
        dict {index: valore_t45} degli eventi rilevati
    """
    factor = (e ** 2) * factor_mult
    sv = {}
    counter = []
    reference_r = None
    floor = None

    for idx, val_t45 in t45_series.items():
        if reference_r is None:
            reference_r = val_t45
            floor = val_t45
            counter.append(val_t45)
            continue

        counter.append(val_t45)
        if len(counter) < window:
            continue

        mc = np.mean(counter[-window:]) - floor
        msp = reference_r - floor

        if (mc - msp) > slope * factor:
            sv[idx] = val_t45
            reference_r = val_t45
            floor = val_t45

    return sv


def predict_ww(engine_df, engine_res, esn, window=11, factor_mult=80):
    """
    Pipeline completa di predizione WW per un singolo motore.

    1. Sostituisce i sensori con i residui
    2. Rimuove gli effetti delle manutenzioni HPT/HPC su T45
    3. Fitta un trend lineare
    4. Rileva dove T45 devia abbastanza dal trend → evento WW

    Args:
        engine_df: DataFrame del motore (originale)
        engine_res: DataFrame dei residui
        esn: Engine Serial Number
        window: finestra media mobile per detection
        factor_mult: sensibilità della detection

    Returns:
        dict con:
            - 'esn': ESN del motore
            - 'detected_events': dict {index: valore}
            - 'n_events': numero eventi rilevati
            - 'slope': pendenza trend lineare
            - 'wwdf': DataFrame processato (per plotting)
            - 'regression': modello LinearRegression fittato
    """
    wwdf = engine_df.copy()
    wwdf[DEGRAD_VARS] = engine_res[DEGRAD_VARS].values

    # Normalizza nome colonna cicli
    if "Cycles" in wwdf.columns and "Cycles_Since_New" not in wwdf.columns:
        wwdf = wwdf.rename(columns={"Cycles": "Cycles_Since_New"})

    # Determina se è training (ha colonne Cumulative_*)
    is_training = "Cumulative_HPT_SVs" in wwdf.columns and "Cumulative_HPC_SVs" in wwdf.columns

    # Rimuovi effetti manutenzione su T45
    if is_training:
        wwdf = remove_effect(wwdf, "Cumulative_HPT_SVs")
        wwdf = remove_effect(wwdf, "Cumulative_HPC_SVs")
    else:
        wwdf = remove_effect_hard_core(wwdf)

    wwdf = wwdf.dropna()

    if len(wwdf) == 0:
        return {
            "esn": esn,
            "detected_events": {},
            "n_events": 0,
            "slope": 0.0,
            "wwdf": wwdf,
            "regression": None,
            "is_training": is_training,
        }

    # Trend lineare globale
    X = wwdf["Cycles_Since_New"].values.reshape(-1, 1)
    Y = wwdf["Sensed_T45"].values
    reg = LinearRegression().fit(X, Y)
    slope = reg.coef_[0]

    # Rileva eventi WW
    if is_training:
        # Nel training, resettiamo reference quando cambia Cumulative_WWs
        sv = {}
        counter = []
        reference_r = None
        floor = None
        last_cum = 0
        factor = (e ** 2) * factor_mult

        for idx, row in wwdf[["Sensed_T45", "Cumulative_WWs"]].iterrows():
            val_t45 = row["Sensed_T45"]
            val_cum = row["Cumulative_WWs"]

            if reference_r is None:
                reference_r = val_t45
                floor = val_t45
                counter.append(val_t45)
                continue

            counter.append(val_t45)
            if len(counter) < window:
                continue

            mc = np.mean(counter[-window:]) - floor
            msp = reference_r - floor

            # Reset al cambio di ciclo WW reale
            if val_cum > last_cum:
                last_cum = val_cum
                reference_r = val_t45
                floor = val_t45

            if (mc - msp) > slope * factor:
                sv[idx] = val_t45
                reference_r = val_t45
                floor = val_t45
    else:
        sv = detect_ww_events(wwdf["Sensed_T45"], slope, window, factor_mult)

    return {
        "esn": esn,
        "detected_events": sv,
        "n_events": len(sv),
        "slope": slope,
        "wwdf": wwdf,
        "regression": reg,
        "is_training": is_training,
    }


def plot_ww_prediction(ww_result):
    """
    Plotta il risultato della predizione WW per un motore.

    Args:
        ww_result: output di predict_ww()
    """
    esn = ww_result["esn"]
    sv = ww_result["detected_events"]
    wwdf = ww_result["wwdf"]
    reg = ww_result["regression"]
    slope = ww_result["slope"]
    is_training = ww_result["is_training"]

    if len(wwdf) == 0 or reg is None:
        print(f"ESN {esn}: nessun dato disponibile per il plot WW")
        return

    X = wwdf["Cycles_Since_New"].values.reshape(-1, 1)

    fig, axs = plt.subplots(1, 2, figsize=(22, 6))
    fig.suptitle(f"WW Prediction — ESN {esn}", fontsize=16)

    # ---- Plot 1: T45 con eventi rilevati ----
    ax = axs[0]
    ax.scatter(wwdf["Cycles_Since_New"], wwdf["Sensed_T45"],
               color="tab:blue", alpha=0.6, s=3, label="T45 (residui)", zorder=2)
    ax.plot(wwdf["Cycles_Since_New"], reg.predict(X),
            color="red", linewidth=2, label=f"Trend (slope={slope:.6f})", zorder=3)

    if len(sv) > 0:
        sv_cycles = wwdf.loc[list(sv.keys()), "Cycles_Since_New"]
        ax.vlines(x=sv_cycles,
                  ymin=wwdf["Sensed_T45"].min(), ymax=wwdf["Sensed_T45"].max(),
                  colors="green", linestyles="dashed", alpha=0.7,
                  label=f"Detected WW ({len(sv)})", zorder=1)

    if is_training and "Cumulative_WWs" in wwdf.columns:
        ww_boundaries = wwdf["Cycles_Since_New"].groupby(wwdf["Cumulative_WWs"]).last()
        n_real = len(wwdf["Cumulative_WWs"].unique())
        ax.vlines(x=ww_boundaries,
                  ymin=wwdf["Sensed_T45"].min(), ymax=wwdf["Sensed_T45"].max(),
                  colors="gray", linestyles="dotted", alpha=0.5,
                  label=f"Real WW boundaries ({n_real})", zorder=1)

    ax.set_title("T45 Residuals + WW Detection")
    ax.set_xlabel("Cycles Since New")
    ax.set_ylabel("Sensed T45 (residui)")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: Detrended T45 (rimuovi trend lineare) ----
    ax = axs[1]
    detrended = wwdf["Sensed_T45"].values - reg.predict(X).flatten()
    ax.scatter(wwdf["Cycles_Since_New"], detrended,
               color="tab:purple", alpha=0.6, s=3, label="T45 detrended", zorder=2)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")

    if len(sv) > 0:
        sv_cycles = wwdf.loc[list(sv.keys()), "Cycles_Since_New"]
        sv_detrended = [detrended[wwdf["Cycles_Since_New"].values == c][0]
                        if len(detrended[wwdf["Cycles_Since_New"].values == c]) > 0 else 0
                        for c in sv_cycles]
        ax.scatter(sv_cycles, sv_detrended,
                   color="green", s=50, marker="v", zorder=3,
                   label=f"WW events ({len(sv)})", edgecolors="black", linewidth=0.5)

    # Rolling mean per visualizzare l'accumulo
    rolling_mean = pd.Series(detrended).rolling(window=20, min_periods=1).mean()
    ax.plot(wwdf["Cycles_Since_New"].values, rolling_mean.values,
            color="tab:orange", linewidth=1.5, alpha=0.8, label="Rolling mean (20)")

    ax.set_title("T45 Detrended — Accumulo Fouling")
    ax.set_xlabel("Cycles Since New")
    ax.set_ylabel("T45 - Trend")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Stampa summary
    if is_training and "Cumulative_WWs" in wwdf.columns:
        n_real = len(wwdf["Cumulative_WWs"].unique())
        print(f"  ESN {esn}: Detected={len(sv)}, Real WW cycles={n_real}, Slope={slope:.6f}")
    else:
        print(f"  ESN {esn}: Detected={len(sv)}, Slope={slope:.6f}")



# %%
# ===== ESECUZIONE =====

# --- Training ---
print("=== WW PREDICTION — TRAINING ===")
ww_results_train = {}
for esn in df["ESN"].unique():
    engine_df = df[df["ESN"] == esn]
    engine_res = _residuals(engine_df)
    if engine_res is None:
        continue
    ww_result = predict_ww(engine_df, engine_res, esn)
    ww_results_train[esn] = ww_result
    plot_ww_prediction(ww_result)

# --- Validation ---
print("\n=== WW PREDICTION — VALIDATION ===")
ww_results_val = {}
for v in dfv:
    for esn in v["ESN"].unique():
        engine_df = v[v["ESN"] == esn]
        engine_res = _residuals(engine_df)
        if engine_res is None:
            continue
        ww_result = predict_ww(engine_df, engine_res, esn)
        ww_results_val[esn] = ww_result
        plot_ww_prediction(ww_result)

# --- Test ---
print("\n=== WW PREDICTION — TEST ===")
ww_results_test = {}
for t in dft:
    for esn in t["ESN"].unique():
        engine_df = t[t["ESN"] == esn]
        engine_res = _residuals(engine_df)
        if engine_res is None:
            continue
        ww_result = predict_ww(engine_df, engine_res, esn)
        ww_results_test[esn] = ww_result
        plot_ww_prediction(ww_result)

# %% [markdown]
# # Salvataggio Modelli su Disco
# Salva tutti i modelli addestrati con joblib così puoi ricaricarli
# senza dover riallenare da zero ogni volta.

# %%
import joblib
import os

MODELS_DIR = "./saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Il regressore lineare usato come ensemble per calcolare i residui
joblib.dump(models,            f"{MODELS_DIR}/linear_ensemble.pkl")

# I coefficienti alpha dell'health index (uno per HPT, uno per HPC)
joblib.dump(chpt,              f"{MODELS_DIR}/chpt.pkl")
joblib.dump(chpc,              f"{MODELS_DIR}/chpc.pkl")

# I LightGBM per la correzione del gap
joblib.dump(lgbm_gap_hpt,      f"{MODELS_DIR}/lgbm_gap_hpt.pkl")
joblib.dump(lgbm_gap_hpc,      f"{MODELS_DIR}/lgbm_gap_hpc.pkl")

# I coefficienti di scale_to_target per ciclo
joblib.dump(scale_coefs_hpt,   f"{MODELS_DIR}/scale_coefs_hpt.pkl")
joblib.dump(scale_coefs_hpc,   f"{MODELS_DIR}/scale_coefs_hpc.pkl")

# I classificatori del "ciclo di lavoro" (quante manutenzioni sono già state fatte)
joblib.dump(clf_hpt,           f"{MODELS_DIR}/clf_hpt.pkl")
joblib.dump(clf_hpc,           f"{MODELS_DIR}/clf_hpc.pkl")

# Le liste di feature names: cruciali per allineare input a modelli salvati
joblib.dump(regs_feature_cols, f"{MODELS_DIR}/regs_feature_cols.pkl")
joblib.dump(clf_feature_cols,  f"{MODELS_DIR}/clf_feature_cols.pkl")

print(f"Tutti i modelli salvati in: {MODELS_DIR}/")


# %% [markdown]
# # Predizione WW corretta + Generazione Submission
#
# Il blocco precedente "Generate Submission" non salvava i risultati WW
# e non assemblava il CSV finale.  Questo blocco corregge entrambi i problemi.

# %%
# ----- funzione di estrapolazione WW -----
# Dopo l'ultimo evento WW rilevato (o dall'inizio del dato se nessuno è stato
# rilevato), T45 sale linearmente con pendenza `slope`.
# Un nuovo evento WW scatta quando la media mobile supera:
#   floor  +  slope * e² * factor_mult
# Partiamo dal valore corrente di T45 e calcoliamo quanti cicli mancano.

from math import e as _e

def cycles_to_next_ww_from_end(ww_result, factor_mult=80):
    """
    Stima quanti cicli mancano alla prossima Water Wash dall'ultimo
    punto disponibile nel dato (val o test).

    Parametri
    ---------
    ww_result   : output di predict_ww()
    factor_mult : stesso valore usato in detect_ww_events (default 80)

    Ritorna
    -------
    int  ≥ 0  — cicli stimati alla prossima WW
    """
    wwdf  = ww_result["wwdf"]
    slope = ww_result["slope"]
    sv    = ww_result["detected_events"]   # dict {row_index: t45_value}

    if len(wwdf) == 0 or slope <= 0:
        # pendenza piatta o nessun dato → non possiamo stimare, ritorniamo 0
        return 0

    # La soglia di attivazione: quanto deve salire T45 sopra la base
    trigger_threshold = slope * (_e ** 2) * factor_mult

    # Floor = T45 all'ultimo evento WW rilevato (o al primo punto del dato)
    if sv:
        last_event_idx = max(sv.keys())
        floor = sv[last_event_idx]
    else:
        floor = wwdf["Sensed_T45"].iloc[0]

    # Valore T45 attuale (fine del dato)
    current_t45 = wwdf["Sensed_T45"].iloc[-1]

    # Rise già accumulato dall'ultimo floor
    current_rise = current_t45 - floor

    # Rise ancora necessario per raggiungere il trigger
    remaining_rise = trigger_threshold - current_rise

    if remaining_rise <= 0:
        # T45 ha già superato la soglia → WW imminente
        return 0

    # Cicli rimanenti = rise rimanente / velocità di salita (slope)
    cycles_remaining = remaining_rise / slope

    return max(0, cycles_remaining)


# %%
# ----- Predizione WW sui file di test (con storage corretto) -----
# dft[i] corrisponde a test_i.csv  → usiamo l'indice i come chiave aggiuntiva
# così possiamo recuperare il risultato sia per ESN che per numero file.

print("=== WW PREDICTION — TEST (submission run) ===")
ww_results_test_final = {}   # chiave: ESN   oppure  indice file (i)

for i, t in enumerate(dft):
    for esn in t["ESN"].unique():
        engine_df  = t[t["ESN"] == esn]
        engine_res = _residuals(engine_df)
        if engine_res is None:
            print(f"  test_{i} (ESN {esn}): residui None, skip")
            continue
        ww_result = predict_ww(engine_df, engine_res, esn)
        cycles_ww = cycles_to_next_ww_from_end(ww_result)

        # Salviamo con entrambe le chiavi per comodità
        ww_results_test_final[esn] = ww_result
        ww_results_test_final[i]   = ww_result

        print(f"  test_{i} (ESN {esn}): eventi rilevati={ww_result['n_events']}  "
              f"slope={ww_result['slope']:.5f}  Cycles_to_WW≈{cycles_ww}")


# %%
# ----- Assemblaggio CSV di submission -----
# Per ogni file test_i prendiamo:
#   • Cycles_to_HPT_SV e Cycles_to_HPC_SV  da results_test (pipeline v2)
#   • Cycles_to_WW                          dall'estrapolazione WW qui sopra

# Range di training: usato come fallback se un motore manca nelle predizioni
_fallback_hpt = float(regs_data["target_hpt"].mean())
_fallback_hpc = float(regs_data["target_hpc"].mean())

rows = []
for i, engine_df in enumerate(dft):
    file_name = f"test_{i}"
    esn       = engine_df["ESN"].iloc[0]

    # --- HPT e HPC ---
    mask = results_test["ESN"] == esn
    if mask.any():
        cycles_hpt = float(results_test.loc[mask, "Cycles_to_HPT_SV"].values[0])
        cycles_hpc = float(results_test.loc[mask, "Cycles_to_HPC_SV"].values[0])
    else:
        cycles_hpt = _fallback_hpt
        cycles_hpc = _fallback_hpc
        print(f"  {file_name}: FALLBACK HPT/HPC (ESN {esn} non trovato in results_test)")

    # --- WW ---
    if esn in ww_results_test_final:
        cycles_ww = cycles_to_next_ww_from_end(ww_results_test_final[esn])
    elif i in ww_results_test_final:
        cycles_ww = cycles_to_next_ww_from_end(ww_results_test_final[i])
    else:
        cycles_ww = 0
        print(f"  {file_name}: WW non disponibile, impostato a 0")

    rows.append({
        "file":             file_name,
        "Cycles_to_WW":     cycles_ww,
        "Cycles_to_HPC_SV": cycles_hpc,
        "Cycles_to_HPT_SV": cycles_hpt,
    })

    print(f"  {file_name} (ESN {esn}):  "
          f"WW={cycles_ww:.0f}  HPC={cycles_hpc:.0f}  HPT={cycles_hpt:.0f}")

# Assembla il DataFrame nel formato richiesto dal portale
submission_df = pd.DataFrame(
    rows,
    columns=["file", "Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]
)

# Salva nella root del progetto (stessa posizione del template submission.csv)
SUBMISSION_OUTPUT = "../../submission.csv"
submission_df.to_csv(SUBMISSION_OUTPUT, index=False)

print(f"\nSubmission salvata in: {SUBMISSION_OUTPUT}")
print(submission_df.to_string(index=False))
