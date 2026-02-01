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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import scipy.optimize as optimize
import scipy.stats as stats
from pyparsing import line
from sympy import O, deg
# %load_ext autoreload
# %autoreload 2
from tools import utils as u, config as cfg, plotting as up, preprocessing as pp

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



# %%
from numpy import sign


model_i = 0
testing_esn = 102
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
degradation_vars = [s for s in u.SENSORS if s not in operating_vars]

# caricamento e preprocessamento iniziale
df = u.load_training()()
df = pp.remove_outliers(df, u.SENSORS)
df = pp.missingfill(df).dropna()

# preparazione train-test split
test_data = df[df["ESN"] == testing_esn].reset_index()
X_test = test_data[operating_vars]
Y_test = test_data[degradation_vars]

# training modelli con shift
models = train_models(df[df["ESN"].isin([x for x in [101,102,103,104] if x != testing_esn])], operating_vars, degradation_vars)

# selezione modello
model = models[model_i]['model']

# predict dei valori
Y_pred = model.predict(np.roll(X_test, model_i, axis=1))

# residui
res = Y_test - Y_pred

# integrazione residui sul dataset originale
res = pp.remove_outliers(res, u.SENSORS, threshold=3)
test_data[degradation_vars] = res
res = test_data.dropna()

## QUESTO è il plot quello che tipo deve sembrare quello dei koreani
# fig, axs = plt.subplots(2,3, figsize=(15,8))
# for i, ax in enumerate(axs.flat):
#     if isinstance(ax, plt.Axes):
#         ax.plot(res.iloc[:,i], linewidth=1)
#         ax.set_title(degradation_vars[i])
#         ax.set_ylabel("Residuals")
#         ax.set_xlabel(f"{res.iloc[:,i].index.name}_res")
#         ax.grid()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
# fig.show()
## finisce qua

# finestra smoothing dei residui
window = 370
step = window//5

res = res.rolling(window, step).median()
res = median_norm(res)
res = res.dropna()

hpt_rul = res["Cycles_to_HPT_SV"].reset_index(drop=True)
hpc_rul = res["Cycles_to_HPC_SV"].reset_index(drop=True)
T3_res = res["Sensed_T3"]
T45_res = res["Sensed_T45"]


# %%
# Ricerca operativa con minimize locale
## Valori iniziali
a_hpt = 1000
a_hpc = -1000

## Vincoli sulle variabili
bounds = [
    (None, None),
    (0, None)
]

result_hpt = minimize(
    objective, 
    x0=a_hpt, 
    args=(T3_res, T45_res, hpt_rul),
    method='Nelder-Mead'
)

result_hpc = minimize(
    objective, 
    x0=a_hpc, 
    args=(T3_res, T45_res, hpc_rul),
    method='Nelder-Mead'
)

a_hpt = result_hpt.x[0]
a_hpc = result_hpc.x[0]

print(f"alpha_hpt:{a_hpt}")
print(f"alpha_hpt:{a_hpc}")

# %%
from scipy.optimize import differential_evolution

bounds = [
    (-1000, 1000),  # Alpha
    (-1000, 1000),  # Beta
]

result_hpt = differential_evolution(
    objective_beta,      
    bounds=bounds,           
    args=(T3_res, T45_res, hpt_rul), 
    strategy='best1bin',     
    maxiter=200,                # generazioni
    popsize=500,                
    workers=-1,
    tol=0,                      # Tolleranza 
)

result_hpc = differential_evolution(
    objective_beta,      
    bounds=bounds,           
    args=(T3_res, T45_res, hpc_rul), 
    strategy='best1bin',     
    maxiter=200,                # generazioni 
    popsize=500,              
    workers=-1,
    tol=0,                      # Tolleranza
)

a_hpt, b_hpt = result_hpt.x
print(f"MIGLIOR RISULTATO TROVATO:")
print(f"Alpha: {a_hpt}")
print(f"Beta: {b_hpt}")

a_hpc, b_hpc = result_hpc.x
print(f"MIGLIOR RISULTATO TROVATO:")
print(f"Alpha: {a_hpc}")
print(f"Beta: {b_hpc}")

# %%
from scipy.optimize import differential_evolution

bounds = [
    (-1000, 1000),  # a
    (-1000, 1000),  # b
    (-1000, 1000),  # c
    (-1000, 1000),  # d
    (-1000, 1000),  # e
    (-1000, 1000),  # f
    (-1000, 1000),  # g
    (-1000, 1000),  # h
]

def HIE(params, vars):
    #return np.sum([-params[i]*vars.iloc[:,i] for i in range(0, 8)])
    return vars.dot(-np.array(params))

    
def objective_experimental(params, vars, RUL):
    hi = HIE(params, vars)
    RUL = RUL.dropna()
    corr = stats.pearsonr(RUL,hi)
    return -corr[0]


result_hpt = differential_evolution(
    objective_experimental,      
    bounds=bounds,           
    args=(res[degradation_vars], hpt_rul), 
    strategy='best1bin',     
    maxiter=200,                # generazioni
    popsize=500,                
    workers=-1,
    tol=0,                      # Tolleranza 
)

result_hpc = differential_evolution(
    objective_experimental,      
    bounds=bounds,           
    args=(res[degradation_vars], hpc_rul), 
    strategy='best1bin',     
    maxiter=200,                # generazioni 
    popsize=500,              
    workers=-1,
    tol=0,                      # Tolleranza
)

coefs_hpt = result_hpt.x
print(f"MIGLIOR RISULTATO TROVATO:")
for c in coefs_hpt:
    print(f"{c}")

coefs_hpc = result_hpc.x
print(f"MIGLIOR RISULTATO TROVATO:")
for c in coefs_hpc:
    print(f"{c}")



# %%

hi_hpt = HIE(coefs_hpt, res[degradation_vars])
hi_hpc = HIE(coefs_hpc, res[degradation_vars])

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
axs[0].plot(hi_hpt, color='tab:blue', label='Health Index (HPT)')
ax0_rul = axs[0].twinx()
ax0_rul.plot(hpt_rul, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
axs[1].plot(hi_hpc, color='tab:green', label='Health Index (HPC)')
ax1_rul = axs[1].twinx()
ax1_rul.plot(hpc_rul, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
fig.tight_layout()
fig.show()


# %%
#hi_hpt = HI(T3_res, T45_res, a_hpt)
#hi_hpt = -a_hpt*T3_res - b_hpt*T45_res
hi_hpt = HIE(coefs_hpt, hpt_rul)
#hi_hpc = HI(T3_res, T45_res, a_hpc)
#hi_hpc = -a_hpc*T3_res - b_hpc*T45_res
hi_hpc = HIE(coefs_hpc, hpc_rul)
# error_hpt = np.sum(hi_hpt - hpt_rul)
# error_hpc = np.sum(hi_hpc - hpc_rul)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
axs[0].plot(hi_hpt, color='tab:blue', label='Health Index (HPT)')
ax0_rul = axs[0].twinx()
ax0_rul.plot(hpt_rul, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
axs[1].plot(hi_hpc, color='tab:green', label='Health Index (HPC)')
ax1_rul = axs[1].twinx()
ax1_rul.plot(hpc_rul, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
fig.tight_layout()
fig.show()


# %%
# col = 1
# plt.figure()
# plt.plot(test_data["Cycles_Since_New"],Y_pred[:,col], linewidth=0.5, label="Predicted")
# plt.plot(test_data["Cycles_Since_New"],Y_test.iloc[:,col], linewidth=0.5, label="Observed")
# plt.show()

# a = 0.01
# t3_res = np.array(df["T3_res"])
# t45_res = np.array(df["T45_res"])
# hi_target = -a * t3_res - t45_res
# print(df)
# plt.figure()
# plt.plot(df["Cycles_since_new"], hi_target)
# plt.show()
