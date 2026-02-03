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
from scipy.optimize import minimize
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

def HIE(params, vars):
    #return np.sum([-params[i]*vars.iloc[:,i] for i in range(0, 8)])
    return vars.dot(-np.array(params))

    
def objective_experimental(params, vars, RUL):
    hi = HIE(params, vars)
    RUL = RUL.dropna()
    corr = stats.pearsonr(RUL,hi)
    return -corr[0]

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
model_i = 0
testing_esn = 102
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
degradation_vars = [s for s in u.SENSORS if s not in operating_vars]


# %%
# caricamento e preprocessamento iniziale
df = u.load_training()()
df = pp.remove_outliers(df, u.SENSORS)
df = pp.missingfill(df).dropna()

# Per fare il training direttamente con gli snapshot di un ciclo collassati
# managed_cols = set(degradation_vars) | set(operating_vars)
# other_cols = [col for col in df.columns if col not in managed_cols]
# agg_logic = {col: 'median' for col in degradation_vars}
# agg_logic.update({col: 'median' for col in operating_vars})
# agg_logic.update({col: 'first' for col in other_cols})

# preparazione train-test split
test_data = df[df["ESN"] == testing_esn].reset_index()
X_test = test_data[operating_vars]
Y_test = test_data[degradation_vars]
df = df[df["ESN"].isin([x for x in [101,102,103,104] if x != testing_esn])]
df = df.groupby(["ESN", "Snapshot"]).median().reset_index()

# training modelli con shift
models = train_models(df, operating_vars, degradation_vars)
# %store models

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
# %store res

hpt_rul = res["Cycles_to_HPT_SV"].reset_index(drop=True)
hpc_rul = res["Cycles_to_HPC_SV"].reset_index(drop=True)
ww_rul = res["Cycles_to_WW"].reset_index(drop=True)
T3_res = res["Sensed_T3"]
T45_res = res["Sensed_T45"]

# %% [markdown]
# # Ricerca di a_hpt e a_hpc LOCALI

# %%
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

# %% [markdown]
# # Ricerca di a_hpt/b_hpt e a_hpc/b_hpc GLOBALI

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

# %% [markdown]
# # Ricerca di a,b,c,d,e,f,g globali combinazione lineare di tutti i sensori

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

result_ww = differential_evolution(
    objective_experimental,      
    bounds=bounds,           
    args=(res[degradation_vars], ww_rul), 
    strategy='best1bin',     
    maxiter=200,                # generazioni 
    popsize=500,              
    workers=-1,
    tol=0,                      # Tolleranza
)

# %store result_hpt
# %store result_hpc
# %store result_ww

coefs_hpt = result_hpt.x
print(f"MIGLIOR RISULTATO TROVATO:")
for c in coefs_hpt:
    print(f"{c}")


coefs_hpc = result_hpc.x
print(f"MIGLIOR RISULTATO TROVATO:")
for c in coefs_hpc:
    print(f"{c}")

coefs_ww = result_ww.x
print(f"MIGLIOR RISULTATO TROVATO:")
for c in coefs_ww:
    print(f"{c}")

# %store coefs_hpt
# %store coefs_hpc
# %store coefs_ww


# %%
# oppure avviare semplicemente questo blocco

coefs_hpt = (102.54752751902083,
            -12.215920703632944,
            -13.937522163047133,
            -4.187395005088061,
            50.41175875590375,
            47.642362206135715,
            -925.3978926377761,
            -13.188043715234654)

coefs_hpc = (363.09492555179804,
            2.0186697716587463,
            13.743038965485157,
            11.855221907006188,
            69.72211072964963,
            0.42476560122623985,
            -989.8022299435106,
            28.63715373059266)

coefs_ww = ( 850.7167357577274,
            -15.072134101084401,
            -24.57278901752793,
            102.1439502693563,
            -119.99243468799672,
            11.396869998772502,
            -917.4479008347435,
            -24.27692144851878)

# %%
# PLOTTING RISULTATO
hi_hpt = HIE(coefs_hpt, res[degradation_vars])
hi_hpc = HIE(coefs_hpc, res[degradation_vars])
hi_ww  = HIE(coefs_ww, res[degradation_vars])

fig, axs = plt.subplots(1, 3, figsize=(30, 6))
axs[0].plot(hi_hpt, color='tab:blue', label='Health Index (HPT)')
ax0_rul = axs[0].twinx()
ax0_rul.plot(hpt_rul, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
axs[1].plot(hi_hpc, color='tab:green', label='Health Index (HPC)')
ax1_rul = axs[1].twinx()
ax1_rul.plot(hpc_rul, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
axs[2].plot(hi_ww, color='tab:green', label='Health Index (HPC)')
ax2_rul = axs[2].twinx()
ax2_rul.plot(ww_rul, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
fig.tight_layout()
fig.show()


# %% [markdown]
# # Plotting dei primi due metodi

# %%
#hi_hpt = HI(T3_res, T45_res, a_hpt)
#hi_hpt = -a_hpt*T3_res - b_hpt*T45_res
hi_hpt = HIE(coefs_hpt, res[degradation_vars])
#hi_hpc = HI(T3_res, T45_res, a_hpc)
#hi_hpc = -a_hpc*T3_res - b_hpc*T45_res
hi_hpc = HIE(coefs_hpc, res[degradation_vars])
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

# %% [markdown]
# ### Classificazione dell'errore con LightGBM per HPC, HPT e WW

# %%
hi_hpc = HIE(coefs_hpc, res[degradation_vars]) 
hi_hpt = HIE(coefs_hpt, res[degradation_vars]) 
hi_ww = HIE(coefs_ww, res[degradation_vars]) 

regr_hpc = LinearRegression()
regr_hpt = LinearRegression()
regr_ww = LinearRegression()

X_base_hpc = hi_hpc.values.reshape(-1,1)
X_base_hpt = hi_hpt.values.reshape(-1,1)
X_base_ww = hi_ww.values.reshape(-1,1)

regr_hpc.fit(X_base_hpc, hpc_rul)
regr_hpt.fit(X_base_hpt, hpt_rul)
regr_ww.fit(X_base_ww, ww_rul)

pred_rul_hpc = regr_hpc.predict(X_base_hpc)
pred_rul_hpt = regr_hpc.predict(X_base_hpt)
pred_rul_ww = regr_hpc.predict(X_base_ww)

gap_true_hpc = hpc_rul - pred_rul_hpc
gap_true_hpt = hpt_rul - pred_rul_hpt
gap_true_ww = ww_rul - pred_rul_ww

window_size = 800

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


lgbm_hpc = lgb.LGBMRegressor(n_estimators=20000, learning_rate=0.001)
lgbm_hpc.fit(X_lgbm_hpc, gap_true_hpc)
# %store lgbm_hpc

lgbm_hpt = lgb.LGBMRegressor(n_estimators=20000, learning_rate=0.001)
lgbm_hpt.fit(X_lgbm_hpt, gap_true_hpt)
# %store lgbm_hpt

lgbm_ww = lgb.LGBMRegressor(n_estimators=20000, learning_rate=0.001)
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

