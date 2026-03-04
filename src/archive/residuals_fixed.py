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

# %%
def train_models(df, operating_vars, degradation_vars) -> dict[int, dict[str,LinearRegression]]:
    X_train = df[operating_vars]
    Y_train = df[degradation_vars]
    models = {}
    # CORRECTION: Loop reduced to logic for single model but kept structure for compatibility
    # Original loop over range(0,8) with roll was redundant for Linear Regression.
    for i in range(0,1): 
        X_temp = X_train # No roll needed for standard linear regression
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
testing_esn = 104
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
degradation_vars = [s for s in u.SENSORS if s not in operating_vars]


# %%
# caricamento e preprocessamento iniziale
df_all = u.load_training()()
df_all = pp.remove_outliers(df_all, u.SENSORS)
df_all = pp.missingfill(df_all).dropna()

# Definizione set di training
df_train = df_all[df_all["ESN"].isin([101, 103, 102])].copy()
# Aggregazione per ciclo (fondamentale per training corretto)
df_train = df_train.groupby(["ESN", "Cycles_Since_New"]).median().reset_index()

# training modello lineare
models = train_models(df_train, operating_vars, degradation_vars)
# %store models

# selezione modello
model = models[model_i]['model']

# ==============================================================================
# CALCOLO RESIDUI SUL TRAINING SET (CORREZIONE DATA LEAKAGE)
# ==============================================================================
# Invece di usare i dati di test (test_data) per ottimizzare i coefficienti,
# usiamo i residui calcolati sullo stesso training set.

X_train = df_train[operating_vars]
Y_train = df_train[degradation_vars]
Y_pred_train = model.predict(X_train)

res_train = Y_train - Y_pred_train
res_train = pp.remove_outliers(res_train, u.SENSORS, threshold=3)

# Smoothing coerente
window = 370
step = window//5 # step per ridurre i punti, opzionale, qui manteniamo coerenza col codice originale
# Attenzione: sul training, facendo rolling perdiamo i primi 'window' punti.
# Dobbiamo assicurarci di allineare res_train e RUL.
res_train_smooth = res_train.rolling(window, min_periods=1).median() # Step 1 per semplicità o step=step
res_train_smooth = median_norm(res_train_smooth)
res_train_smooth = res_train_smooth.dropna()

# Allineamento RUL Training
idx = res_train_smooth.index
df_train_aligned = df_train.loc[idx]

hpt_rul_train = df_train_aligned["Cycles_to_HPT_SV"]
hpc_rul_train = df_train_aligned["Cycles_to_HPC_SV"]
ww_rul_train = df_train_aligned["Cycles_to_WW"]
T3_res_train = res_train_smooth["Sensed_T3"]
T45_res_train = res_train_smooth["Sensed_T45"]


# %% [markdown]
# # Ricerca di a,b,c,d,e,f,g globali combinazione lineare di tutti i sensori (TRAINING)

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

print("Ottimizzazione HPT sui dati di training...")
result_hpt = differential_evolution(
    objective_experimental,      
    bounds=bounds,           
    args=(res_train_smooth[degradation_vars], hpt_rul_train), 
    strategy='best1bin',     
    maxiter=500,                
    popsize=50,                
    workers=-1,
    tol=0,
)

print("Ottimizzazione HPC sui dati di training...")
result_hpc = differential_evolution(
    objective_experimental,      
    bounds=bounds,           
    args=(res_train_smooth[degradation_vars], hpc_rul_train), 
    strategy='best1bin',     
    maxiter=500, 
    popsize=50,              
    workers=-1,
    tol=0,
)

print("Ottimizzazione WW sui dati di training...")
result_ww = differential_evolution(
    objective_experimental,      
    bounds=bounds,           
    args=(res_train_smooth[degradation_vars], ww_rul_train), 
    strategy='best1bin',     
    maxiter=500, 
    popsize=50,              
    workers=-1,
    tol=0,
)

coefs_hpt = result_hpt.x
coefs_hpc = result_hpc.x
coefs_ww = result_ww.x

print("Coefficienti Trovati (Training):")
print("HPT:", coefs_hpt)
print("HPC:", coefs_hpc)
print("WW:", coefs_ww)


# %% [markdown]
# # Plotting dei risultati di ottimizzazione (Training)

# %%
# PLOTTING RISULTATO SU TRAINING
hi_hpt_plot = HIE(coefs_hpt, res_train_smooth[degradation_vars])
hi_hpc_plot = HIE(coefs_hpc, res_train_smooth[degradation_vars])
hi_ww_plot  = HIE(coefs_ww, res_train_smooth[degradation_vars])

fig, axs = plt.subplots(1, 3, figsize=(30, 6))
axs[0].plot(hi_hpt_plot.values, color='tab:blue', label='Health Index (HPT)')
ax0_rul = axs[0].twinx()
ax0_rul.plot(hpt_rul_train.values, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
axs[0].set_title("Training: HPT Health Index vs RUL")
axs[0].legend(loc='upper left')
ax0_rul.legend(loc='upper right')

axs[1].plot(hi_hpc_plot.values, color='tab:green', label='Health Index (HPC)')
ax1_rul = axs[1].twinx()
ax1_rul.plot(hpc_rul_train.values, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
axs[1].set_title("Training: HPC Health Index vs RUL")
axs[1].legend(loc='upper left')
ax1_rul.legend(loc='upper right')

axs[2].plot(hi_ww_plot.values, color='tab:purple', label='Health Index (WW)')
ax2_rul = axs[2].twinx()
ax2_rul.plot(ww_rul_train.values, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale')
axs[2].set_title("Training: WW Health Index vs RUL")
axs[2].legend(loc='upper left')
ax2_rul.legend(loc='upper right')

fig.tight_layout()
fig.show()


# %% [markdown]
# ### Training LightGBM per correzione errore (SUL TRAINING SET)

# %%
# 1. Calcolo Health Index su Training
hi_hpc_train = HIE(coefs_hpc, res_train_smooth[degradation_vars]) 
hi_hpt_train = HIE(coefs_hpt, res_train_smooth[degradation_vars]) 
hi_ww_train = HIE(coefs_ww, res_train_smooth[degradation_vars]) 

# 2. Addestramento "Base Predictor" (Regressione Lineare Semplice: HI -> RUL)
# Questo serve per avere una stima di base su cui il LGBM imparerà l'errore
regr_hpc = LinearRegression()
regr_hpt = LinearRegression()
regr_ww = LinearRegression()

X_base_hpc_train = hi_hpc_train.values.reshape(-1,1)
X_base_hpt_train = hi_hpt_train.values.reshape(-1,1)
X_base_ww_train = hi_ww_train.values.reshape(-1,1)

regr_hpc.fit(X_base_hpc_train, hpc_rul_train)
regr_hpt.fit(X_base_hpt_train, hpt_rul_train)
regr_ww.fit(X_base_ww_train, ww_rul_train)

# 3. Calcolo GAP (Errore) sul Training
pred_rul_hpc_base_train = regr_hpc.predict(X_base_hpc_train)
pred_rul_hpt_base_train = regr_hpt.predict(X_base_hpt_train)
pred_rul_ww_base_train = regr_ww.predict(X_base_ww_train)

gap_train_hpc = hpc_rul_train - pred_rul_hpc_base_train
gap_train_hpt = hpt_rul_train - pred_rul_hpt_base_train
gap_train_ww = ww_rul_train - pred_rul_ww_base_train

# 4. Feature Engineering per LightGBM (Slope/Intercept)
window_size = 800 # Variabile definita

feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc_train, window_size)
feat_slope_hpt, feat_intercept_hpt = get_rolling_slope_intercept(hi_hpt_train, window_size)
feat_slope_ww, feat_intercept_ww = get_rolling_slope_intercept(hi_ww_train, window_size)

X_lgbm_hpc_train = pd.DataFrame({'HI': hi_hpc_train.values, 'Slope': feat_slope_hpc, 'Intercept': feat_intercept_hpc})
X_lgbm_hpt_train = pd.DataFrame({'HI': hi_hpt_train.values, 'Slope': feat_slope_hpt, 'Intercept': feat_intercept_hpt})
X_lgbm_ww_train = pd.DataFrame({'HI': hi_ww_train.values, 'Slope': feat_slope_ww, 'Intercept': feat_intercept_ww})

# Pulizia dai primi valori nulli/zero del rolling slope se necessario
mask_hpc = X_lgbm_hpc_train['Slope'] != 0
X_lgbm_hpc_train = X_lgbm_hpc_train[mask_hpc]
gap_train_hpc = gap_train_hpc[mask_hpc]

mask_hpt = X_lgbm_hpt_train['Slope'] != 0
X_lgbm_hpt_train = X_lgbm_hpt_train[mask_hpt]
gap_train_hpt = gap_train_hpt[mask_hpt]

mask_ww = X_lgbm_ww_train['Slope'] != 0
X_lgbm_ww_train = X_lgbm_ww_train[mask_ww]
gap_train_ww = gap_train_ww[mask_ww]


# 5. Training Effettivo LightGBM
print("Training LightGBM models...")
lgbm_hpc = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.01) # Parametri ridotti per sicurezza
lgbm_hpc.fit(X_lgbm_hpc_train, gap_train_hpc)

lgbm_hpt = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.01)
lgbm_hpt.fit(X_lgbm_hpt_train, gap_train_hpt)

lgbm_ww = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.01)
lgbm_ww.fit(X_lgbm_ww_train, gap_train_ww)


# %% [markdown]
# # Plotting Performance LightGBM (Training)
# Verifica se il modello ha imparato a correggere l'errore sistematico.

# %%
# HPC
pred_gap_train_hpc = lgbm_hpc.predict(X_lgbm_hpc_train)
pred_rul_final_train_hpc = pred_rul_hpc_base_train[mask_hpc] + pred_gap_train_hpc

plt.figure(figsize=(10, 6))
plt.scatter(gap_train_hpc, pred_gap_train_hpc, alpha=0.6, s=15, color='blue', label='Predicted vs True Gap')
min_val = min(gap_train_hpc.min(), pred_gap_train_hpc.min())
max_val = max(gap_train_hpc.max(), pred_gap_train_hpc.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', label='Perfect Prediction')
plt.title('HPC (Training): Gap Prediction Accuracy')
plt.xlabel('True Gap Difference')
plt.ylabel('Predicted Gap Difference')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(hpc_rul_train[mask_hpc].values, 'k--', linewidth=2, label='True RUL')
plt.plot(pred_rul_hpc_base_train[mask_hpc], color='tab:red', alpha=0.6, linestyle='-.', label='Linear Prediction (Base)')
plt.plot(pred_rul_final_train_hpc, color='tab:green', linewidth=2, label='LightGBM Corrected Prediction')
plt.title('HPC (Training): Impact of LightGBM Correction on RUL')
plt.legend()
plt.grid(True)
plt.show()

# HPT
pred_gap_train_hpt = lgbm_hpt.predict(X_lgbm_hpt_train)
pred_rul_final_train_hpt = pred_rul_hpt_base_train[mask_hpt] + pred_gap_train_hpt

plt.figure(figsize=(10, 6))
plt.scatter(gap_train_hpt, pred_gap_train_hpt, alpha=0.6, s=15, color='blue', label='Predicted vs True Gap')
min_val = min(gap_train_hpt.min(), pred_gap_train_hpt.min())
max_val = max(gap_train_hpt.max(), pred_gap_train_hpt.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', label='Perfect Prediction')
plt.title('HPT (Training): Gap Prediction Accuracy')
plt.xlabel('True Gap Difference')
plt.ylabel('Predicted Gap Difference')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(hpt_rul_train[mask_hpt].values, 'k--', linewidth=2, label='True RUL')
plt.plot(pred_rul_hpt_base_train[mask_hpt], color='tab:red', alpha=0.6, linestyle='-.', label='Linear Prediction (Base)')
plt.plot(pred_rul_final_train_hpt, color='tab:green', linewidth=2, label='LightGBM Corrected Prediction')
plt.title('HPT (Training): Impact of LightGBM Correction on RUL')
plt.legend()
plt.grid(True)
plt.show()

# WW
pred_gap_train_ww = lgbm_ww.predict(X_lgbm_ww_train)
pred_rul_final_train_ww = pred_rul_ww_base_train[mask_ww] + pred_gap_train_ww

plt.figure(figsize=(10, 6))
plt.scatter(gap_train_ww, pred_gap_train_ww, alpha=0.6, s=15, color='blue', label='Predicted vs True Gap')
min_val = min(gap_train_ww.min(), pred_gap_train_ww.min())
max_val = max(gap_train_ww.max(), pred_gap_train_ww.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', label='Perfect Prediction')
plt.title('WW (Training): Gap Prediction Accuracy')
plt.xlabel('True Gap Difference')
plt.ylabel('Predicted Gap Difference')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(30, 6))
plt.plot(ww_rul_train[mask_ww].values, 'k--', linewidth=2, label='True RUL')
plt.plot(pred_rul_ww_base_train[mask_ww], color='tab:red', alpha=0.6, linestyle='-.', label='Linear Prediction (Base)')
plt.plot(pred_rul_final_train_ww, color='tab:green', linewidth=2, label='LightGBM Corrected Prediction')
plt.title('WW (Training): Impact of LightGBM Correction on RUL')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# # Testing Corretto (Con Aggregazione e Pipeline Completa)

# %%
# Testiamo sul dataset di Test originale (che può essere ESN 102 o altro file)
# Qui usiamo load_testing() come richiesto dall'utente nel blocco finale
df_test_raw = u.load_testing()()
df_test_raw = pp.remove_outliers(df_test_raw, u.SENSORS)
df_test_raw = pp.missingfill(df_test_raw).dropna()

engines = {}
for eng in df_test_raw["ESN"].unique():
    print(f"Processing Engine {eng}...")
    engines[eng] = {}
    
    # 1. Estrazione dati raw
    test_data = df_test_raw[df_test_raw["ESN"] == eng].reset_index()
    
    # 2. AGGREGAZIONE (FIX CRITICO): Median per ciclo
    test_data_agg = test_data.groupby(["Cycles_Since_New"]).median().reset_index()

    # 3. Preparazione Input Modello Lineare (Rolling opzionale per smoothing input, ma base è 1 punto/ciclo)
    rolling_size = 100
    step = 1
    # Nota: Usiamo .rolling() sui dati aggregati
    X_test = test_data_agg[operating_vars].rolling(rolling_size, step=step).median().dropna()
    Y_test = test_data_agg[degradation_vars].rolling(rolling_size, step=step).median().dropna()
    
    # 4. Predizione Lineare Base
    Y_pred = model.predict(X_test)

    # 5. Calcolo Residui
    res = Y_test - Y_pred
    
    # Allineamento indici per mantenere RUL corrette
    idx_aligned = res.index
    test_data_aligned = test_data_agg.loc[idx_aligned]
    
    engines[eng]["X_test"] = X_test.copy()
    engines[eng]["Y_test"] = Y_test.copy()
    engines[eng]["Y_pred"] = Y_pred.copy()
    engines[eng]["res"] = res.copy()

    # 6. Calcolo Health Index (Usando coefficienti ottimizzati nel training)
    hi_hpt = HIE(coefs_hpt, res[degradation_vars])
    hi_hpc = HIE(coefs_hpc, res[degradation_vars])
    hi_ww  = HIE(coefs_ww, res[degradation_vars])

    engines[eng]["hi_hpt"] = hi_hpt
    engines[eng]["hi_hpc"] = hi_hpc
    engines[eng]["hi_ww"] = hi_ww

    # 7. HPC Pipeline (Feature -> LGBM -> Correction)
    feat_slope_hpc, feat_intercept_hpc = get_rolling_slope_intercept(hi_hpc, window_size)
    X_lgbm_hpc = pd.DataFrame({
        'HI': hi_hpc.values,
        'Slope': feat_slope_hpc,
        'Intercept': feat_intercept_hpc
    })
    
    gap_pred_hpc = lgbm_hpc.predict(X_lgbm_hpc)              # Predizione Errore
    base_pred_hpc = regr_hpc.predict(hi_hpc.values.reshape(-1,1)) # Predizione Base da HI
    final_rul_hpc = gap_pred_hpc + base_pred_hpc.flatten()   # Somma (Correzione)

    # 8. HPT Pipeline
    feat_slope_hpt, feat_intercept_hpt = get_rolling_slope_intercept(hi_hpt, window_size)
    X_lgbm_hpt = pd.DataFrame({
        'HI': hi_hpt.values,
        'Slope': feat_slope_hpt,
        'Intercept': feat_intercept_hpt
    })
    
    gap_pred_hpt = lgbm_hpt.predict(X_lgbm_hpt)
    base_pred_hpt = regr_hpt.predict(hi_hpt.values.reshape(-1,1))
    final_rul_hpt = gap_pred_hpt + base_pred_hpt.flatten()

    # 9. WW Pipeline
    feat_slope_ww, feat_intercept_ww = get_rolling_slope_intercept(hi_ww, window_size)
    X_lgbm_ww = pd.DataFrame({
        'HI': hi_ww.values,
        'Slope': feat_slope_ww,
        'Intercept': feat_intercept_ww
    })
    
    gap_pred_ww = lgbm_ww.predict(X_lgbm_ww)
    base_pred_ww = regr_ww.predict(hi_ww.values.reshape(-1,1))
    final_rul_ww = gap_pred_ww + base_pred_ww.flatten()

    # 10. Plotting
    # Recuperiamo le RUL reali se presenti (per confronto)
    has_labels = "Cycles_to_HPC_SV" in test_data_aligned.columns
    
    plt.subplots(1,3, figsize=(25,8))
    plt.suptitle(f'Engine ESN {eng} - Health Index and RUL Predictions')
    
    plt.subplot(1,3,1)
    if has_labels: plt.plot(test_data_aligned["Cycles_to_HPC_SV"].values, label='True RUL HPC', color='grey', linestyle='--')
    plt.plot(final_rul_hpc, label='LGBM Corrected Pred HPC', color='orange')
    plt.legend()
    plt.title("HPC")
    
    plt.subplot(1,3,2)
    if has_labels: plt.plot(test_data_aligned["Cycles_to_HPT_SV"].values, label='True RUL HPT', color='grey', linestyle='--')
    plt.plot(final_rul_hpt, label='LGBM Corrected Pred HPT', color='orange')
    plt.legend()
    plt.title("HPT")
    
    plt.subplot(1,3,3)
    if has_labels: plt.plot(test_data_aligned["Cycles_to_WW"].values, label='True RUL WW', color='grey', linestyle='--')
    plt.plot(final_rul_ww, label='LGBM Corrected Pred WW', color='orange')
    plt.legend()
    plt.title("WW")
    
    plt.show()

# %store engines
