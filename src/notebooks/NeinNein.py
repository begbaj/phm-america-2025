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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
# %load_ext autoreload
# %autoreload 2
from tools import utils as u, config as cfg, plotting as up, preprocessing as pp

# %%
def train_models(df) -> dict[int, dict[str,LinearRegression]]:
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



# %%
from pyparsing import line
from sympy import O


df = u.load_training()()
cycles = ['Cycles_Since_New']
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
degradation_vars = [s for s in u.SENSORS if s not in operating_vars]

df = pp.remove_outliers(df, u.SENSORS)
df = pp.missingfill(df).dropna()

testing_esn = 103
test_data = df[df["ESN"] == testing_esn].reset_index()
X_test = test_data[operating_vars]
Y_test = test_data[degradation_vars]
models = train_models(df[df["ESN"].isin([x for x in [101,102,103,104] if x != testing_esn])])

model_i = 0
model = models[model_i]['model']
Y_pred = model.predict(np.roll(X_test, model_i, axis=1))

twes = TWE(Y_pred, Y_test, 10, 3)

res = Y_test - Y_pred
res[cycles] = test_data[cycles]
agg_logic = {col: 'median' for col in degradation_vars}
# agg_logic.update({col: 'first' for col in cycles})
# Creo i residui "Engine-Level Residuals" facendo la mediana tra gli snapshot
res = res.groupby('Cycles_Since_New', as_index=False).agg(agg_logic).reset_index(drop=True)
res = pp.remove_outliers(res, u.SENSORS, threshold=3)
res = res.dropna()




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

# Abbiamo calcolato alpha per ati normalizzati, bisogna normalizzare anche qui!
def HI(res, alpha, wind):
    wind = int(np.round(wind))
    step = 1
    res_smooth = res.rolling(window=wind, step=step, min_periods=1).median().dropna()
    T3_res = minmax(res_smooth, "Sensed_T3")
    T45_res = minmax(res_smooth, "Sensed_T45")
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
    return np.sqrt(np.mean((hi - RUL)**2)) + 1

# def objective_full_hpt(params, res, df):
#     alpha, wind = params
#     wind = int(wind)
#     if wind < 5: wind = 5 
#     s = max(1, wind // 5)
#     res = median_norm(res.rolling(window=wind, step=s, min_periods=1).median())
#     scaled=df.copy()
#     scaled["Cycles_to_HPT_SV"] = minmax(df, "Cycles_to_HPT_SV")
#     ST3_res = minmax(res, "Sensed_T3")
#     ST45_res = minmax(res, "Sensed_T45")
#     rul = scaled.loc[scaled["ESN"] == testing_esn, "Cycles_to_HPT_SV"].reset_index(drop=True)
#     hi = -alpha*ST3_res - ST45_res
#     return np.sqrt(np.mean((hi - rul)**2)) + 1

def objective_full_hpt(params, res_df, full_df):
    alpha, wind = params
    # Protezione per la finestra (deve essere intera e >= 5)
    w = int(np.round(wind))
    if w < 5: w = 5
    # Lo step DEVE essere 1
    s = 1
    # 1. Smoothing (applichiamo la finestra variabile)
    res_smooth = res_df.rolling(window=w, step=s, min_periods=1).median().dropna()
    # res_smooth = res_smooth - res_smooth.median()
    # 2. Scaling locale dei sensori (per rendere l'HI sensibile alla finestra)
    def scale(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
    st3 = scale(res_smooth["Sensed_T3"])
    st45 = scale(res_smooth["Sensed_T45"])
    # 3. Calcolo HI
    hi = -alpha * st3 - st45
    # 4. Allineamento Target RUL
    rul_raw = full_df.loc[full_df["ESN"] == testing_esn, "Cycles_to_HPT_SV"].values
    #res_smooth['rul_sampled'] = rul_raw[::s] # Campioniamo con lo stesso passo dello smoothing
    # Normalizziamo la RUL tra 0 e 1
    rul_norm = (rul_raw - rul_raw.min()) / (rul_raw.max() - rul_raw.min() + 1e-9)
    # Pareggio lunghezze
    L = min(len(hi), len(rul_norm))
    # 5. Normalizziamo l'HI solo per il calcolo dell'errore (opzionale ma consigliato)
    hi_for_error = (hi[:L] - hi[:L].min()) / (hi[:L].max() - hi[:L].min() + 1e-9)
    return np.sqrt(np.mean((hi_for_error.values - rul_norm[:L])**2))

# def objective_full_hpc(params, res, df):
#     alpha, wind = params
#     wind = int(wind)
#     if wind < 5: wind = 5 
#     s = max(1, wind // 5)
#     res = median_norm(res.rolling(window=wind, step=s, min_periods=1).median())
#     scaled=df.copy()
#     scaled["Cycles_to_HPC_SV"] = minmax(df, "Cycles_to_HPC_SV")
#     ST3_res = minmax(res, "Sensed_T3")
#     ST45_res = minmax(res, "Sensed_T45")
#     rul = scaled.loc[scaled["ESN"] == testing_esn, "Cycles_to_HPC_SV"].reset_index(drop=True)
#     hi = -alpha*ST3_res - ST45_res
#     return np.sqrt(np.mean((hi - rul)**2)) + 1

def objective_full_hpc(params, res_df, full_df):
    alpha, wind = params
    # Protezione per la finestra (deve essere intera e >= 5)
    w = int(np.round(wind))
    if w < 5: w = 5
    # Lo step DEVE essere 1
    s = 1
    # 1. Smoothing (applichiamo la finestra variabile)
    res_smooth = res_df.rolling(window=w, step=s, min_periods=1).median().dropna()
    # res_smooth = res_smooth - res_smooth.median()
    # 2. Scaling locale dei sensori (per rendere l'HI sensibile alla finestra)
    def scale(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
    st3 = scale(res_smooth["Sensed_T3"])
    st45 = scale(res_smooth["Sensed_T45"])
    # 3. Calcolo HI
    hi = -alpha * st3 - st45
    # 4. Allineamento Target RUL
    rul_raw = full_df.loc[full_df["ESN"] == testing_esn, "Cycles_to_HPC_SV"].values
    #res_smooth['rul_sampled'] = rul_raw[::s] # Campioniamo con lo stesso passo dello smoothing
    # Normalizziamo la RUL tra 0 e 1
    rul_norm = (rul_raw - rul_raw.min()) / (rul_raw.max() - rul_raw.min() + 1e-9)
    # Pareggio lunghezze
    L = min(len(hi), len(rul_norm))
    # 5. Normalizziamo l'HI solo per il calcolo dell'errore (opzionale ma consigliato)
    hi_for_error = (hi[:L] - hi[:L].min()) / (hi[:L].max() - hi[:L].min() + 1e-9)
    return np.sqrt(np.mean((hi_for_error.values - rul_norm[:L])**2))


# window = 370
# step = window//5
# res = res.rolling(window, step).median()
# res = median_norm(res)

# scaled = df[["Cycles_to_HPT_SV", "Cycles_to_HPC_SV"]].copy()
# scaled_res = res.copy()

# scaled["Cycles_to_HPT_SV"] = minmax(scaled, "Cycles_to_HPT_SV")
# scaled["Cycles_to_HPC_SV"] = minmax(scaled, "Cycles_to_HPC_SV")
# ST3_res = minmax(res, "Sensed_T3")
# ST45_res = minmax(res, "Sensed_T45")

agg_logic = {col: 'median' for col in degradation_vars}
# agg_logic.update({col: 'first' for col in cycles})
# Creo i residui "Engine-Level Residuals" facendo la mediana tra gli snapshot

hpt_rul_df = df.loc[df["ESN"] == testing_esn, ["Cycles_Since_New", "Cycles_to_HPT_SV"]]
hpc_rul_df = df.loc[df["ESN"] == testing_esn, ["Cycles_Since_New", "Cycles_to_HPC_SV"]]

hpt_rul = hpt_rul_df.groupby('Cycles_Since_New', as_index=False).agg('first').reset_index(drop=True)
hpt_rul = hpt_rul.drop(columns=['Cycles_Since_New'])
hpc_rul = hpc_rul_df.groupby('Cycles_Since_New', as_index=False).agg('first').reset_index(drop=True)
hpc_rul = hpc_rul.drop(columns=['Cycles_Since_New'])


a_hpt = 1.5
a_hpc = 1.5

# Cambiando questo valore, non cambia nulla
# inoltre, la funzione minimize non ne ottimizza il valore, cambia solo alpha
w_hpt = 100
w_hpc = 100

bounds = [
    (None, None),
    (0, 400)
]

result_hpt = minimize(
    objective_full_hpt,
    x0=(a_hpt,w_hpt),
    args=(res, df),
    method="Nelder-Mead",
    bounds=bounds
    #L-BFGS-B
)

result_hpc = minimize(
    objective_full_hpc,
    x0=(a_hpt,w_hpc),
    args=(res, df),
    method="Nelder-Mead",
    bounds=bounds
)

# result_hpt = minimize(
#     objective, 
#     x0=a_hpt, 
#     args=(ST3_res, ST45_res, hpt_rul),
#     method='Nelder-Mead'
# )

# result_hpc = minimize(
#     objective, 
#     x0=a_hpc, 
#     args=(ST3_res, ST45_res, hpc_rul),
#     method='Nelder-Mead'
# )

a_opt_hpt = result_hpt.x[0]
a_opt_hpc = result_hpc.x[0]
w_opt_hpt = int(np.round(result_hpt.x[1]))
w_opt_hpc = int(np.round(result_hpc.x[1]))


print(f"alpha_hpt:{a_opt_hpt}")
print(f"alpha_hpc:{a_opt_hpc}")
print("")
print(f"window_hpt:{w_opt_hpt}")
print(f"window_hpc:{w_opt_hpc}")

hi_hpt = HI(res, a_opt_hpt, w_opt_hpt)
hi_hpc = HI(res, a_opt_hpc, w_opt_hpc)
error_hpt = np.sum(hi_hpt - hpt_rul)
error_hpc = np.sum(hi_hpc - hpc_rul)



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
