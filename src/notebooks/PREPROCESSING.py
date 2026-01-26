# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: phm-america-2025 (3.14.2)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocessamento

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
from time import sleep
from scipy.stats import skew, kurtosis

# %load_ext autoreload
# %autoreload 2
from tools import utils as u, config as cfg, plotting as up, features as f
from tools.types.plotdata import PlotData
from tools.types.enums import *

# %% [markdown]
# # Snapshot Table Format
# creiamo un .csv per ogni snapshot (8 in totale) i seguenti campi:
# - _esn (chiave)
# - _sensor (chiave)
# - _index (chiave) (relativo al sensore e all'esn)
# - signal_value

# %%
# 1. PREPARAZIONE INDICI
# Reset dell'indice per preservare l'indice originale come 'global_index'
train = u.load_training()()
dfp = train.reset_index().rename(columns={"index": "global_index"})
del train

## PREPROCESSING

dfp , history = u.preprocess_pipeline(dfp,
                                   outlier_method='isoforest',
                                   outlier_threshold=0.08,
                                   smoothing_window=100,
                                   smoothing_step=25,
                                   )

### PREPROCESSING 

# Rinominazione di alcune colonne per semplicità di scrittura
rename_map = {
    'ESN': 'esn',
    'Snapshot': 'snap',
    'Cumulative_WWs': 'ww_cycle',
    'Cumulative_HPC_SVs': 'hpc_cycle',
    'Cumulative_HPT_SVs': 'hpt_cycle',
    'Cycles_to_WW': 'to_next_ww_cycle',
    'Cycles_to_HPC_SV': 'to_next_hpc_cycle',
    'Cycles_to_HPT_SV': 'to_next_hpt_cycle',
    'Cycles_Since_New': 'cycle'
}
dfp = dfp.rename(columns=rename_map)

# Rimozione Sensed_ dai sensori (clogged view)
sensor_cols = [c for c in dfp.columns if c.startswith('Sensed_')]
sensor_rename_map = {c: c.replace('Sensed_', '') for c in sensor_cols}
dfp = dfp.rename(columns=sensor_rename_map)
final_sensor_names = list(sensor_rename_map.values())

# Nuovi indici
dfp['esn_index'] = dfp.groupby('esn').cumcount()
dfp['snap_index'] = dfp.groupby(["esn", 'snap']).cumcount()
dfp['ww_cycle_index']  = dfp.groupby(['ww_cycle', "snap", "esn"]).cumcount()
dfp['hpc_cycle_index'] = dfp.groupby(['hpc_cycle', "snap", "esn"]).cumcount()
dfp['hpt_cycle_index'] = dfp.groupby(['hpt_cycle', "snap", "esn"]).cumcount()

# Aggiunta della colonna "faulty" per ogni tipo di evento
fault_map = {
    'to_next_ww_cycle': 'fault_ww_cycle',
    'to_next_hpc_cycle': 'fault_hpc_cycle',
    'to_next_hpt_cycle': 'fault_hpt_cycle'
}

for source_col, fault_name in fault_map.items():
    dfp[fault_name] = 0
    dfp.loc[dfp[source_col] == 0, fault_name] = 1
    dfp.loc[dfp.groupby('esn').cumcount(ascending=False) == 0, fault_name] = 1
new_fault_columns = ['fault_ww_cycle', 'fault_hpc_cycle', 'fault_hpt_cycle']

# 4. DEFINIZIONE ORDINE COLONNE
# Definiamo l'ordine esatto in cui vogliamo che appaiano nel CSV Wide
# Prima gli identificatori, poi gli indici di manutenzione, infine i sensori
cols_order = [
    'snap_index',
    'esn',
    'cycle',
    'snap',
    'esn_index',
    'global_index',
    'ww_cycle_index',
    'hpc_cycle_index',
    'hpt_cycle_index',
    'ww_cycle',
    'hpc_cycle',
    'hpt_cycle',
    'to_next_ww_cycle',
    'to_next_hpc_cycle',
    'to_next_hpt_cycle'
] + final_sensor_names + new_fault_columns

dfp = dfp[cols_order]

# for snap_id, group_data in dfp.groupby('snap'):
#     print(f"Scrittura file per SNAP {snap_id}...")
#     filename = f"snapshot_{snap_id}.csv"
#     path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename=filename)
#     # index=False perché 'global_index' è già una colonna esplicita
#     group_data.to_csv(path, index=False)

path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="training.csv")
dfp.to_csv(path, index=False)

print("-- Operazione Completata: Tutti i file sono stati salvati in formato Wide --")
# u.SENSORS = final_sensor_names
del new_fault_columns, sensor_cols, sensor_rename_map

# %%
up.plot_pipeline_comparison(history, "Sensed_T45")

# %% [markdown]
# # estrazione feature basiche statistiche

# %%
# group = ['esn', 'snap']
# window = 50
# sortcols = ['esn', 'esn_index']
# features = ['mean', 'std',]
# sensors = ["Altitude","Mach","Pamb","Pt3","TAT","WFuel","VAFN","VBV","Fan_Speed","Core_Speed","T25","T3","Ps3","T45","P25","T5"]

# dfa = u.preproc_features(dfp, group, sensors, features, sortcols, window_size=window, step=window//2)

# path_feat = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="training_feature_table.csv")
# dfa.to_csv(path_feat, index=False)

# print(f"OPERAZIONE COMPLETATA.")
# print(f"Righe prima: {len(dfp)} | Righe dopo (aggregate): {len(dfa)}")
# print(f"File salvato in: {path_feat}")


# %%
# groups = list(dfa.groupby(['esn', 'snap']))
# n_groups = len(groups)

# # Creiamo una griglia (es. 3 colonne)
# cols = 4
# rows = (n_groups + cols - 1) // cols
# fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), constrained_layout=True)
# axes = axes.flatten() # Rendiamo l'array 1D per iterare facilmente

# for i, ((esn, snap), data) in enumerate(groups):
#     ax = axes[i]
    
#     # Plot dei segnali
#     ax.plot(data["snap_index"], data["T45"], label="Orig", alpha=0.2, color='gray')
#     ax.plot(data["snap_index"], data["T45_std"], label="STD", alpha=0.7, color='orange')
#     ax.plot(data["snap_index"], data["T45_mean"], label="Mean", alpha=0.7, color='red')
    
#     ax.set_title(f"ESN: {esn} | Snap: {snap}")
#     ax.set_xlabel("Snap Index")
#     ax.set_ylabel("T45")
#     ax.legend(loc='upper right', fontsize='small')
#     ax.grid(True, alpha=0.3)

# # Rimuoviamo eventuali axes vuoti se n_groups < rows*cols
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])

# plt.show()


# %% [markdown]
# # Estrazione feature sulle performance del motore

# %%
# PROVA

meta_cols = [
    'cycle', 'snap_index',
    'ww_cycle', 'hpc_cycle', 'hpt_cycle',
    'ww_cycle_index', 'hpc_cycle_index', 'hpt_cycle_index',
    'to_next_ww_cycle', 'to_next_hpc_cycle', 'to_next_hpt_cycle',
    'fault_ww_cycle', 'fault_hpc_cycle', 'fault_hpt_cycle'
]

agg_logic = {}

test = dfp.copy()

# Scegliamo solo ESN e il contatore di riga come chiavi.
group_keys = ['esn', 'cycle']

# SUI SENSORI: facciamo la MEDIA (qui passiamo da 8 righe a 1 riga)
for col in final_sensor_names:
    if col in test.columns:
        agg_logic[col] = 'mean'

# SULLE COLONNE META: prendiamo il PRIMO valore (perché è uguale in tutti gli snapshot di quel momento)
for col in meta_cols:
    if col in test.columns:
        agg_logic[col] = 'first'

# Rieseguiamo l'aggregazione usando queste nuove chiavi
df_averaged = test.groupby(group_keys, as_index=False).agg(agg_logic)

df_averaged = df_averaged.rename(columns={'snap_index': 'esn_index'})
df_averaged = df_averaged.sort_values(['esn', 'esn_index']).dropna()

# 6. SALVATAGGIO
path_avg = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="averaged_final.csv")
df_averaged.to_csv(path_avg, index=False)

print(f"OPERAZIONE COMPLETATA.")
print(f"Righe prima: {len(test)} | Righe dopo (mediate): {len(df_averaged)}")
print(f"Rapporto di compressione: {len(test)/len(df_averaged):.1f}x (dovrebbe essere circa 8.0)")

# %%
#features = [f.FThermalEfficiency.DELTA_HPC, f.FThermalEfficiency.DELTA_PR_TH_HPC, f.FThermalEfficiency.DELTA_PR_TH_HPC_2]
to_calc = df_averaged.copy()
features = [] # [f.FThermalEfficiency.DELTA_PR_TH_HPC_2]
target = 'HPC'
statistical_features = [] # ['mean', 'rms']
fulltarget = f'to_next_{target.lower()}_cycle'
colname = f.get_all_performance_colnames()
skip = 0

if target == "HPC":
    dff, val = f.pipeline_hpc(to_calc, features, colname, statistical_features, window=1, step=0, stat_groupby=["esn"], stat_sortby=["esn", "esn_index"], target=fulltarget)
elif target == "HPT":
    dff, val = f.pipeline_hpt(dfp, features, colname, statistical_features, window=100, step=25, stat_groupby=["esn"], stat_sortby=["esn", "esn_index"], target=fulltarget)
else:
    print("Non se po fa")
    skip = 1

if skip != 1:
    # Estrazione delle migliori feature assolute (considerando tutti i raggruppamenti)
    per = val.sort_values(by='pearson_corr', key=abs, ascending=False).head(10)
    spe = val.sort_values(by='spearman_corr', key=abs, ascending=False).head(10)
    tot = val.sort_values(by='tot_val', key=abs, ascending=False).head(10)

    print("-" * 30)
    print("TOP 10 PEARSON (Across all snaps):")
    print(per)

    print("\nTOP 10 SPEARMAN (Across all snaps):")
    print(spe)

    print("-" * 30)
    print(f"Total Unique Best Features: {len(tot)}")
    print(tot)
    run = u.get_timestamp()



# %%
path_feat = u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"training_feature_{target}_{run}_metadata.csv")
dff.to_csv(path_feat, index=False)
path_feat = u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"training_feature_{target}_{run}_data.csv")
dff[tot["feature"]].to_csv(path_feat, index=False)

# %%
filter_feature = ["DP_TH_HPC_2_MEAN", "DP_TH_HPC_2_RMS", "DP_TH_HPC_2"] #"DP_TH_HPC_2"  # oppure None
max_features_to_show = 10           # Limite per non intasare la RAM
esn_list = [101, 102, 103, 104]     # I motori che vuoi controllare

plots = up.plot_features(dff, esn_list, tot, target, fulltarget, filter_feature, max_features_to_show)
for (fig, figname) in plots:
    path = u.plot_path("features_plots", f"{target}", f"{run}", "Aggregated (snap collapse)" ,filename=figname)
    fig.savefig(path)


# %%
filter_feature = None               #"DP_TH_HPC_2"  # oppure None
max_features_to_show = 10           # Limite per non intasare la RAM
esn_list = [101, 102, 103, 104]     # I motori che vuoi controllare

plots = up.plot_features_per_snap(dff, esn_list, tot, target, fulltarget, filter_feature, max_features_to_show)
for (fig, figname) in plots:
    fig.show()
    path = u.plot_path("features_plots", f"{target}", f"{run}", "Disaggregated (snap divided)" ,filename=figname)
    fig.savefig(path)


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAZIONE FINESTRA ---
WINDOW_SIZE = 5000 # Finestra più piccola per maggiore reattività

# --- STEP 1: Feature Engineering con Finestra Piccola ---
# Calcoliamo medie mobili su una finestra ridotta per i sensori grezzi
sensor_cols = dff[tot["feature"]].columns # ["DP_TH_HPC_2_MEAN", "DP_TH_HPC_2_RMS", "DP_TH_HPC_2"]

for col in sensor_cols:
    dff[f'{col}_smooth_short'] = dff.groupby('esn')[col].transform(
        lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean()
    )

# Reset e Ciclo Relativo
dff['is_reset'] = dff.groupby('esn')['RUL'].diff() > 0
dff['life_id'] = dff.groupby('esn')['is_reset'].cumsum()
dff['relative_cycle'] = dff.groupby(['esn', 'life_id']).cumcount()

# --- STEP 2: Preparazione Dati ---
# Usiamo le nuove feature a finestra corta
features = [f"{col}_smooth_short" for col in sensor_cols] + ["relative_cycle"]
target = "RUL"

train_df = dff[dff["esn"] != 104].dropna()
test_df = dff[dff["esn"] == 104].dropna()

# --- STEP 3: Standardizzazione ---
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[features])
X_test = scaler.transform(test_df[features])
y_train = train_df[target]
y_test = test_df[target]

# --- STEP 4: Modello e Predizione ---
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = np.maximum(0, model.predict(X_test))

# --- STEP 5: Visualizzazione ---
plt.figure(figsize=(15, 6))
plt.plot(test_df.index, y_test.values, label='RUL Reale', color='blue', alpha=0.4)
plt.plot(test_df.index, y_pred, label=f'RUL Predetta (Window={WINDOW_SIZE})', color='red', linewidth=1.5)

# Evidenzia i reset (inizio nuove vite)
for r in test_df[test_df['is_reset']].index:
    plt.axvline(x=r, color='green', linestyle='--', alpha=0.3)

plt.title(f'Predizione RUL con Finestra di {WINDOW_SIZE} Campioni')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- STEP 1: Definizione Feature e Target ---
# Usiamo esattamente quello che hai chiesto
features = ["DP_TH_HPC_2_MEAN", "DP_TH_HPC_2_RMS", "DP_TH_HPC_2", "relative_cycle"]
target = "to_next_hpc_cycle"

# Creiamo l'identificativo dei chunk per non fare confusione tra le vite
# Un chunk finisce e ne inizia un altro quando la RUL aumenta (reset)
dff['is_reset'] = dff.groupby('esn')[target].diff() > 0
dff['chunk_id'] = dff.groupby('esn')['is_reset'].cumsum()

# Feature fondamentale: la posizione relativa nel chunk (ciclo attuale della vita)
dff['relative_cycle'] = dff.groupby(['esn', 'chunk_id']).cumcount()
features_final = features + ['relative_cycle']

# --- STEP 2: Split Train e Test (Motore 104 fuori dal train) ---
train_df = dff[dff["esn"] != 104].dropna(subset=[target] + features)
test_df = dff[dff["esn"] == 104].dropna(subset=[target] + features)

X_train_raw = train_df[features_final]
y_train = train_df[target]
X_test_raw = test_df[features_final]
y_test = test_df[target]

# --- STEP 3: Standardizzazione ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# --- STEP 4: Random Forest Regressor ---
# Usiamo un numero generoso di alberi per catturare bene i chunk
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# --- STEP 5: Predizione ---
y_pred = rf_model.predict(X_test)
y_pred = np.maximum(0, y_pred) # La RUL non può essere negativa


# %%

plt.figure(figsize=(15, 7))

# Plot RUL Reale e Predetta
plt.plot(test_df.index, y_test.values, label='RUL Reale (to_next_hpc_cycle)', color='blue', alpha=0.6, linewidth=2)
plt.plot(test_df.index, y_pred, label='RUL Predetta (Random Forest)', color='red', linestyle='--', alpha=0.9)

# Evidenziamo graficamente i 5 chunk
resets = test_df[test_df['is_reset']].index
for i, r in enumerate(resets):
    plt.axvline(x=r, color='green', linestyle=':', label='Reset / Nuovo Chunk' if i == 0 else "")

plt.title('Predizione RUL per Chunk (Motore 104) - Random Forest')
plt.xlabel('Indice Temporale')
plt.ylabel('Cicli al prossimo HPC')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# Importanza delle feature per capire cosa "pesa" di più nei chunk
importances = pd.Series(rf_model.feature_importances_, index=features_final).sort_values(ascending=False)
print("\nImportanza delle Feature nei Chunk:")
print(importances)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

# --- 1. OTTIMIZZAZIONE MEMORIA E SELEZIONE ---
target = "to_next_hpc_cycle"
features_list = list(tot['feature']) # Solo le feature indicate da te
cols_to_use = list(set(['esn', target] + features_list))

# Lavoriamo su una copia ridotta con precisione float32
df_work = dff[cols_to_use].copy()
for col in features_list:
    df_work[col] = df_work[col].astype(np.float32)

# --- 2. LOGICA DI CHUNK (Reset Manutenzione) ---
# Identifichiamo i reset e calcoliamo il ciclo relativo (fondamentale per la logica temporale)
df_work['is_reset'] = (df_work.groupby('esn')[target].diff() > 0).astype(np.int8)
df_work['relative_cycle'] = df_work.groupby(['esn', df_work['is_reset'].cumsum()]).cumcount().astype(np.int32)

# Feature finali: Solo quelle fornite + la posizione nel chunk (permette di "agganciare" la RUL)
features_final = features_list + ['relative_cycle']

# --- 3. PREPARAZIONE DATASET ---
MAX_RUL = 14000 # Target Piecewise: risolve la "gara" iniziale dei grafici 2-5
train_mask = df_work['esn'] != 104
test_mask = df_work['esn'] == 104

# Conversione in array NumPy per massimizzare la velocità
X_train_raw = df_work.loc[train_mask, features_final].values
y_train = np.minimum(df_work.loc[train_mask, target].values, MAX_RUL)

X_test_raw = df_work.loc[test_mask, features_final].values
y_test_real = df_work.loc[test_mask, target].values

# --- 4. PIPELINE: SCALING + PCA (Fix Errore) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# n_components=0.95 richiede svd_solver='auto' o 'full'
pca = PCA(n_components=0.95) 
X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
X_test_pca = pca.transform(X_test_scaled).astype(np.float32)

print(f"PCA: {len(features_final)} feature ridotte a {pca.n_components_} componenti.")

# --- 5. ADDESTRAMENTO OTTIMIZZATO (XGBoost Hist) ---
# tree_method='hist' è essenziale per la velocità computazionale
xgb = XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist', 
    n_jobs=-1,
    random_state=42,
    # Aggiungiamo regolarizzazione per evitare il "jittering" delle immagini 6 e 7
    reg_lambda=2.0, 
    reg_alpha=0.5
)

param_dist = {
    'n_estimators': [600, 1000],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 5, 7],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# RandomizedSearch per velocità
random_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=10, 
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

random_search.fit(X_train_pca, y_train)
best_model = random_search.best_estimator_

# --- 6. PREDIZIONE E POST-PROCESSING ---
y_pred = best_model.predict(X_test_pca)
# Smoothing finale per pulire la curva (Window di 20 cicli)
y_pred_smooth = pd.Series(y_pred).rolling(window=20, min_periods=1, center=True).mean().values
y_pred_smooth = np.maximum(0, y_pred_smooth)

# --- 7. VISUALIZZAZIONE ---
plt.figure(figsize=(15, 6))
plt.plot(y_test_real, label='RUL Reale', color='royalblue', alpha=0.5, linewidth=2)
plt.plot(y_pred_smooth, label='RUL Predetta (XGB + PCA)', color='crimson', linestyle='--')

plt.title('RUL Ottimizzata (Motore 104) - Solo Feature Selezionate + PCA')
plt.ylabel('Cicli al prossimo HPC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"MAE Finale: {mean_absolute_error(y_test_real, y_pred_smooth):.2f}")

# %%
for g in dff.groupby(['esn', 'snap']):
    plt.figure(figsize=(18, 10))
    plt.title(f"ESN: {g[0]}")
    plt.plot(g[1]['esn_index'], g[1]['PR_ENGINE_GLOBAL_MEAN'], label='THE_DELTA_T_HPT')
    plt.twinx().plot(g[1]['esn_index'], g[1]['to_next_hpt_cycle'], color='orange')
    # plt.vlines(x=g[1].loc[g[1]['fault_hpt_cycle'] == 1, 'esn_index'], color='red', linestyle='--', label='Fault HPT Cycle', ymax=plt.ylim()[1], ymin=plt.ylim()[0])
    plt.xlabel('Cycles')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

# --- 1. CONFIGURAZIONE ---
WINDOW_SIZE = 30  
BATCH_SIZE = 64
LR = 0.0005
EPOCHS = 400
MAX_RUL = 12500    
target_col = "to_next_hpc_cycle"

# Usiamo SOLO le feature originali indicate in tot['feature'] + il ciclo relativo
features_list = list(tot['feature']) + ['relative_cycle']

# --- 2. PREPARAZIONE DATI (Memory Efficient) ---
# Creiamo il df di lavoro con le colonne corrette
df_work = dff[['esn', target_col] + list(tot['feature'])].copy()

# Logica Chunk
df_work['is_reset'] = (df_work.groupby('esn')[target_col].diff() > 0).astype(int)
df_work['chunk_id'] = df_work.groupby('esn')['is_reset'].cumsum()
df_work['relative_cycle'] = df_work.groupby(['esn', 'chunk_id']).cumcount()

# Split train/test
train_mask = df_work['esn'] != 104
test_mask = df_work['esn'] == 104

# Scaling e PCA
scaler = StandardScaler()
# Qui usiamo features_list che è definita sopra e contiene le colonne esistenti
X_train_scaled = scaler.fit_transform(df_work.loc[train_mask, features_list])
X_test_scaled = scaler.transform(df_work.loc[test_mask, features_list])

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
X_test_pca = pca.transform(X_test_scaled).astype(np.float32)

y_train_all = np.minimum(df_work.loc[train_mask, target_col].values, MAX_RUL).astype(np.float32)
y_test_all = df_work.loc[test_mask, target_col].values.astype(np.float32)

# --- 3. CREAZIONE SEQUENZE (Sliding Window per ESN) ---
def create_sequences_by_esn(data, targets, esn_values, window_size):
    sequences, labels = [], []
    unique_esns = np.unique(esn_values)
    
    for esn in unique_esns:
        mask = (esn_values == esn)
        esn_data = data[mask]
        esn_targets = targets[mask]
        
        if len(esn_data) < window_size:
            continue
            
        for i in range(window_size, len(esn_data)):
            sequences.append(esn_data[i-window_size:i])
            labels.append(esn_targets[i])
            
    return np.array(sequences), np.array(labels)

# Creiamo le sequenze separando correttamente i motori
X_train_seq, y_train_seq = create_sequences_by_esn(X_train_pca, y_train_all, df_work.loc[train_mask, 'esn'].values, WINDOW_SIZE)
X_test_seq, y_test_seq = create_sequences_by_esn(X_test_pca, y_test_all, df_work.loc[test_mask, 'esn'].values, WINDOW_SIZE)

class RULDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y).unsqueeze(1)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

train_loader = DataLoader(RULDataset(X_train_seq, y_train_seq), batch_size=BATCH_SIZE, shuffle=True)

# --- 4. ARCHITETTURA TRANSFORMER ---
class RULTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, WINDOW_SIZE, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(model_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        x = self.transformer(x)
        return self.head(x[:, -1, :])

# --- 5. TRAINING ---
# --- 5. TRAINING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize Model
model = RULTransformer(input_dim=X_train_pca.shape[1]).to(device)

# 2. Define Loss Function (The missing part!)
criterion = nn.MSELoss() 

# 3. Define Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print(f"Training su {device}...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Now 'criterion' is defined, so this works:
        loss = criterion(model(batch_x), batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Update learning rate at the end of the epoch
    scheduler.step() 
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# --- 6. PREDIZIONE E PLOT ---
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_test_seq).to(device)).cpu().numpy()

plt.figure(figsize=(15, 6))
plt.plot(y_test_seq, label='RUL Reale (104)', color='blue', alpha=0.5)
plt.plot(y_pred, label='Predizione Transformer', color='red', linestyle='--')
plt.title('RUL Transformer - Motore 104')
plt.legend()
plt.show()

print(f"MAE Transformer: {mean_absolute_error(y_test_seq, y_pred):.2f}")
