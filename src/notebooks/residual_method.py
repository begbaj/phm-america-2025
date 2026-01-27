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
#     display_name: phm-america-2025 (3.11.9)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Import

# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
# %load_ext autoreload
# %autoreload 2
from tools import utils as u, config as cfg, plotting as up

# %% [markdown]
# # Preprocessing

# %%
train = u.load_training()()
# Reset dell'indice per preservare l'indice originale come 'global_index'
dfp = train.reset_index().rename(columns={"index": "global_index"})
del train

## PREPROCESSING effettivo
dfp , history = u.preprocess_pipeline(dfp,
                                   outlier_method='isoforest',
                                   outlier_threshold=0.08,
                                   smoothing_window=50,
                                   smoothing_step=1,
                                   )

engine_ids = dfp['ESN'].unique()

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

# Controllo finale dei NaN
dfp = dfp.dropna()
cols_to_fill = final_sensor_names
#dfp[cols_to_fill] = dfp.groupby(['esn', 'snap'])[cols_to_fill].ffill()
#dfp[cols_to_fill] = dfp.groupby(['esn', 'snap'])[cols_to_fill].bfill()

# Nuovi indici
dfp['esn_index'] = dfp.groupby('esn').cumcount()
dfp['snap_index'] = dfp.groupby(["esn", 'snap']).cumcount()
dfp['ww_cycle_index']  = dfp.groupby(['ww_cycle', "snap", "esn"]).cumcount()
dfp['hpc_cycle_index'] = dfp.groupby(['hpc_cycle', "snap", "esn"]).cumcount()
dfp['hpt_cycle_index'] = dfp.groupby(['hpt_cycle', "snap", "esn"]).cumcount()

# Definiamo l'ordine esatto in cui vogliamo che appaiano nel CSV Wide
cols_order = [
    'esn',
    'cycle',
    'snap',
    'global_index',
    'esn_index',
    'snap_index',    
    'ww_cycle_index',
    'hpc_cycle_index',
    'hpt_cycle_index',
    'ww_cycle',
    'hpc_cycle',
    'hpt_cycle',
    'to_next_ww_cycle',
    'to_next_hpc_cycle',
    'to_next_hpt_cycle'
] + final_sensor_names

dfp = dfp[cols_order]

# Salvataggio
path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="training.csv")
dfp.to_csv(path, index=False)
print("-- Operazione Completata: Tutti i file sono stati salvati in formato Wide --")

u.SENSORS = final_sensor_names
del sensor_cols, sensor_rename_map


# %% [markdown]
# # Calcolo dei residui

# %%
# Creiamo una copia del DataFrame
dfr = dfp.copy()
all_sensors = u.SENSORS 
# Definiamo quali sono i sensori di input, le variabili operative
operating_vars = ['Altitude', 'Mach', 'Pamb', 'TAT', 'VAFN', 'VBV', 'Fan_Speed', 'Pt2']
# Definiamo quali sono i sensori di output, le variabili di degrado
degradation_vars = [s for s in all_sensors if s not in operating_vars]
# Creiamo le colonne dei residui
res_cols = [f"{s}_res" for s in degradation_vars]
# Inizializziamo le colonne nel dataframe principale con NaN
for col in res_cols:
    dfr[col] = np.nan
# Definisco la map per il rename delle colone
rename_vars_map = {
    'WFuel': 'WFuel_res',
    'Core_Speed': 'Core_Speed_res',
    'T25': 'T25_res',
    'T3': 'T3_res',
    'Ps3': 'Ps3_res',
    'T45': 'T45_res',
    'P25': 'P25_res',
    'T5': 'T5_res'
}

for esn in engine_ids:
    for snap in u.SNAPSHOTS:
        # 1. Filtro i dati per il singolo motore e il singolo snapshot
        dfr_engine = dfr[(dfr['esn'] == esn) & (dfr['snap'] == snap)].sort_values('snap_index')

        # 2. Selezioniamo il "periodo sano"
        train_data = dfr_engine.iloc[:20]
        
        X_train = train_data[operating_vars]
        y_train = train_data[degradation_vars]
        
        # 3. Addestro il modello lineare f(s_o)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 4. Applico il modello a tutta la vita del motore per vedere la differenza
        X_all = dfr_engine[operating_vars]
        y_real = dfr_engine[degradation_vars]
        
        # f(s_o) predetto
        y_pred = model.predict(X_all)
        
        # 5. Calcolo il residuo: r = s_d - f(s_o)
        residuals = y_real - y_pred
        # Rinomino per far coincidere i nomi delle colonne tra residuals e dfr
        residuals = residuals.rename(columns=rename_vars_map)
        
        # Salvo i residui nel dataframe principale
        dfr.loc[dfr_engine.index, res_cols] = residuals


cols_to_save = [c for c in dfr.columns if c not in res_cols] + res_cols
dfr = dfr[cols_to_save]
# Salvataggio
path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="training_with_residuals.csv")
dfr.to_csv(path, index=False)
print("-- Operazione Completata: Tutti i file sono stati salvati in formato Wide --")


# %% [markdown]
# #### Plotting dei residui

# %%
# Plotting della media dei residui
event = 'hpt'
to_next_col = f'to_next_{event}_cycle'
dfr_mean = dfr.copy()
dfr_mean[res_cols] = dfr_mean.groupby(['esn', 'snap'])[res_cols].transform(
    lambda x: x.rolling(window=40, min_periods=1).mean()
)
dfr_mean[res_cols] = dfr_mean.groupby(['esn', 'snap'])[res_cols].bfill()
dfr_mean[to_next_col] = dfr[to_next_col]
# up.plot_residuals_dashboard(dfr_mean, res_cols)

# %%
to_next_cols = []
events = u.EVENTS
for event in events:
    to_next_cols.append(f'to_next_{event}_cycle')
# Plotting con filtro mediana
dfr_median = dfr_mean.copy()
# Definisco le logiche di aggregazione
agg_logic = {col: 'median' for col in res_cols}
agg_logic.update({col: 'first' for col in to_next_cols})
# Creo i residui "Engine-Level Residuals" facendo la mediana tra gli snapshot
dfr_median = dfr_mean.groupby(['esn', 'snap_index']).agg(agg_logic).reset_index()
# Salvataggio
path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="training_with_residuals_median.csv")
dfr_median.to_csv(path, index=False)
print("-- Operazione Completata: Tutti i file sono stati salvati in formato Wide --")
# Plotting
up.plot_engine_level_residuals(dfr_median, res_cols, to_next_col, event)

# %% [markdown]
# ## HPT Health Index

# %%
to_next_hpt_col = 'to_next_hpt_cycle'
to_next_hpc_col = 'to_next_hpc_cycle'
event = 'hpt'
# Calcolo HPT Health Index
dfr_hpt = dfr_median.copy()
dfr_hpt = u.calculate_hpt_health_index_all(dfr_hpt, 'T3_res', 'T45_res', to_next_hpc_col)

# Mapping dei cicli
dfr_hpt, params = u.fit_hpt_mapping(dfr_hpt, to_next_hpt_col)

# Plotting
up.plot_engine_level_hi(dfr_hpt, ['HI_HPT'], to_next_hpt_col, event)
