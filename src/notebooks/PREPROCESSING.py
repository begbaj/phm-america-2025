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
#     display_name: phm-america-2025
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
from tools import utils as u, config as cfg, algorithms as alg, plotting as up
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

# Rinominazione di alcune colonne per semplicità di scrittura
rename_map = {
    'ESN': 'esn',
    'Snapshot': 'snap',
    'Cumulative_WWs': 'ww_cycle',
    'Cumulative_HPC_SVs': 'hpc_cycle',
    'Cumulative_HPT_SVs': 'hpt_cycle',
    'Cycles_to_WW': 'to_next_ww_cycle',
    'Cycles_to_HPC_SV': 'to_next_hpc_cycle',
    'Cycles_to_HPT_SV': 'to_next_hpt_cycle'
}
dfp = dfp.rename(columns=rename_map)

# Rimozione Sensed_ dai sensori (clogged view)
sensor_cols = [c for c in dfp.columns if c.startswith('Sensed_')]
sensor_rename_map = {c: c.replace('Sensed_', '') for c in sensor_cols}
dfp = dfp.rename(columns=sensor_rename_map)
final_sensor_names = list(sensor_rename_map.values())

# Nuovi indici
dfp['esn_index'] = dfp.groupby('esn').cumcount()
dfp['snap_index'] = dfp.groupby('snap').cumcount()
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
u.SENSORS = final_sensor_names
del final_sensor_names, new_fault_columns, sensor_cols, sensor_rename_map

# %%
group = ['esn', 'snap']
window = 50
sortcols = ['esn', 'esn_index']
features = ['mean', 'std',]
sensors = ["Altitude","Mach","Pamb","Pt2","TAT","WFuel","VAFN","VBV","Fan_Speed","Core_Speed","T25","T3","Ps3","T45","P25","T5"]

dfa = u.preproc_features(dfp, group, sensors, features, sortcols, window_size=window, step=window//2)

path_feat = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="training_feature_table.csv")
dfa.to_csv(path_feat, index=False)

print(f"OPERAZIONE COMPLETATA.")
print(f"Righe prima: {len(dfp)} | Righe dopo (aggregate): {len(dfa)}")
print(f"File salvato in: {path_feat}")


# %%
groups = list(dfa.groupby(['esn', 'snap']))
n_groups = len(groups)

# Creiamo una griglia (es. 3 colonne)
cols = 4
rows = (n_groups + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), constrained_layout=True)
axes = axes.flatten() # Rendiamo l'array 1D per iterare facilmente

for i, ((esn, snap), data) in enumerate(groups):
    ax = axes[i]
    
    # Plot dei segnali
    ax.plot(data["snap_index"], data["T45"], label="Orig", alpha=0.2, color='gray')
    ax.plot(data["snap_index"], data["T45_std"], label="STD", alpha=0.7, color='orange')
    ax.plot(data["snap_index"], data["T45_mean"], label="Mean", alpha=0.7, color='red')
    
    ax.set_title(f"ESN: {esn} | Snap: {snap}")
    ax.set_xlabel("Snap Index")
    ax.set_ylabel("T45")
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)

# Rimuoviamo eventuali axes vuoti se n_groups < rows*cols
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()

