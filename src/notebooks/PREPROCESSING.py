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
# # Preprocessamento

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
from time import sleep

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

# Rinomazionazione di alcune colonne per semplicità di scrittura
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

# Rimozionione Sensed_ dai sensori (clogged view)
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



for snap_id, group_data in dfp.groupby('snap'):
    print(f"Scrittura file per SNAP {snap_id}...")
    filename = f"snapshot_{snap_id}.csv"
    path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename=filename)
    # index=False perché 'global_index' è già una colonna esplicita
    group_data.to_csv(path, index=False)
#path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="reformatted.csv")
#dfp.to_csv(path, index=False)

print("-- Operazione Completata: Tutti i file sono stati salvati in formato Wide --")

# %%
dfp['ww_cycle_index']  = dfp.groupby(['ww_cycle', "snap", "esn"]).cumcount()
dfp['hpc_cycle_index'] = dfp.groupby(['hpc_cycle', "snap", "esn"]).cumcount()
dfp['hpt_cycle_index'] = dfp.groupby(['hpt_cycle', "snap", "esn"]).cumcount()

meta_cols = [
    'ww_cycle', 'hpc_cycle', 'hpt_cycle',
    'ww_cycle_index', 'hpc_cycle_index', 'hpt_cycle_index',
    'to_next_ww_cycle', 'to_next_hpc_cycle', 'to_next_hpt_cycle',
    'fault_ww_cycle', 'fault_hpc_cycle', 'fault_hpt_cycle'
]

agg_logic = {}

# Creiamo un nuovo indice pulito che conta le righe PER OGNI SNAPSHOT e PER OGNI MOTORE
dfp['esn_index'] = dfp.groupby(['snap', 'esn']).cumcount()

# Scegliamo solo ESN e il contatore di riga come chiavi.
group_keys = ['esn', 'esn_index']

# SUI SENSORI: facciamo la MEDIA (qui passiamo da 8 righe a 1 riga)
for col in final_sensor_names:
    if col in dfp.columns:
        agg_logic[col] = 'mean'

# SULLE COLONNE META: prendiamo il PRIMO valore (perché è uguale in tutti gli snapshot di quel momento)
for col in meta_cols:
    if col in dfp.columns:
        agg_logic[col] = 'first'

# Rieseguiamo l'aggregazione usando queste nuove chiavi
df_averaged = dfp.groupby(group_keys, as_index=False).agg(agg_logic)

df_averaged = df_averaged.sort_values(['esn', 'esn_index'])

# 6. SALVATAGGIO
path_avg = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="averaged_final.csv")
df_averaged.to_csv(path_avg, index=False)

print(f"OPERAZIONE COMPLETATA.")
print(f"Righe prima: {len(dfp)} | Righe dopo (mediate): {len(df_averaged)}")
print(f"Rapporto di compressione: {len(dfp)/len(df_averaged):.1f}x (dovrebbe essere circa 8.0)")

