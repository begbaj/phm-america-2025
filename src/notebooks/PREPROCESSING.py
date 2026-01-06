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

#rinomazionazione di alcune colonne per semplicità di scrittura
rename_map = {
    'ESN': 'esn',
    'Snapshot': 'snap',
    'Cumulative_WWs': 'wwCycle',
    'Cumulative_HPC_SVs': 'hpcCycle',
    'Cumulative_HPT_SVs': 'hptCycle'
}
dfp = dfp.rename(columns=rename_map)

#rimozionione Sensed_ dai sensori (clogged view)
sensor_cols = [c for c in dfp.columns if c.startswith('Sensed_')]
sensor_rename_map = {c: c.replace('Sensed_', '') for c in sensor_cols}
dfp = dfp.rename(columns=sensor_rename_map)
final_sensor_names = list(sensor_rename_map.values())

# Nuovi indici
dfp['esnIndex'] = dfp.groupby('esn').cumcount()
dfp['snapIndex'] = dfp.groupby('snap').cumcount()
dfp['wwCycleIndex']  = dfp.groupby(['wwCycle', "snap", "esn"]).cumcount()
dfp['hpcCycleIndex'] = dfp.groupby(['hpcCycle', "snap", "esn"]).cumcount()
dfp['hptCycleIndex'] = dfp.groupby(['hptCycle', "snap", "esn"]).cumcount()

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
    'hpt_cycle'
] + final_sensor_names

dfp = dfp[cols_order]



# for snap_id, group_data in dfp.groupby('snap'):
#     print(f"Scrittura file per SNAP {snap_id}...")
#     filename = f"snapshot_{snap_id}.csv"
#     path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename=filename)
#     # index=False perché 'global_index' è già una colonna esplicita
#     group_data.to_csv(path, index=False)
path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="reformatted.csv")
dfp.to_csv(path, index=False)

print("-- Operazione Completata: Tutti i file sono stati salvati in formato Wide --")
