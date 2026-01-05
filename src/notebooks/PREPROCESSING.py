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
df_processed = train.reset_index().rename(columns={"index": "global_index"})

# Calcoliamo l'indice relativo al motore (progressivo per ogni ESN)
df_processed['esn_index'] = df_processed.groupby('ESN').cumcount()
# Calcoliamo l'indice relativo allo snapshot (progressivo per ogni Snapshot)
df_processed['snap_index'] = df_processed.groupby('Snapshot').cumcount()

rename_map = {
    'ESN': 'esn',
    'Cumulative_WWs': 'ww_cycle',
    'Cumulative_HPC_SVs': 'hpc_cycle',
    'Cumulative_HPT_SVs': 'hpt_cycle'
}

df_processed = df_processed.rename(columns=rename_map)

# 3. PULIZIA NOMI SENSORI (Opzionale ma consigliato)
# "Sensed_Altitude" -> "Altitude"
sensor_cols = [c for c in df_processed.columns if c.startswith('Sensed_')]
sensor_rename_map = {c: c.replace('Sensed_', '') for c in sensor_cols}
df_processed = df_processed.rename(columns=sensor_rename_map)
final_sensor_names = list(sensor_rename_map.values())

# 4. DEFINIZIONE ORDINE COLONNE
# Definiamo l'ordine esatto in cui vogliamo che appaiano nel CSV Wide
# Prima gli identificatori, poi gli indici di manutenzione, infine i sensori
cols_order = [
    'esn',
    'Snapshot',
    'global_index',
    'esn_index',
    'snap_index',
    'ww_cycle',
    'hpc_cycle',
    'hpt_cycle'
] + final_sensor_names

df_processed = df_processed[cols_order]

for snap_id, group_data in df_processed.groupby('Snapshot'):
    print(f"Scrittura file per SNAP {snap_id}...")
    filename = f"snapshot_{snap_id}"
    path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename=filename)
    # index=False perché 'global_index' è già una colonna esplicita
    group_data.to_csv(path, index=False)

print("-- Operazione Completata: Tutti i file sono stati salvati in formato Wide --")

# %%
# 1. DEFINIZIONE DELLE COLONNE INDICE
# Queste sono le colonne che identificano univocamente una riga nel formato Wide.
# In pratica, tutto ciò che NON è 'sensor' o 'signal_value'.
index_cols = [
    'esn', 
    'global_index', 
    'esn_index', 
    'snap_index', 
    'ww_maint_idx', 
    'hpc_maint_idx', 
    'hpt_maint_idx'
]
df = pd.read_csv(u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="snapshot_1"))
# 2. PIVOTING (Trasformazione)
# Usiamo pivot() che è più performante di pivot_table() quando non ci sono duplicati
df_wide_restored = df.pivot(
    index=index_cols,       # Le colonne che rimangono fisse (diventeranno l'indice temporaneo)
    columns='sensor',       # I valori di questa colonna diventano le nuove Intestazioni
    values='signal_value'   # I valori da inserire nelle celle
)

# 3. PULIZIA STRUTTURALE
# Il pivot crea un MultiIndex. Usiamo reset_index per farlo tornare un DataFrame piatto.
df_wide_restored = df_wide_restored.reset_index()

# Rimuoviamo il nome 'sensor' che rimane sopra le colonne dopo il pivot
df_wide_restored.columns.name = None

# (OPZIONALE) Se vuoi rimettere il prefisso "Sensed_" ai nomi delle colonne:
# Identifichiamo le colonne che non sono negli index_cols
sensor_columns = [c for c in df_wide_restored.columns if c not in index_cols]
rename_map = {c: f"Sensed_{c}" for c in sensor_columns}
df_wide_restored = df_wide_restored.rename(columns=rename_map)

# Visualizzazione risultato
print("Dimensioni Wide:", df_wide_restored.shape)
print(df_wide_restored.head())
