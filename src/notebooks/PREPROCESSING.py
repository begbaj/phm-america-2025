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
# Assicuriamoci che l'indice attuale sia accessibile come colonna 'global_index'
df_processed = train.reset_index().rename(columns={"index": "global_index"})

# Calcoliamo l'indice relativo al motore (progressivo per ogni ESN)
df_processed['esn_index'] = df_processed.groupby('ESN').cumcount()

# Calcoliamo l'indice relativo allo snapshot (progressivo per ogni Snapshot)
df_processed['snap_index'] = df_processed.groupby('Snapshot').cumcount()

# 2. SELEZIONE COLONNE
# Identifichiamo automaticamente tutte le colonne che iniziano con "Sensed_"
sensor_cols = [c for c in df_processed.columns if c.startswith('Sensed')]

# Definiamo le colonne ID che vogliamo mantenere fisse
id_cols = ['ESN', 'Snapshot', 'global_index', 'esn_index', 'snap_index']

# 3. TRASFORMAZIONE (MELT)
# Trasformiamo da Wide a Long: crea una riga per ogni singolo valore di sensore
df_long = df_processed.melt(
    id_vars=id_cols, 
    value_vars=sensor_cols, 
    var_name='sensor', 
    value_name='signal_value'
)

# Rinominiamo la colonna ESN in esn (minuscolo) come richiesto
df_long = df_long.rename(columns={'ESN': 'esn'})
df_long['sensor'] = df_long['sensor'].str.replace('Sensed_', '')
# 4. SALVATAGGIO PER SNAPSHOT
# Invece di un loop manuale, usiamo groupby: sicuro, veloce e automatico.
print("Inizio salvataggio file...")

for snap_id, group_data in df_long.groupby('Snapshot'):
    print(f"Scrittura SNAP {snap_id}...")
    
    # Selezioniamo solo le colonne richieste nell'ordine specifico
    final_output = group_data[['esn', 'sensor', 'snap_index', 'esn_index', 'global_index', 'signal_value']]
    
    # Costruiamo il path dinamico
    path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename=f"snapshot_{snap_id}")
    
    # Salviamo (index=False perché i nostri indici sono già colonne dati)
    final_output.to_csv(path, index=False)

print("--Tutti i file salvati correttamente--")
