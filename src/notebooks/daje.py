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
# Import section
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
from matplotlib.path import Path
from collections import deque

# %load_ext autoreload
# %autoreload 2
from tools import utils as u, config as cfg, plotting as up
from tools.types.plotdata import PlotData
from tools.types.enums import *

# %% [markdown]
# #### Original data loading

# %%
# DA RIVEDERE COME DEFINIRE GLI STEP POINTS
# Load original data and event points
otraining = u.load_training()
df = otraining()

events_per_motor = {}
final_events = {}

# Data points extraction
for esn in u.ESN:
    df_motor = df[df['ESN'] == esn].copy().reset_index(drop=True)
    m_wws_fn, m_hpc_fn, m_hpt_fn = u.load_event_points(df_motor)   
    if m_wws_fn is not None:
        events_per_motor[esn] = {
            'wws': m_wws_fn(),
            'hpc': m_hpc_fn(),
            'hpt': m_hpt_fn()
        }
    final_events[esn] = {}
    for type in u.EVENTS:
        df_evento = events_per_motor[esn][type]
        struttura_pulita = df_evento[['Cycles_Since_New']].reset_index()
        struttura_pulita = struttura_pulita.rename(columns={'index': 'Index'})
        final_events[esn][type] = struttura_pulita


# %% [markdown]
# #### Dataset refactoring

# %%
# Dataset refactoring for better visualization and exploration
# NaN handling and ewma

# Questione: abbiamo uniformato la lunghezza dei dati dei sensori con la moving average. 
# Non è un problema! Tanto abbiamo gli event points, quindi recuperiamo i run to failure

save_path = f'{cfg.DATA_BASE_PATH}REFACTOR'

if not os.path.exists(save_path):
    os.makedirs(save_path)

colonne_identificative = ['ESN', 'Cycles_Since_New', 'Snapshot']

df_filtered = df[colonne_identificative + u.SENSORS].copy()

for snap in u.SNAPSHOTS:
    refactored_data = u.refactor_table(df_filtered, snap, 10)
    file_name = f'training_data_snapshot_{snap}.csv'
    full_path = os.path.join(save_path, file_name)
    refactored_data.to_csv(full_path, index=False)
    print(f"Salvato: {full_path}")
