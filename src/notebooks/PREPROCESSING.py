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
from tools import utils as u, config as cfg, algorithms as alg, plotting as up, features as f
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

# %% [markdown]
# # estrazione feature basiche statistiche

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


# %% [markdown]
# # Estrazione feature sulle performance del motore

# %%
features = [f.FThermalEfficiency.DELTA_HPC, f.FThermalEfficiency.DELTA_PR_TH_HPC, f.FThermalEfficiency.DELTA_PR_TH_HPC_2]
target = 'HPC'
fulltarget = f'to_next_{target.lower()}_cycle'
colname = f.get_all_performance_colnames()
dff = f.performance_features(dfp, features)
dff = f.calc_statistical_features(dff, features=['mean'], columns=colname, groupby=["esn", "snap"], window_size=50, step=25).dropna()
# val = f.evaluate_correlation(dff, target='to_next_hpt_cycle', groupby=['esn', 'snap'])
val = f.evaluate_correlation_per_snap(dff, target=fulltarget, top_n=10)

# 4. Creazione della Pivot Table per il confronto
# Mostra come la correlazione di Spearman cambia per la stessa feature tra diversi snap
feature_evolution = val.pivot(index='snap', columns='feature', values='spearman_corr')

# 5. Estrazione delle migliori feature assolute (considerando tutti i raggruppamenti)
per = val.sort_values(by='pearson_corr', key=abs, ascending=False).head(10)
spe = val.sort_values(by='spearman_corr', key=abs, ascending=False).head(10)
tot = val.sort_values(by='tot_val', key=abs, ascending=False).head(10)

print("-" * 30)
print("TOP 10 PEARSON (Across all snaps):")
print(per[['snap', 'feature', 'pearson_corr', 'n_samples']])

print("\nTOP 10 SPEARMAN (Across all snaps):")
print(spe[['snap', 'feature', 'spearman_corr', 'n_samples']])


print("-" * 30)
print(f"Total Unique Best Features: {len(tot)}")
print(tot[['snap', 'feature', 'pearson_corr', 'spearman_corr']])

# %%
max_esn_to_plot = 1
for esn in range(101, 105):
    esn_data = dff[dff['esn'] == esn]
    for snap in esn_data['snap'].unique():
        group_data = esn_data[esn_data['snap'] == snap].sort_values('esn_index')
        snap_best_features = val[val['snap'] == snap]
        snap_best_features = snap_best_features[snap_best_features['feature'] == "DP_TH_HPC_2" ]
        
        for _, row in snap_best_features.iterrows():
            feat_name = row['feature']

            # Inizializzazione Figura
            fig, ax1 = plt.subplots(figsize=(18, 10))
            
            plt.title(f"MOTOR ANALYSIS | ESN: {esn} - Phase: {snap}\nFeature: {feat_name}")
            ax1.set_xlabel('Time (Cycles)')
            ax1.set_ylabel(f'Value: {feat_name}', color='tab:blue', fontweight='bold')

            # 1. Plot della Feature reale
            l1, = ax1.plot(group_data['esn_index'], group_data[feat_name], 
                           label=f'Actual {feat_name}', color='tab:blue', linewidth=2, marker='o', markersize=4, alpha=0.7)

            # 3. Linee verticali per i guasti (HPT Fault)
            fault_indices = group_data.loc[group_data[f'fault_{target.lower()}_cycle'] == 1, 'esn_index']
            for f_idx in fault_indices:
                ax1.axvline(x=f_idx, color='red', linestyle=':', alpha=0.8, label='Fault Event')

            fault_indices = group_data.loc[group_data[f'fault_ww_cycle'] == 1, 'esn_index']
            for f_idx in fault_indices:
                ax1.axvline(x=f_idx, color='red', linestyle=':', alpha=0.8, label='Fault Event')

            # 4. Secondo Asse (Destra) per il Target (RUL)
            ax2 = ax1.twinx()
            ax2.set_ylabel(f'RUL (To Next {target})', color='tab:orange', fontweight='bold')
            l3, = ax2.plot(group_data['esn_index'], group_data[fulltarget], 
                           color='tab:orange', label='Target (RUL)', alpha=0.8, linewidth=2)


            # --- Estetica e Legenda ---
            ax1.grid(True, linestyle='--', alpha=0.5)
            lines = [l1, l3]
            labels = [l.get_label() for l in lines]
            # Evitiamo duplicati nella legenda per il Fault Event
            unique_labels = dict(zip(labels, lines))
            ax1.legend(unique_labels.values(), unique_labels.keys(), loc='lower left', frameon=True, shadow=True)

            plt.tight_layout()
            plt.show()

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

