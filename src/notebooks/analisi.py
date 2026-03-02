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
# TUTTI GLI IMPORT

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tools import utils as u, config as cfg, plotting as up, preprocessing as pp
import tools

# %load_ext autoreload
# %autoreload 2

# %%
SENSORS = tools.types.enums.SENSORS

# %%
# TRAINING
df = u.load_training()
df = pp.remove_outliers(df, SENSORS)
df = pp.missingfill(df).dropna()

# %%
# Print delle matrici di correlazione per ogni ESN

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(25, 20))
axes = axes.flatten()

esn_list = df['ESN'].unique()

for i, esn in enumerate(esn_list[:4]):
    ax = axes[i]
    
    group = df[df['ESN'] == esn]
    corr_matrix = group.select_dtypes(include=['number']).drop(columns=['ESN'], errors='ignore').corr()
    
    sns.heatmap(
        corr_matrix, 
        ax=ax, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        annot_kws={"size": 9},
        cbar=True
    )
    
    ax.set_title(f"ESN: {esn}", fontsize=14, fontweight='bold')
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout(pad=4.0)
plt.show()

# %%
# Print delle matrici di correlazione per ogni snapshot di ogni ESN

esn_list = df['ESN'].unique()
snap_list = df['Snapshot'].unique()

for k, esn in enumerate(esn_list[:4]):

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 20))
    axes = axes.flatten()

    for i, snap in enumerate(snap_list[:8]):
        ax = axes[i]
    
        group = df[(df['ESN'] == esn) & (df['Snapshot'] == snap)]
        corr_matrix = group.select_dtypes(include=['number']).drop(columns=['ESN', 'Snapshot'], errors='ignore').corr()
    
        sns.heatmap(
            corr_matrix, 
            ax=ax, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            annot_kws={"size": 9},
            cbar=True
        )
        
        ax.set_title(f"Snapshot: {snap}", fontsize=9)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout(pad=4.0)
    plt.suptitle(f'ESN: {esn}', fontsize=14, fontweight='bold')
    plt.show()
