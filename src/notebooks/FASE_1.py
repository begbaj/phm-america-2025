# ---
# jupyter:
#   jupytext:
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
# # Importazione ed Analisi Dati
# Attenzione: TUTTI gli import DEVONO stare nel primo blocco.
# Questo notebook NON deve contenere logiche di plotting o di manipolazione dati se non operazioni essenziali. Queste logiche vanno definite in tools.plotting e tools.algorithms.

# %%
import pandas as pd
import matplotlib as plt
import numpy as np
import scipy as sp

# %load_ext autoreload
# %autoreload 2
from tools import utils as u, config as cfg, algorithms as alg, plotting as up

# %%
otraining = u.load_training()
owws, ohpc, ohpt = u.load_event_points(orig_training())
# load_training testing e validation restituiscono un la funzione WrapData per mantenere il dato originale intatto senza modifiche.
# per accedere al dato e copiarlo basterà chiamarla come funzione che copierà il dataframe originale in una nuova variabile.
df = otraining()
wws = owws()
hpc = ohpc()
hpt = ohpt()


# %%
def printcent(s: str):
    print()
    print("-"*10 + s.title())

printcent("Dimensione")
print(f"righe: {df.shape[0]}")
print(f"colonne: {df.shape[1]}")

printcent("Righe per motore") 
print(f"{df.groupby('ESN').size()}")

printcent("Valori nulli")
print(u.df_row_filter(df.isnull().sum()))

printcent("Distribuzione dei valori nulli")
n = u.df_col_filter(df.groupby('ESN').apply(lambda x: x.isnull().sum(), include_groups=False))
print(n)
del n


# %% [markdown]
# # Medie e Deviazione standard

# %%
# wws_counts = wws.groupby('ESN').size().reset_index(name='Totale_Eventi_WW')
# wws['cycles_between_ww'] = wws.groupby('ESN')['Cycles_Since_New'].diff()
# ww_stats = wws.groupby('ESN').agg(
#     Media_Intervallo_WW=('cycles_between_ww', 'mean'),
#     DevStd_Intervallo_WW=('cycles_between_ww', 'std'),
#     Num_Eventi=('ESN', 'count')
# ).reset_index()
# print(ww_stats)

A = u.df_filter_by_key(df, "ESN", 101)
B = u.df_filter_by_key(df, "ESN", 102)
C = u.df_filter_by_key(df, "ESN", 103)
D = u.df_filter_by_key(df, "ESN", 104)
data = {}
for a in [A,B,C,D]:
    esn = str(a['ESN'].iloc[0])
    printcent(f"Cicli tra un evento e l'altro - {esn}")
    ww, hpc, hpt = u.df_avg_stdd_cycles_to_event(a)

    print(f"EVENTO \tMEAN\tDEVIAZIONE STD: ")
    print(f"WW   :\t{round(ww[0])} \t {round(ww[1])}")
    print(f"HPC  :\t{round(hpc[0])}\t {round(hpc[1])}")
    print(f"HPT  :\t{round(hpt[0])}\t {round(hpt[1])}")

    data[esn] = {}
    data[esn]["WW_MEAN"] = ww[0]
    data[esn]["HPC_MEAN"] = hpc[0]
    data[esn]["HPT_MEAN"] = hpt[0]

    data[esn]["WW_STD"] = ww[1]
    data[esn]["HPC_STD"] = hpc[1]
    data[esn]["HPT_STD"] = hpt[1]

data = pd.DataFrame(data).T
figa = up.plot_avg_std_cycles_to_event(data,0)
figb = up.plot_avg_std_cycles_to_event(data,1)
figc = up.plot_avg_std_cycles_to_event(data,2)
figa.show()
figb.show()
figc.show()

# %%

# %%
