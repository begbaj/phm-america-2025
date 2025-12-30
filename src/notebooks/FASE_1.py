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
# # Importazione ed Analisi Dati
# Attenzione: TUTTI gli import DEVONO stare nel primo blocco.
# Questo notebook NON deve contenere logiche di plotting o di manipolazione dati se non operazioni essenziali. Queste logiche vanno definite in tools.plotting e tools.algorithms.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os

# %load_ext autoreload
# %autoreload 2
from tools import utils as u, config as cfg, algorithms as alg, plotting as up
from tools.types.plotdata import PlotData
from tools.types.enums import *

# %%
otraining = u.load_training()
#otraining = u.load_forward_fill()
# otraining = u.load_smooth_training(otraining, 10)
owws, ohpc, ohpt = u.load_event_points(otraining())
# load_training testing e validation restituiscono un la funzione WrapData per mantenere il dato originale intatto senza modifiche.
# per accedere al dato e copiarlo basterà chiamarla come funzione che copierà il dataframe originale in una nuova variabile.
df = otraining()
wws = owws()
hpcs = ohpc()
hpts = ohpt()


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

    nww = u.df_filter_by_key(wws, 'ESN', int(esn)).shape[0]
    nhpc = u.df_filter_by_key(hpcs, 'ESN', int(esn)).shape[0]
    nhpt = u.df_filter_by_key(hpts, 'ESN', int(esn)).shape[0]

    print(f"EVENTO \tMEAN\tDEVIAZIONE STD \t EVENTI: ")
    print(f"WW   :\t{round(ww[0])} \t {round(ww[1])} \t\t {nww}")
    print(f"HPC  :\t{round(hpc[0])}\t {round(hpc[1])} \t\t {nhpc}")
    print(f"HPT  :\t{round(hpt[0])}\t {round(hpt[1])} \t\t {nhpt}")

    data[esn] = {}
    data[esn]["WW_MEAN"]  = ww[0]
    data[esn]["HPC_MEAN"] = hpc[0]
    data[esn]["HPT_MEAN"] = hpt[0]

    data[esn]["WW_STD"]  = ww[1]
    data[esn]["HPC_STD"] = hpc[1]
    data[esn]["HPT_STD"] = hpt[1]

    data[esn]["EVENTS_WW"] = nww
    data[esn]["EVENTS_HPC"] = nhpc
    data[esn]["EVENTS_HPT"] = nhpt


data = pd.DataFrame(data).T
# un grafico solo potrebbe essere ottimale
figa = up.plot_avg_std_cycles_to_event(data,0)
figb = up.plot_avg_std_cycles_to_event(data,1)
figc = up.plot_avg_std_cycles_to_event(data,2)
figa.show()
figb.show()
figc.show()

# %% [markdown]
# # Stazionarietà
# Parlare di stazionarietà dei segnali, nel caso dei dati a disposizione, non ha granchè senso se i dati vengono valutati tutti insieme. Bisogna perciò fare una analisi separata per ogni gruppo di dati che contengono dati dello stesso snapshot in voli diversi nello stesso motore.
#
# Si parla di suddivisione ESN->Sensor->Snapshot: u.ess_filter(df, esn, sensor, snapshot)

# %%
for fig in up.plot_stat_ess(df):
    fig.show()

# %% [markdown]
# # RUL

# %%
for d, e, sens, snap in u.ess_iter(df):
    print(f"Dataset Analysis:")
    print(f"-----------------")
    print(f"{e} - {sens} - snap {snap}")
    print(f"RMS Value:    {alg.rms(d):.4f}")
    print(f"Shape Factor: {alg.shape_factor(d):.4f}")
    print(f"Skewness:     {d[sens].skew():.4f}")
    print(f"Kurtosis:     {d[sens].kurtosis():.4f}")


# %%
for esn_id in u.ESN:
    edata = u.df_filter_by_key(df, "ESN", esn_id)
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    print(f"Generazione dashboard per Motore ESN: {esn_id}...")
    for i, sensor in enumerate(ESENSORS):
        ax = axes[i]
        series = edata[str(sensor)]

        if len(series) > 16:
            roll_mean = series.rolling(window=16).mean()
            roll_std = series.rolling(window=16).std()

            ax.plot(series.values, alpha=0.3, label='Raw', color='gray')
            ax.plot(roll_mean.values, label='Media Mobile', color='blue', linewidth=1.5)
            ax.plot(roll_std.values, label='Std Mobile', color='red', linewidth=1)
        else:
            ax.text(0.5, 0.5, 'Dati insufficienti', ha='center', va='center')
        # Formattazione singolo grafico
        ax.set_title(f"{sensor_name}", fontsize=10)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=8)
        if i == 0:
            ax.legend(loc='upper left', fontsize='x-small')

    fig.suptitle(f"Analisi Stazionarietà - Motore ESN {esn_id}", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(f"Analisi_Stazionarieta_ESN_{esn_id}.png", bbox_inches='tight')
    plt.show()

# %%
window, overlap = 7, 4
step = window - overlap


for (d, pdata) in u.ess_iter(df, plotdata=True, order=["snapshot", "sensor", "esn"], rand=True):

    if not isinstance(pdata, PlotData) or not isinstance(d, pd.DataFrame):
        break 

    # esp = hpcs.loc[hpcs["ESN"] == pdata.esn].index
    esp = wws.loc[wws["ESN"] == pdata.esn].index

    featgroups = alg.moving_features_with_stop(
        signal=u.to_signal(d, pdata.sensor),
        stop=esp,
        N=window,
        step=step
    )

    print(alg.evaluate_feature_groups_stats(featgroups[0]))

    pdata.size=(20,10)
    pdata.cols=3
    pdata.repair = str(RepairEventType.WW)

    # if isinstance(pdata, up.PlotData):
    #     up.plot_stat_feat_individually(featgroups, pdata, repair=pdata.repair, stop=False, show=True, save=True)
