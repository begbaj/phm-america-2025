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
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os

# %load_ext autoreload
# %autoreload 2
from tools import utils as u, config as cfg, algorithms as alg, plotting as up

# %%
# otraining = u.load_training()
otraining = u.load_forward_fill()
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

    nww = len(u.df_filter_by_key(wws, 'ESN', int(esn)))
    nhpc = len(u.df_filter_by_key(hpcs, 'ESN', int(esn)))
    nhpt = len(u.df_filter_by_key(hpts, 'ESN', int(esn)))

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
window = 5
overlap = 4
step = window - overlap

items = list(u.ess_iter(df))
n_plots = len(items)

fig, axes = plt.subplots(
    n_plots, 1,
    figsize=(15, 4 * n_plots),
    sharex=False
)

# caso con un solo subplot
if n_plots == 1:
    axes = [axes]

for ax, (d, e, sens, snap) in zip(axes, items):

    # eventi di riparazione per ESN
    esp = wws.loc[wws["ESN"] == e]

    rms_groups = alg.moving_rms_with_stop(
        signal=d[sens].values,
        stop=esp,
        N=window,
        o=step
    )

    # curve sovrapposte
    for gid, rms_vals in rms_groups.items():
        ax.plot(rms_vals, label=f"Group {gid}", alpha=0.8)

    ax.set_title(
        f"ESN: {e} | Sensor: {sens} | Snapshot: {snap}"
    )
    ax.set_ylabel("RMS amplitude")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("RMS window index")

fig.suptitle(
    f"Moving RMS dashboard | Window={window}, Overlap={overlap}",
    fontsize=14
)

fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
# %%
# Grafici features statistiche run to failure eventi hpc
# --- CONFIGURAZIONE ---
window, overlap = 5, 1
step = window - overlap
target = 0
# --- ESECUZIONE ---
for (d, e, sens, snap) in u.ess_iter(df):
    # 1. Calcolo feature
    esp = hpc.loc[hpc["ESN"] == e]
    featgroups = alg.moving_features_with_stop(
        signal=d[sens].values,
        stop=esp,
        N=window,
        o=step
    )
    output_dir = f"{cfg.STAT_FEATURES_PATH}/HPC/{e}/{sens}/"
    # 2. Validazione
    if target not in featgroups:
        print(f"Evento {target} non trovato per ESN {e}")
        continue
    # 3. Plotting tramite funzione dedicata
    up.plot_stat_feat(
        featgroups[target],
        e,
        sens,
        snap,
        target,
        features,
        1,
        output_dir
    )
# %%
window, overlap = 7, 4
step = window - overlap
old_esn = 0

for (d, pdata) in u.ess_iter(df, plotdata=True, order=["esn", "snapshot", "sensor"]):
    # 1. Calcolo feature
    esp = hpts.loc[hpts["ESN"] == pdata.esn]
    featgroups = alg.moving_features_with_stop(
        signal=d[pdata.sensor].values,
        stop=esp,
        N=window,
        o=step
    )

    pdata.size=(20,10)
    pdata.cols=3
    pdata.repair = u.RepairEventType.HPT

    if isinstance(pdata, up.PlotData):
        up.plot_stat_feat(featgroups, pdata, repair=pdata.repair, show=True, save=True)
