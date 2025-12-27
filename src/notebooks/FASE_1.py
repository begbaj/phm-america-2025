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
target_event = 0  # 0 per il primo pezzo, 1 per il secondo, ecc.
features = ["rms", "mean", "std", "kurtosis", "skewness", "shape_factor"]

window = 5
overlap = 1
step = window - overlap

items = list(u.ess_iter(df))
n_plots = len(items)
for (d, e, sens, snap) in items:
    # 1. Calcolo le feature per questo sensore/ESN
    # eventi di riparazione per ESN
    esp = wws.loc[wws["ESN"] == e]
    all_features_groups = alg.moving_features_with_stop(
        signal=d[sens].values,
        stop=esp,
        N=window,
        o=step
    )

    # curve sovrapposte
    # for gid, rms_vals in rms_groups.items():
    #     ax.plot(rms_vals, label=f"Group {gid}", alpha=0.8)

    # --- LOGICA DI SELEZIONE ---
    # Verifichiamo se l'evento richiesto esiste nel dizionario
    # Verifichiamo se il segmento esiste
    if target_event not in all_features_groups:
        print(f"Evento {target_event} non trovato per ESN {e}")
        continue
        
    # 2. Creo una figura specifica per questo sensore con 6 subplot (3x2)
    fig, axes = plt.subplots(
        len(features), 1, 
        figsize=(15, 3 * len(features)), 
        sharex=True  # Condividono l'asse X per allineare i tempi
    )
    
    fig.suptitle(
        f"Analisi Multi-Parametrica | ESN: {e} | Sensor: {sens} | Snapshot: {snap} \nEvento n.: {target_event+1}", 
        fontsize=16, y=1.02
    )

    # 4. Ciclo sui subplot per ogni feature
    for ax, feat in zip(axes, features):
        vals = all_features_groups[target_event][feat]
        
        ax.plot(vals, label=feat.upper(), color='tab:blue', linewidth=1.5)
        
        # Estetica del singolo subplot
        ax.set_ylabel(feat.upper(), fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        
        # Opzionale: aggiunge una linea tratteggiata per la media globale del segmento
        if len(vals) > 0:
            ax.axhline(np.mean(vals), color='red', linestyle=':', alpha=0.5)

    axes[-1].set_xlabel("Window Index (Tempo)", fontsize=12)
    
    plt.tight_layout()
    plt.show()
# %%
