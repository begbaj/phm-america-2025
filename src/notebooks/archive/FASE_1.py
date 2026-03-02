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
#     display_name: phm-america-2025 (3.11.9)
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
from matplotlib.path import Path
from collections import deque

# %load_ext autoreload
# %autoreload 2
from tools import utils as u, config as cfg, algorithms as alg, plotting as up
from tools.types.plotdata import PlotData
from tools.types.enums import *

# %%
otraining = u.load_training()
#otraining = u.load_forward_fill()
otraining = u.load_smooth_training(otraining, 10)
owws, ohpc, ohpt = u.load_event_points(otraining())
# load_training testing e validation restituiscono un la funzione WrapData per mantenere il dato originale intatto senza modifiche.
# per accedere al dato e copiarlo basterà chiamarla come funzione che copierà il dataframe originale in una nuova variabile.
df = otraining()
wws = owws()
hpcs = ohpc()
hpts = ohpt()

# %%
# Creazione dei csv per la simulazione

# --- CONFIGURAZIONE INVILUPPO ---
envelope_vertices = [
    (0.0, 0), (0.0, 10000), (0.2, 10000), (0.5, 25000), 
    (0.6, 35000), (0.7, 40000), (0.8, 40000), (0.8, 25000), 
    (0.7, 2000), (0.5, 0), (0.0, 0)
]
flight_envelope = Path(envelope_vertices)

# --- CONFIGURAZIONE OUTPUT ---
header_line1 = ",,Basic Inputs,,,,,Engine Health Parameters,,,,,,,,,,,,,,Sensor/Actuator Biases Applied Before Model Call,,,,,,,,Sensor/Actuator Biases Applied After Model Call,,,,,,,"
header_line2 = "Notes,,Altitude,Mach,N1c,dTamb,,fan_WcMod,fan_PRMod,fan_EffMod,lpc_WcMod,lpc_PRMod,lpc_EffMod,hpc_WcMod,hpc_PRMod,hpc_EffMod,hpt_WcMod,hpt_EffMod,lpt_WcMod,lpt_EffMod,,Pamb_bias,Pt2_bias,Tt2_bias,N1mech_bias,VBV_bias,VAFN_bias,HP_EM_pwr_bias,,N3mech_bias,Wf_bias,Tt25_bias,Pt25_bias,Tt3_bias,Ps3_bias,Tt45_bias,Tt5_bias"

# Colonne da estrarre
column_mapping = {
    "Sensed_Altitude": "Altitude",
    "Sensed_Mach": "Mach",
    "Sensed_Fan_Speed": "N1c"
}

raw_cols = header_line2.split(',')
clean_cols = [name if name != '' else f"EMPTY_{i}" for i, name in enumerate(raw_cols)]

# --- GESTIONE MEDIE ---
window_size = 10
history_cols = ["Altitude", "Mach", "N1c"]
running_history = {col: deque(maxlen=window_size) for col in history_cols}

for esn in u.ESN:
    for snap in u.SNAPSHOTS:
        mask = (df['ESN'] == esn) & (df['Snapshot'] == snap)
        subset = df.loc[mask, list(column_mapping.keys())].copy()
        
        if not subset.empty:
            # 1. Creiamo il DataFrame di output con lo stesso numero di righe del subset
            num_rows = len(subset)
            final_df = pd.DataFrame(0.0, index=np.arange(num_rows), columns=clean_cols)
            
            # 2. Verifichiamo la validità punto per punto (o sul valore medio dello snapshot)
            # Qui decidiamo: se la MEDIA dello snapshot è valida, teniamo tutto lo snapshot
            avg_mach = subset['Sensed_Mach'].mean()
            avg_alt = subset['Sensed_Altitude'].mean()
            avg_fan = subset['Sensed_Fan_Speed'].mean()

            is_valid = (
                flight_envelope.contains_point((avg_mach, avg_alt)) and
                850 <= avg_fan <= 2550
            )

            if is_valid:
                # Se è VALIDO: Copiamo i dati riga per riga dal subset originale
                final_df['Altitude'] = subset['Sensed_Altitude'].values
                final_df['Mach'] = subset['Sensed_Mach'].values
                final_df['N1c'] = subset['Sensed_Fan_Speed'].values
                
                # Aggiorniamo la cronologia delle medie per i futuri punti invalidi
                # (usiamo la media di questo snapshot per la memoria)
                running_history['Altitude'].append(avg_alt)
                running_history['Mach'].append(avg_mach)
                running_history['N1c'].append(avg_fan)
            else:
                # Se NON è VALIDO: Riempiamo tutte le righe con la media storica
                for col in history_cols:
                    val_to_use = sum(running_history[col]) / len(running_history[col]) if running_history[col] else 0.0
                    final_df[col] = val_to_use
            
            # dTamb rimane a 0.0 per tutte le righe come richiesto
            final_df["dTamb"] = 0.0

            # --- SALVATAGGIO ---
            os.makedirs(os.path.dirname(cfg.DATA_SIMULATION_PATH), exist_ok=True)
            final_save_path = f"{cfg.DATA_SIMULATION_PATH}_ESN-{esn}_Snap-{snap}.csv"
            
            with open(final_save_path, 'w', encoding='utf-8') as f:
                f.write(header_line1 + "\n")
                f.write(header_line2 + "\n")
                # Scrittura dati (dTamb e gli altri saranno 0.0)
                final_df.to_csv(f, index=False, header=False, lineterminator='\n')
            
            print(f"ESN {esn} Snap {snap}: {'OK' if is_valid else 'SOSTITUITO (Media)'}")


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
# ## Profili run to failure

# %%

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
    print(f"RMS Value:    {alg.rms_signal(d):.4f}")
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
        step=step
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


for (d, pdata) in u.ess_iter(df, plotdata=True, order=["esn", "snapshot", "sensor"]):

    if not isinstance(pdata, PlotData) or not isinstance(d, pd.DataFrame):
        break 

    esp = hpts.loc[hpts["ESN"] == pdata.esn].index

    featgroups = alg.moving_features_with_stop(
        signal=u.to_signal(d, pdata.sensor),
        stop=esp,
        N=window,
        step=step
    )

    pdata.size=(20,10)
    pdata.cols=3
    pdata.repair = RepairEventType.HPT

    if isinstance(pdata, up.PlotData):
        up.plot_stat_feat(featgroups, pdata, repair=pdata.repair, stop=False, show=True, save=True)

# %%
