# Definizione di una funzione di plotting:
#   def <nome_del_plot>(<args>):
#       <logica di manipolazione dei dati>
#       plt.close() # o equivalente per pulire la RAM
#
# Definizione di una funzione helper:
#   la funzione deve essere la più generica possibile,
#   astraendo i dettagli specifici del dataset. Se la logica richiede specificità,
#   forse è meglio inserirla in una funzione di plotting.
#   Devono restituire al massimo una figura senza visualizzarla o salvarla.
#   le funzioni helper iniziano con "_".
#
#   def _<nome_funzione_helper>(<args>) -> <figura>:
#       <logica helper>
#       return <figura>
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from tools import utils as u, config as cfg
from pandas import DataFrame
from matplotlib.figure import Figure
from typing import overload
import os

class PlotColors:
    def __init__(self, colors = None) -> None:
        self.colors = colors

    def minmax(self, min, max, values, cmname = "viridis") -> None:
        norm = mcolors.Normalize(vmin=min, vmax=max)
        cmap = cm.get_cmap(cmname)
        colors = cmap(norm(values))
        self.colors = colors


def plot_avg_std_cycles_to_event(data: DataFrame, event:int, figsize: tuple[float, float] = (15,10)) -> Figure:
    """
    data è la lista di tuple ww, hpc e hpt
    primo valore di ogni tupla contiene media,
    il secondo la deviaizone standard

    event:
        0 - ww
        1 - hpc
        2 - hpt
    """
    event = (["WW", "HPC", "HPT"])[event]
    fig = plt.figure(figsize=figsize)
    plt.bar(
        ["101", "102", "103", "104"],
        data[f"{event}_MEAN"],
        yerr = data[f"{event}_STD"],
    )
    plt.xlabel("ESN")
    plt.ylabel("Cycles")
    return fig



def plot_stat_ess(data: DataFrame):
    """
    Plot di stazionarietà per ESS (ESN -> SENSOR -> SNAPSHOT)
    Ogni boxplot mostra la distribuzione del sensore tra voli,
    a parità di snapshot (fase di volo).
    """
    for esn in u.ESN:
        fig, axes = plt.subplots( nrows=4, ncols=4, figsize=(16, 12), sharey='row')
        fig.suptitle(f"Stazionarietà condizionata alla fase – ESN {esn}", fontsize=16)
        for i, sensor in enumerate(u.SENSORS):
            ax = axes[i//4, i%4]
            ddict = {}
            for j, snapshot in enumerate(u.SNAPSHOTS):
                ddict[j] = u.df_ess_filter(data, esn, sensor, snapshot).values.squeeze()

            ax.boxplot(ddict.values(), labels=ddict.keys(), patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='darkblue'),
                        medianprops=dict(color='red'),
                        whiskerprops=dict(color='green'))

            ax.set_title(f"Sensor {sensor}", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")

        # plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.tight_layout()
        yield fig


def plot_stat_feat(data: DataFrame, esn:int, sensor:str, snapshot:int, event_id:int, features_list:list,
                   step:int, event_type:int, save_path:str=None):
    """
    Plot dei grafici per le feature statistiche dei segnali
    """
    event_name = cfg.EVENT_TYPES.get(event_type, f"Tipo {event_type}")
    n_features = len(features_list)
    fig, axes = plt.subplots(
        n_features, 1, 
        figsize=(15, 3 * n_features), 
        sharex=True
    )
    fig.suptitle(
        f"Analisi Multi-Parametrica Run to failure | ESN: {esn} | Sensor: {sensor} | Snapshot: {snapshot} \n"
        f"Categoria: {event_name} | Evento n.: {event_id + 1}", 
        fontsize=16, y=1.02
    )
    for ax, feat in zip(axes, features_list):
        if feat not in data:
            continue 
        vals = data[feat]
        ax.plot(vals, label=feat.upper(), color='tab:blue', linewidth=1.5)
        # Estetica
        ax.set_ylabel(feat.upper(), fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        if len(vals) > 0:
            ax.axhline(np.mean(vals), color='red', linestyle=':', alpha=0.5)
    axes[-1].set_xlabel("Window Index (Cicli)", fontsize=12)
    plt.tight_layout()
    # --- GESTIONE SALVATAGGIO E CARTELLE ---
    if save_path:
        # Crea la cartella se non esiste
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Cartella creata: {save_path}")
        filename = f"ESN_{esn}_{event_name}_{sensor}_Evento_{event_id+1}_Snapshot_{snapshot}.png"
        file_path = os.path.join(save_path, filename)
        plt.savefig(file_path, bbox_inches='tight')
    plt.close()
