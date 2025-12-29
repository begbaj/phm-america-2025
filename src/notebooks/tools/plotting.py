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
from ast import FunctionType
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math
import numpy as np
import pandas as pd
from tools import utils as u, config as cfg
from pandas import DataFrame
from matplotlib.figure import Figure
from typing import overload
from tools.types.plotdata import PlotData
from tools.types.enums import ESENSORS, RepairEventType
import os


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

def _dynamic_grid(dlength, cols=3, size=(5, 4)):
    """
    Creates a dynamic grid of subplots based on the number of elements in data_list.
    
    Parameters:
    - n: numero di elementi
    - plot_func: A function(item, ax) that takes a single item and an axis to plot on.
    - cols: Number of columns in the grid.
    - size: Tuple (width, height) for *each individual subplot*.
    """
    rows = math.ceil(dlength / cols)
    figsize = (cols * size[0], rows * size[1])
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = axes.flatten() if dlength > 1 else [axes]
    return fig, axes_flat




def plot_stat_feat(data: dict, pdata: PlotData, repair: RepairEventType, featlist:list = None, save:bool=False, show=True):

    """
    Plot dei grafici per le feature statistiche dei segnali
    """

    if featlist is None:
        featlist = u.FEATURES

    fig, axes = _dynamic_grid(len(featlist),cols=pdata.cols, size=pdata.size)

    fig.suptitle(
        f"Run to failure | ESN: {pdata.esn} | Sensor: {pdata.sensor} | Snapshot: {pdata.snap} \n"
        f"Manutenzione: {repair}", 
        fontsize=16, y=1.02
    )

    for ax, feat in zip(axes, featlist):
        vals = []
        all_x = []  
        max_len = 0

        for g, e in data.items():
            y_data = e[feat]
            vals.extend(y_data)
            all_x.extend(range(len(y_data)))
            if len(y_data) > max_len:
                max_len = len(y_data)
            ax.plot(y_data, label=g, linewidth=1, alpha=0.4)
        ax.axvline(all_x[-1])

        if len(vals) > 0: # polyfit
            z = np.polyfit(all_x, vals, max_len)
            p = np.poly1d(z)
            x_trend = np.arange(max_len)
            ax.plot(x_trend, p(x_trend), color='blue', alpha=0.6, linewidth=2)

        if len(vals) > 0: # polyfit
            z = np.polyfit(all_x, vals, 4)
            p = np.poly1d(z)
            x_trend = np.arange(max_len)
            ax.plot(x_trend, p(x_trend), color='green',linestyle=":" ,alpha=0.7, linewidth=3)

        if len(vals) > 0: # polyfit
            z = np.polyfit(all_x, vals, 1)
            p = np.poly1d(z)
            x_trend = np.arange(max_len)
            ax.plot(x_trend, p(x_trend), color='red',alpha=1, linewidth=4)

        ax.set_title(feat.upper(), fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Window Index (Cicli)", fontsize=12)
    plt.tight_layout()

    if save:
        filename = "-".join([str(pdata.esn), str(pdata.sensor), str(pdata.snap), str(pdata.repair)]) + ".png"
        path = u.plot_path("STAT_FEATURES", pdata.repair, pdata.esn, pdata.sensor, filename=filename)
        plt.savefig(path, bbox_inches='tight')

    if show:
        plt.show()

    plt.close(fig)
    return fig