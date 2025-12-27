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
from tools import utils as u
from pandas import DataFrame
from matplotlib.figure import Figure
from typing import overload

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




