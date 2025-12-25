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




