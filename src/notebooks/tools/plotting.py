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
import itertools
from operator import invert
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
import seaborn as sns

_colormap = "viridis"

cbase = list(mcolors.BASE_COLORS.keys())
ctab  = list(mcolors.TABLEAU_COLORS.keys())
ccss  = list(mcolors.CSS4_COLORS.keys())

def _get_color_cycler(color_list=cbase):
    return itertools.cycle(color_list)

def _get_cmap(colormap=None):
    if colormap:
        return plt.colormaps[colormap]
    return plt.colormaps[_colormap]

def _get_discrete_cmap(steps, colormap = None):
    if colormap:
        return plt.get_cmap(colormap, steps)
    return plt.get_cmap(_colormap, steps)

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

def plot_stat_feat_individually_mean(data: dict, pdata: PlotData, repair: RepairEventType, stop=True, featlist: list = None, save: bool = False, show=True):
    """
    Plot dei grafici per le feature statistiche dei segnali
    """

    if featlist is None:
        featlist = u.FEATURES

    fig, axes = _dynamic_grid(len(featlist), cols=pdata.cols, size=pdata.size)

    fig.suptitle(
        f"Run to failure | ESN: {pdata.esn} | Sensor: {pdata.sensor} | Snapshot: {pdata.snap} \n"
        f"Manutenzione: {repair}", 
        fontsize=16, y=1.02
    )

    trends = [
        #(5, 'blue', '-', 0.6, 2),               
        #(4, 'green', ':', 0.7, 2),               # 4th degree
        (1, 'red', '-', 0.8, 2)                  # Linear
    ]

    for ax, feat in zip(axes, featlist):
        cmap = _get_color_cycler()

        series_data = [] # Stores tuples of (label, values)
        
        max_len = 0
        min_len = float('inf')

        for g, e in data.items():
            y_data = e[feat]
            current_len = len(y_data)
            
            if current_len == 0: continue

            max_len = max(max_len, current_len)
            min_len = min(min_len, current_len)
            
            series_data.append((g, y_data))

        if min_len == float('inf'): min_len = 0

        # Prepare x values for trend fitting
        if min_len == 0:
            continue
        

        # Inside your loop:
        for label, y_data in series_data:
            ccol = next(cmap)
            
            # 1. Prepare data and X-axis
            y_plot = y_data[:min_len] if stop else y_data
            x_vals = np.arange(len(y_plot))
            
            # 2. Plot the raw data
            ax.plot(x_vals, y_plot, label=label, color=ccol, linewidth=1, alpha=0.1)

            if not stop:
                ax.axvline(len(y_data) - 1, color=ccol)

            # 3. Plot the Rolling Mean (Smoothed Data)
            # We still unpack 'degree' to keep the tuple format, but we don't use it.
            for degree, color, style, alpha, width in trends:
                
                # Calculate dynamic window size
                window = max(5, int(len(y_plot) * 0.1))
                
                y_smooth = pd.Series(y_plot).rolling(window=window, center=True, min_periods=1).std().to_numpy()
                ax.plot(x_vals, y_smooth, color=ccol, linestyle=style, alpha=alpha, linewidth=width)
                y_smooth = pd.Series(y_plot).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
                ax.plot(x_vals, y_smooth, color=ccol, linestyle=style, alpha=alpha-0.05, linewidth=width-0.5)
            
        # 4. Styling
        ax.set_title(feat.upper(), fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Window Index (Cicli)", fontsize=12)
    plt.tight_layout()

    if save:
        filename = f"{pdata.esn}-{pdata.sensor}-{pdata.snap}-{pdata.repair}.png"
        path = u.plot_path("STAT_FEATURES", pdata.repair, pdata.esn, pdata.sensor, filename=filename)
        plt.savefig(path, bbox_inches='tight')

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_stat_feat_individually(data: dict, pdata: PlotData, repair: RepairEventType, stop=True, featlist: list = None, save: bool = False, show=True):
    """
    Plot dei grafici per le feature statistiche dei segnali
    """

    if featlist is None:
        featlist = u.FEATURES

    fig, axes = _dynamic_grid(len(featlist), cols=pdata.cols, size=pdata.size)

    fig.suptitle(
        f"Run to failure | ESN: {pdata.esn} | Sensor: {pdata.sensor} | Snapshot: {pdata.snap} \n"
        f"Manutenzione: {repair}", 
        fontsize=16, y=1.02
    )

    trends = [
        #(5, 'blue', '-', 0.6, 2),               
        #(4, 'green', ':', 0.7, 2),               # 4th degree
        (1, 'red', '-', 0.8, 2)                  # Linear
    ]

    for ax, feat in zip(axes, featlist):
        cmap = _get_color_cycler()

        series_data = [] # Stores tuples of (label, values)
        
        max_len = 0
        min_len = float('inf')

        for g, e in reversed(data.items()):
            y_data = e[feat]
            current_len = len(y_data)
            
            if current_len == 0: continue

            max_len = max(max_len, current_len)
            min_len = min(min_len, current_len)
            
            series_data.append((g, y_data))

        if min_len == float('inf'): min_len = 0

        # Prepare x values for trend fitting
        if min_len == 0:
            continue
        

        for label, y_data in series_data:
            fit_x = np.arange(len(y_data))
            x_trend = np.arange(len(y_data))
            ccol = next(cmap)
            y_plot = y_data[:min_len] if stop else y_data
            
            ax.plot(y_plot, label=label, color=ccol, linewidth=1, alpha=0.4)

            if not stop:
                ax.axvline(len(y_data) - 1, color=ccol)

            for degree, color, style, alpha, width in trends:
                z = np.polyfit(fit_x, y_plot, degree)
                p = np.poly1d(z)
                ax.plot(x_trend, p(x_trend), color=ccol, linestyle=style, alpha=alpha, linewidth=width)

        # 4. Styling
        ax.set_title(feat.upper(), fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Window Index (Cicli)", fontsize=12)
    plt.tight_layout()

    if save:
        filename = f"{pdata.esn}-{pdata.sensor}-{pdata.snap}-{pdata.repair}.png"
        path = u.plot_path("STAT_FEATURES", pdata.repair, pdata.esn, pdata.sensor, filename=filename)
        plt.savefig(path, bbox_inches='tight')

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_stat_feat(data: dict, pdata: PlotData, repair: RepairEventType, stop=True, featlist: list = None, save: bool = False, show=True):
    """
    Plot dei grafici per le feature statistiche dei segnali
    """
    if featlist is None:
        featlist = u.FEATURES

    fig, axes = _dynamic_grid(len(featlist), cols=pdata.cols, size=pdata.size)

    fig.suptitle(
        f"Run to failure | ESN: {pdata.esn} | Sensor: {pdata.sensor} | Snapshot: {pdata.snap} \n"
        f"Manutenzione: {repair}", 
        fontsize=16, y=1.02
    )

    for ax, feat in zip(axes, featlist):
        cmap = _get_color_cycler()

        series_data = [] # Stores tuples of (label, values)
        all_vals = []    # Flattened y values for polyfit
        all_x = []       # Flattened x values for polyfit
        
        max_len = 0
        min_len = float('inf')

        for g, e in reversed(data.items()):
            y_data = e[feat]
            current_len = len(y_data)
            
            if current_len == 0: continue

            max_len = max(max_len, current_len)
            min_len = min(min_len, current_len)
            
            series_data.append((g, y_data))
            all_vals.extend(y_data)
            all_x.extend(range(current_len))

        if min_len == float('inf'): min_len = 0

        for label, y_data in series_data:
            ccol = next(cmap)
            y_plot = y_data[:min_len] if stop else y_data
            
            ax.plot(y_plot, label=label, color=ccol, linewidth=1, alpha=0.4)

            if not stop:
                ax.axvline(len(y_data) - 1, color=ccol)

        if len(all_vals) > 0:
            if stop:
                target_len = min_len
                fit_x = all_x[:min_len]
                fit_y = all_vals[:min_len]
            else:
                target_len = max_len
                fit_x = all_x
                fit_y = all_vals

            # Cap the max degree to 10 or (length-1), whichever is smaller.
            # This prevents the SVD/LinAlgError.
            safe_high_degree = min(target_len - 1, 10)
            
            # If safe_high_degree is too low (e.g. data has 2 points), ensure it doesn't break logic
            if safe_high_degree < 1: safe_high_degree = 1

            x_trend = np.arange(target_len)
            
            trends = [
                (safe_high_degree, 'blue', '-', 0.6, 2), # High degree (Capped)
                (4, 'green', ':', 0.7, 3),               # 4th degree
                (1, 'red', '-', 1.0, 4)                  # Linear
            ]

            for degree, color, style, alpha, width in trends:
                z = np.polyfit(fit_x, fit_y, degree)
                p = np.poly1d(z)
                ax.plot(x_trend, p(x_trend), color=color, linestyle=style, alpha=alpha, linewidth=width)

        # 4. Styling
        ax.set_title(feat.upper(), fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Window Index (Cicli)", fontsize=12)
    plt.tight_layout()

    if save:
        filename = f"{pdata.esn}-{pdata.sensor}-{pdata.snap}-{pdata.repair}.png"
        path = u.plot_path("STAT_FEATURES", pdata.repair, pdata.esn, pdata.sensor, filename=filename)
        plt.savefig(path, bbox_inches='tight')

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_features(dff: DataFrame, esn_list: list[int], tot: DataFrame, target: str, fulltarget: str, filter_feature: str | None = None, max_features_to_show: int = 6):
    if filter_feature:
        if type(filter_feature) is list:
            target_features = filter_feature
        else:
            target_features = [filter_feature]
    else:
        # Prende le migliori feature assolute (senza snap)
        target_features = tot['feature'].unique()[:max_features_to_show]

    figs = []
    for esn in esn_list:
        esn_data = dff[dff['esn'] == esn].sort_values('esn_index')
        if esn_data.empty: continue

        n_features = len(target_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(40, 6 * n_rows), squeeze=False)
        fig.suptitle(f"ANALISI CICLO DI VITA | ESN: {esn}", fontsize=16, fontweight='bold', y=1.02)
        
        axes_flat = axes.flatten()

        for i, feat_name in enumerate(target_features):
            if feat_name not in esn_data.columns: continue
            
            ax1 = axes_flat[i]
            
            # 1. Plot della Feature (Asse Sinistro)
            color_feat = 'tab:blue'
            l1, = ax1.plot(esn_data['esn_index'], esn_data[feat_name], 'o', 
                        color=color_feat, linewidth=1.5, label=f'Feature: {feat_name}')
            ax1.set_ylabel(feat_name, color=color_feat, fontweight='bold')
            ax1.set_xlabel('Cicli (Time)')

            # 2. Plot del Target/RUL (Asse Destro)
            ax2 = ax1.twinx()
            color_rul = 'tab:green'
            l2, = ax2.plot(esn_data['esn_index'], esn_data[fulltarget], 
                        color=color_rul, linestyle='--', alpha=0.7, label='Target (RUL)')
            ax2.set_ylabel('RUL', color=color_rul, fontweight='bold')

            # 3. Linee di Fault (HPC)
            fault_col = f'fault_{target.lower()}_cycle'
            if fault_col in esn_data.columns:
                for f_idx in esn_data.loc[esn_data[fault_col] == 1, 'esn_index']:
                    ax1.axvline(x=f_idx, color='red', linestyle=':', alpha=0.8, label='Fault')

            ax1.set_title(f"Trend: {feat_name}")
            ax1.grid(True, axis='both', alpha=0.3)

        # Pulizia subplot vuoti
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        figname = f"features_esn_{esn}.png"
        plt.show()
        figs.append((fig, figname))
    return figs

def plot_features_per_snap(dff: DataFrame, esn_list: list[int], tot: DataFrame, target: str, fulltarget: str, filter_feature: str | None = None, max_features_per_snap: int = 6):
    """
    Genera plot organizzati per ESN e per SNAP (fasi temporali).
    Ritorna una lista di tuple (figura, nome_file).
    """
    figs = []

    for esn in esn_list:
        esn_all_data = dff[dff['esn'] == esn]
        if esn_all_data.empty:
            continue

        # Iteriamo su ogni fase (snap) disponibile per questo motore
        for snap in sorted(esn_all_data['snap'].unique()):
            group_data = esn_all_data[esn_all_data['snap'] == snap].sort_values('esn_index')
            
            # 1. Selezione Feature per questo snap specifico
            if filter_feature:
                snap_best = tot[(tot['snap'] == snap) & (tot['feature'] == filter_feature)]
            else:
                # Prende le migliori N feature per questo snap basandosi sulla correlazione totale
                snap_best = tot[tot['snap'] == snap].sort_values('tot_val', key=abs, ascending=False).head(max_features_per_snap)

            target_features = snap_best['feature'].unique().tolist()
            n_features = len(target_features)
            
            if n_features == 0:
                continue

            # 2. Configurazione Griglia Subplot (Massimo 3 colonne)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            # Utilizziamo una larghezza generosa (es. 12 per asse) per visibilità ottimale
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, 6 * n_rows), squeeze=False)
            fig.suptitle(f"PHASE ANALYSIS | ESN: {esn} | SNAP: {snap}", fontsize=18, fontweight='bold', y=1.02)
            
            axes_flat = axes.flatten()

            # 3. Ciclo di Plotting sulle Feature identificate
            for i, feat_name in enumerate(target_features):
                if feat_name not in group_data.columns:
                    continue
                
                ax1 = axes_flat[i]
                
                # ASSE SINISTRO: Feature reale (Line + Points per vedere la granularità dello snap)
                color_feat = 'tab:blue'
                l1, = ax1.plot(group_data['esn_index'], group_data[feat_name], 
                               color=color_feat, linewidth=2, marker='o', markersize=4, 
                               alpha=0.7, label=f'Actual {feat_name}')
                
                ax1.set_title(f"Feature: {feat_name}", fontsize=14, pad=10)
                ax1.set_xlabel('Cycles')
                ax1.set_ylabel('Feature Value', color=color_feat, fontweight='bold')
                ax1.tick_params(axis='y', labelcolor=color_feat)

                # ASSE DESTRO: Target/RUL
                ax2 = ax1.twinx()
                color_rul = 'tab:green'
                l2, = ax2.plot(group_data['esn_index'], group_data[fulltarget], 
                               color=color_rul, linestyle='--', linewidth=2, label='RUL Trend')
                
                ax2.set_ylabel(f'To Next {target}', color=color_rul, fontweight='bold')
                ax2.tick_params(axis='y', labelcolor=color_rul)

                # LINEE DI FAULT (Verticali)
                fault_col = f'fault_{target.lower()}_cycle'
                if fault_col in group_data.columns:
                    faults = group_data.loc[group_data[fault_col] == 1, 'esn_index']
                    for f_idx in faults:
                        ax1.axvline(x=f_idx, color='red', linestyle=':', linewidth=2, alpha=0.8)

                # Estetica e Legenda
                ax1.grid(True, linestyle='--', alpha=0.5)
                lines = [l1, l2]
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='best', frameon=True, fontsize=9)

            # Nascondi subplot vuoti
            for j in range(i + 1, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.tight_layout()
            figname = f"features_esn{esn}_snap{snap}.png"
            plt.show()
            figs.append((fig, figname))
            
    return figs

def plot_pipeline_comparison(history, sensor_name):
    """Genera un confronto della pipeline con confronto Boxplot Prima/Dopo nello Step 2."""
    steps = list(history.keys())
    n_steps = len(steps)
    
    fig, axes = plt.subplots(n_steps, 1, figsize=(15, 4 * n_steps))
    fig.suptitle(f"DATA EVOLUTION PIPELINE | Sensor: {sensor_name}", 
                 fontsize=18, fontweight='bold', y=0.98)

    colors = ['#95a5a6', '#e74c3c', '#f1c40f', '#2ecc71', '#9143a3']
    boxplot_index = 2

    for i, step in enumerate(steps):
        ax = axes[i]
        data_current = history[step][sensor_name].dropna()
        
        # --- LOGICA BOXPLOT DI CONFRONTO (STEP 2) ---
        if i == boxplot_index:
            # Recuperiamo i dati dello step precedente per il confronto
            prev_step = steps[i-1]
            data_prev = history[prev_step][sensor_name].dropna()
            
            # Creiamo i due boxplot affiancati
            # Nota: passiamo una lista di array [precedente, attuale]
            bp = ax.boxplot([data_prev, data_current], vert=True, patch_artist=True, widths=0.5)
            
            # Coloriamo i due box in modo diverso per distinguerli
            bp['boxes'][0].set(facecolor=colors[i-1], alpha=0.5) # Colore dello step precedente
            bp['boxes'][1].set(facecolor=colors[i], alpha=0.8)   # Colore dello step attuale
            
            # Estetica mediana e outlier
            plt.setp(bp['medians'], color='yellow', linewidth=2)
            plt.setp(bp['fliers'], marker='o', markersize=3, alpha=0.2)
            
            ax.set_xticks([1, 2])
            ax.set_xticklabels([f"BEFORE ({prev_step})", f"AFTER ({step})"], fontweight='bold')
            ax.set_ylabel("Value Range")
            ax.set_title(f"STEP {i}: OUTLIERS REMOVAL", fontsize=14, fontweight='bold', loc='left')

        # --- LOGICA SCATTER PER GLI ALTRI STEP ---
        else:
            data = history[step][sensor_name]
            ax.scatter(data.index, data, s=2, label=step, color=colors[i], alpha=0.6)
            ax.plot(data.index, data, color=colors[i], alpha=0.2, linewidth=0.5)
            
            ax.set_title(f"STEP {i}: {step.upper()}", fontsize=14, fontweight='bold', loc='left')
            ax.set_xlabel("Cycles / Index")
            ax.set_ylabel("Sensor Value")

        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_residuals_dashboard(df, residual_cols):
    """
    Crea una dashboard dove ogni figura rappresenta un sensore di residuo.
    In ogni figura, c'è un subplot per ogni ESN con ogni snapshot.
    """
    engine_ids = df['esn'].unique()
    n_engines = len(engine_ids)
    
    cols = 3
    rows = (n_engines + cols - 1) // cols

    for res_col in residual_cols:
        fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), sharex=False)
        fig.suptitle(f'Analisi Residui: {res_col}', fontsize=20, y=1.02)
        
        axes = axes.flatten()
        
        for i, esn in enumerate(engine_ids):
            ax = axes[i]
            # Filtro per il motore specifico
            engine_data = df[df['esn'] == esn]
            
            # Plotto ogni snapshot con un colore diverso
            sns.lineplot(
                data=engine_data, 
                x='snap_index', 
                y=res_col, 
                hue='snap', 
                ax=ax, 
                palette='viridis',
                legend='full' if i == 0 else False
                )
            
            ax.set_title(f'ESN: {esn}')
            ax.set_xlabel('Cicli')
            ax.set_ylabel('Residuo')
            # Linea dello zero (riferimento "sano")
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()


def plot_engine_level_residuals(df, residual_cols, to_next_col, event):
    engine_ids = df['esn'].unique()
    n_engines = len(engine_ids)
    
    cols = 3
    rows = (n_engines + cols - 1) // cols

    for res_col in residual_cols:
        fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
        fig.suptitle(f'Engine-Level Residuals: {res_col} - event: {event}', fontsize=22, y=1.02)
        
        axes = axes.flatten()
        
        for i, esn in enumerate(engine_ids):
            ax = axes[i]
            # Filtro dati per il motore specifico
            engine_data = df[df['esn'] == esn]
            
            sns.lineplot(
                data=engine_data, 
                x='snap_index', 
                y=res_col, 
                ax=ax, 
                color='black',
                linewidth=1,
                label='Engine-Level'
            )

            # Cerco lo snap_index dove to_next_col è 0
            event_cycles = engine_data.loc[engine_data[to_next_col] == 0, 'snap_index']
            if not event_cycles.empty:
                # Itero sui valori trovati
                for idx, val_x in enumerate(event_cycles):
                    # Aggiungo la label solo alla prima linea per non sporcare la legenda
                    label_text = f'{event}' if idx == 0 else ""
                    
                    ax.axvline(
                        x=val_x, 
                        color='red', 
                        linestyle='-.', 
                        linewidth=1.5, 
                        alpha=0.7, 
                        label=label_text
                    )
            
            ax.set_title(f'ESN: {esn}', fontweight='bold')
            ax.set_xlabel('Cicli')
            ax.set_ylabel('Residuo')
            
            # Linea di riferimento zero (motore sano)
            ax.axhline(0, color='blue', linestyle='--', alpha=0.5)
            
            if i == 0:
                ax.legend()

        # Rimuovo i subplot vuoti
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()



def plot_engine_level_hi(df, residual_cols, to_next_col, event):
    engine_ids = df['esn'].unique()
    n_engines = len(engine_ids)
    cols = 3
    rows = (n_engines + cols - 1) // cols

    for res_col in residual_cols:
        fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
        fig.suptitle(f'{event.upper()} Health Index: {res_col} vs Cycles to Event ({event.upper()})', fontsize=22, y=1.02)
        
        axes = axes.flatten()
        
        for i, esn in enumerate(engine_ids):
            ax = axes[i]
            engine_data = df[df['esn'] == esn].sort_values('snap_index')
            
            # --- ASSE 1: Health Index (HI) ---
            sns.lineplot(
                data=engine_data, 
                x='snap_index', 
                y=res_col, 
                ax=ax, 
                color='blue',
                linewidth=2,
                label='Health Index (HI)'
            )
            ax.set_ylabel('Health Index', color='blue', fontweight='bold')
            ax.tick_params(axis='y', labelcolor='blue')

            # --- ASSE 2: To Next Event (Cicli Residui) ---
            ax2 = ax.twinx()
            sns.lineplot(
                data=engine_data,
                x='snap_index',
                y=to_next_col,
                ax=ax2,
                color='red',
                linestyle='-.',
                linewidth=1.5,
                alpha=0.7,
                label=f'Cycles to {event}'
            )
            ax2.set_ylabel('Cycles to Event', color='red', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='red')

            ax.set_title(f'ESN: {esn}', fontweight='bold')
            ax.set_xlabel('Cycles')

            if i == 0:
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper left')
            else:
                ax.get_legend().remove() if ax.get_legend() else None

        # Rimuovo i subplot vuoti
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()
        