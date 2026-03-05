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

# %%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import iplot
from statsmodels.tsa.stattools import adfuller
from utils import generals
import os
from plotly.subplots import make_subplots
cacca

# %%
# Dati di train originali
train = None
with open("../../Data/PHM2025_training_data/training_data.csv", "r") as f:
    train = pd.read_csv(f)

# Dati di train con forward fill per i NaN
train_ffill = None
with open("../../Data/PHM2025_training_data_ffill.csv", "r") as f:
    train_ffill = pd.read_csv(f)

# %%
# Conteggio numero di righe totali e per motore
rows = len(train)
print(f"Numero totale di righe: {rows}")
print(f"\nNumero di righe per ogni motore:")
train.groupby('ESN').size()

# Conteggio dei valori nulli per colonna
train.isnull().sum()
# Colonne con valori nulli:
# - Sensed_WFuel : peso misurato del carburante
# - Sensed_Core_Speed : velocità di rotazione misurata dello 
#   spool ad alta pressione, che comprende l'albero con il HPC e la HPT
# - Sensed_T3 : temperatura dell'aria misurata in uscita al HPC
# - Sensed_Ps3 : pressione misurata in uscita al HPC
# - Sensed_T45 : temperatura del gas combusto misurata all'uscita della HPT
# - Sensed_T5 : temperatura misurata all'uscita della turbina a bassa pressione

# Conteggio dei valori nulli per ogni motore
train.groupby('ESN').apply(lambda x: x.isnull().sum(), include_groups=False)
# Da valutare 
# Sono presenti maggiormente nel motore 104

# %%
# Add a cycle index 'scaled_index' for each ESN
train['scaled_index'] = train.groupby('ESN').cumcount()

# %%
# Farward Fill per riempimento dei NaN

df_ffill = train.copy()
df_ffill = df_ffill.sort_values(['ESN', 'Snapshot', 'Cycles_Since_New'])

for sensor in generals.SENSORS:
    sensor_name = sensor.value if hasattr(sensor, 'value') else sensor
    print(f"Forward fill sul sensore: {sensor_name}")
    df_ffill[sensor_name] = df_ffill.groupby(['ESN', 'Snapshot'])[sensor_name].ffill()

# 1. Verifica finale: controlliamo se ci sono ancora NaN
for sensor in generals.SENSORS:
    sensor_name = sensor.value if hasattr(sensor, 'value') else sensor
    nan_residui = df_ffill[sensor_name].isna().sum()
    print(f"Processo completato. NaN residui nel sensore {sensor_name}: {nan_residui}")
# 2. Definisci il nome del file
output_dir = "../../Data"
final_save_path = f"{output_dir}/PHM2025_training_data_ffill.csv"
# 3. Salvataggio in CSV
# index=False evita di aggiungere una colonna inutile per gli indici di riga
df_ffill.to_csv(final_save_path, index=False)
print(f"File salvato con successo come: {final_save_path}")


# %%
# Grafici di confronto per serie originali e ffill

def plot_ffill_comparison(df_orig, df_ffill, esn, snapshot):
    # Filtriamo i dati per il motore e lo snapshot specifico
    orig_sub = df_orig[(df_orig['ESN'] == esn) & (df_orig['Snapshot'] == snapshot)].sort_values('Cycles_Since_New')
    ffill_sub = df_ffill[(df_ffill['ESN'] == esn) & (df_ffill['Snapshot'] == snapshot)].sort_values('Cycles_Since_New')
    if orig_sub.empty:
        print(f"Nessun dato per ESN {esn}, Snapshot {snapshot}")
        return
    sensor_names = [sensor.value if hasattr(sensor, 'value') else sensor for sensor in generals.SENSORS]
    # Creiamo una griglia 4x4 (per 16 sensori)
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    fig.suptitle(f'Dashboard Serie con Farward Fill - Motore: {esn} | Snapshot: {snapshot}', fontsize=20)    
    axes = axes.flatten()
    for i, sensor_name in enumerate(sensor_names):
        ax = axes[i]
        plot_data_orig = orig_sub[sensor_name].reset_index(drop=True)
        plot_data_ffill = ffill_sub[sensor_name].reset_index(drop=True)
        # Plot originale
        ax.plot(plot_data_orig.index, orig_sub[sensor_name], color='red', linestyle='--', alpha=0.3, linewidth=1, label='Original')
        # Plot Ffill
        ax.plot(plot_data_ffill.index, ffill_sub[sensor_name], color='blue', linestyle='-', alpha=0.9, linewidth=1.5, label='Farward Fill')
        ax.set_title(f'{sensor_name}', fontsize=10)
        ax.grid(True, alpha=0.2)
        if i == 0: ax.legend() 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Salvataggio dashboard
    output_dir = f"img/FORWARD-FILL/ESN_{esn}"
    os.makedirs(output_dir, exist_ok=True)
    final_save_path = f"{output_dir}/Dashboard_Snapshot_{snapshot}.png"
    plt.savefig(final_save_path)
    plt.close()
    print(f"Dashboard salvata: {final_save_path}")

for esn in generals.ESN:
    for snap in generals.Snapshot:
        plot_ffill_comparison(train, train_ffill, esn, snap)

# %%
# Identificazione degli event points
# Assuming your DataFrame is 'df' and the column is 'sensor_value'
wws_points = train[train['Cumulative_WWs'] > train['Cumulative_WWs'].shift(1)]
hpc_points = train[train['Cumulative_HPC_SVs'] > train['Cumulative_HPC_SVs'].shift(1)]
hpt_points = train[train['Cumulative_HPT_SVs'] > train['Cumulative_HPT_SVs'].shift(1)]

# %%
# Calcolo della media e della deviazione standard tra i cicli tra ogni ww per motore
wws_points_2 = wws_points.copy() # lavoro su una copia
wws_counts = wws_points_2.groupby('ESN').size().reset_index(name='Totale_Eventi_WW')
wws_points_2['cycles_between_ww'] = wws_points_2.groupby('ESN')['Cycles_Since_New'].diff()
ww_stats = wws_points_2.groupby('ESN').agg(
    Media_Intervallo_WW=('cycles_between_ww', 'mean'),
    DevStd_Intervallo_WW=('cycles_between_ww', 'std'),
    Num_Eventi=('ESN', 'count')
).reset_index()
print(ww_stats)
# Grafico per chiarezza
ww_stats['ESN'] = ww_stats['ESN'].astype(str)
fig_ww = px.bar(
    ww_stats, 
    x='ESN', 
    y='Media_Intervallo_WW',
    error_y='DevStd_Intervallo_WW',
    text='Num_Eventi',
    title='Media Cicli tra Water Wash (WW) per Motore',
    labels={
        'ESN': 'Motore (ESN)', 
        'Media_Intervallo_WW': 'Media Cicli tra WW',
        'Num_Eventi': 'Numero di eventi'
    },
    color='Media_Intervallo_WW',
    color_continuous_scale='Viridis'
)
fig_ww.update_traces(
    textposition='inside',    
    insidetextanchor='start'
)
fig_ww.update_layout(
    xaxis_tickangle=-45,
    yaxis_title="Cicli Medi (con Dev. Std.)",
    hovermode="x unified",
    height=500
)

# Calcolo della media e della deviazione standard tra i cicli tra ogni HPC_SV per motore
hpc_points_2 = hpc_points.copy() # lavoro su una copia
hpc_counts = hpc_points_2.groupby('ESN').size().reset_index(name='Totale_Eventi_HPC_SV')
hpc_points_2['cycles_between_hpc_sv'] = hpc_points_2.groupby('ESN')['Cycles_Since_New'].diff()
hpc_stats = hpc_points_2.groupby('ESN').agg(
    Media_Intervallo_HPC_SV=('cycles_between_hpc_sv', 'mean'),
    DevStd_Intervallo_HPC_SV=('cycles_between_hpc_sv', 'std'),
    Num_Eventi=('ESN', 'count')
).reset_index()
print(hpc_stats)
# Grafico per chiarezza
hpc_stats['ESN'] = hpc_stats['ESN'].astype(str)
fig_hpc = px.bar(
    hpc_stats, 
    x='ESN', 
    y='Media_Intervallo_HPC_SV',
    error_y='DevStd_Intervallo_HPC_SV',
    text='Num_Eventi',
    title='Media Cicli tra HPC shop visit per Motore',
    labels={
        'ESN': 'Motore (ESN)', 
        'Media_Intervallo_HCP_SV': 'Media Cicli tra HPC SV',
        'Num_Eventi': 'Numero di eventi'
    },
    color='Media_Intervallo_HPC_SV',
    color_continuous_scale='Viridis'
)
fig_hpc.update_traces(
    textposition='inside',    
    insidetextanchor='start'
)
fig_hpc.update_layout(
    xaxis_tickangle=-45,
    yaxis_title="Cicli Medi (con Dev. Std.)",
    hovermode="x unified",
    height=500
)

# Calcolo della media e della deviazione standard tra i cicli tra ogni HPT_SV per motore
hpt_points_2 = hpt_points.copy() # lavoro su una copia
hpt_counts = hpt_points.groupby('ESN').size().reset_index(name='Totale_Eventi_HPT_SV')
hpt_points_2['cycles_between_hpt_sv'] = hpt_points_2.groupby('ESN')['Cycles_Since_New'].diff()
hpt_stats = hpt_points_2.groupby('ESN').agg(
    Media_Intervallo_HPT_SV=('cycles_between_hpt_sv', 'mean'),
    DevStd_Intervallo_HPT_SV=('cycles_between_hpt_sv', 'std'),
    Num_Eventi=('ESN', 'count')
).reset_index()
print(hpt_stats)
# Grafico per chiarezza
hpt_stats['ESN'] = hpt_stats['ESN'].astype(str)
fig_hpt = px.bar(
    hpt_stats, 
    x='ESN', 
    y='Media_Intervallo_HPT_SV',
    error_y='DevStd_Intervallo_HPT_SV',
    text='Num_Eventi',
    title='Media Cicli tra HPT shop visit per Motore',
    labels={
        'ESN': 'Motore (ESN)', 
        'Media_Intervallo_HCT_SV': 'Media Cicli tra HPT SV',
        'Num_Eventi': 'Numero di eventi'
    },
    color='Media_Intervallo_HPT_SV',
    color_continuous_scale='Viridis'
)
fig_hpt.update_traces(
    textposition='inside',    
    insidetextanchor='start'
)
fig_hpt.update_layout(
    xaxis_tickangle=-45,
    yaxis_title="Cicli Medi (con Dev. Std.)",
    hovermode="x unified",
    height=500
)

# Stampa dei grafici
fig_ww.show()
fig_hpc.show()
fig_hpt.show()

# %%
# Test per verificare la stazionarietà dei segnali dei sensori per Snapshot - controllo delle statistiche mobili dei segnali

output_dir = globals.PLOT_PATH + "/STAIONARITY-BY-SNAPSHOT"
os.makedirs(output_dir, exist_ok=True)

for esn_id in globals.ESN:
    sensor_data_tot = train[train['ESN'] == esn_id].copy()
    for snapshot in globals.Snapshot:
        sensor_data = sensor_data_tot[sensor_data_tot['Snapshot'] == snapshot].copy() 
        # Se uno snapshot ha troppi pochi dati, lo salta
        if len(sensor_data) < 5:
            continue
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        print(f"Generazione dashboard per Motore ESN: {esn_id} - Snapshot: {snapshot}...")
        for i, sensor in enumerate(globals.SENSORS):
            sensor_name = sensor.value if hasattr(sensor, 'value') else sensor
            ax = axes[i]
            series = sensor_data[sensor_name].dropna()
            if len(series) > 100:
                roll_mean = series.rolling(window=100).mean()
                roll_std = series.rolling(window=100).std()
                ax.plot(series.values, alpha=0.4, label='Raw', color='gray', linestyle=':')
                ax.plot(roll_mean.values, label='Media Mobile', color='blue', linewidth=2)
                ax.plot(roll_std.values, label='Std Mobile', color='red', linewidth=1)
            else:
                ax.text(0.5, 0.5, 'Dati insufficienti', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{sensor_name}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc='upper left', fontsize='x-small')
        fig.suptitle(f"Analisi Stazionarietà - ESN {esn_id} | Snapshot: {snapshot}", fontsize=20, y=1.02)
        plt.tight_layout()
        # Salvataggio file con nome specifico per ESN e Snapshot
        filename = f"{esn_id}_Snap_{snapshot}.png".replace("/", "_")
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

# %%
# Test ADF per verificare la stazionarietà dei segnali dei sensori

results_list = []

for esn_id in globals.ESN:
    sensor_data_tot = train[train['ESN'] == esn_id]    
    for snapshot in globals.Snapshot:
        snapshot_data = sensor_data_tot[sensor_data_tot['Snapshot'] == snapshot]        
        for sensor in globals.SENSORS:
            sensor_name = sensor.value if hasattr(sensor, 'value') else sensor
            series = snapshot_data[sensor_name].dropna()
            # Controllo dei requisiti minimi per ADF
            if series.nunique() <= 1 or len(series) < 10:
                results_list.append({
                    'ESN': esn_id, 
                    'Snapshot': snapshot,
                    'Sensor': sensor_name, 
                    'p-value': np.nan, 
                    'Stazionario': 'Dati Costanti/Insufficienti'
                })
                continue
            try:
                # Esecuzione test ADF
                res = adfuller(series)
                p_value = res[1]
                is_stationary = True if p_value <= 0.05 else False
                results_list.append({
                    'ESN': esn_id, 
                    'Snapshot': snapshot,
                    'Sensor': sensor_name, 
                    'p-value': round(p_value, 4), 
                    'Stazionario': is_stationary
                })
            except Exception as e:
                results_list.append({
                    'ESN': esn_id, 
                    'Snapshot': snapshot,
                    'Sensor': sensor_name, 
                    'p-value': np.nan, 
                    'Stazionario': 'Errore'
                })
df_results = pd.DataFrame(results_list)

# Visualizzazione dei sensori NON stazionari per ogni snapshot
print("Prime 20 righe di sensori NON stazionari (divisi per Snapshot):")
print(df_results[df_results['Stazionario'] == False].head(20))

# %%
# Matrice di correlazione delle colonne divise per ESN e per ogni snapshot
# Utile per PCA futura
output_dir = globals.PLOT_PATH + "/CORRELATION-MATRIX"
os.makedirs(output_dir, exist_ok=True)
for esn_id in globals.ESN:
    for snapshot in globals.Snapshot:
        cm = train[(train["ESN"] == esn_id) & (train["Snapshot"] == snapshot)].corr()
        plt.figure(figsize=(15,15))
        sns.heatmap(cm, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'Correlation Matrix Heatmap - ESN {esn_id} - Snapshot {snapshot}')
        plt.savefig(f"{output_dir}/ESN_{esn_id}_Snapshot_{snapshot}.png".replace("/", "_"))
        plt.show()

# %%
# Define output directory
output_dir = globals.PLOT_PATH + "/CORRELATION-MATRIX-BY-ESN-SNAPSHOT"
os.makedirs(output_dir, exist_ok=True)

for esn_id in globals.ESN:
    for snapshot in globals.Snapshot:
        # Filter data (using & for cleaner boolean indexing)
        subset = train[(train["ESN"] == esn_id) & (train["Snapshot"] == snapshot)]
        
        # Basic check to ensure we have data before plotting
        if subset.empty:
            continue

        cm = subset.corr()

        # Create Plotly Heatmap
        # text_auto=".2f" replaces annot=True and fmt=".2f"
        # 'RdBu_r' is the standard Plotly equivalent to 'coolwarm'
        fig = px.imshow(
            cm,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title=f'Correlation Matrix Heatmap - ESN {esn_id} - Snapshot {snapshot}'
        )

        # Update layout to mimic the large figure size (15x15 inches approx 1200-1500px)
        fig.update_layout(
            width=1200, 
            height=1200,
            title_x=0.5 # Center title
        )

        # Sanitize filename components specifically, rather than the whole path
        safe_esn = str(esn_id).replace("/", "_")
        safe_snap = str(snapshot).replace("/", "_")
        filename_base = f"ESN_{safe_esn}_Snapshot_{safe_snap}"

        # Save as PNG
        fig.write_image(f"{output_dir}/{filename_base}.png")

        # Save as HTML (interactive)
        fig.write_html(f"{output_dir}/{filename_base}.html")


# %%
# Define output directory
output_dir = globals.PLOT_PATH + "/CORRELATION-MATRIX-COMBINED"
os.makedirs(output_dir, exist_ok=True)

# Convert iterables to lists to determine grid dimensions
esn_list = list(globals.ESN)
snap_list = list(globals.Snapshot)
n_rows = len(esn_list)
n_cols = len(snap_list)

# Create a subplot grid
# Rows correspond to ESNs, Columns correspond to Snapshots
fig = make_subplots(
    rows=n_rows, 
    cols=n_cols,
    subplot_titles=[f"ESN {esn} - Snap {snap}" for esn in esn_list for snap in snap_list],
    vertical_spacing=0.04,  # Adjust spacing between rows
    horizontal_spacing=0.04 # Adjust spacing between columns
)
from importlib import reload
reload(globals)
for row_idx, esn_id in enumerate(esn_list):
    for col_idx, snapshot in enumerate(snap_list):
        # Filter data
        subset = train[(train["ESN"] == esn_id) & (train["Snapshot"] == snapshot)]
        subset = globals.sensors_subset(subset)  # Keep only sensor columns
        
        if subset.empty:
            continue

        cm = subset.corr()
        
        # Only show the color scale (legend) on the last plot to reduce clutter
        show_scale = (row_idx == n_rows - 1 and col_idx == n_cols - 1)

        # Add Heatmap Trace
        fig.add_trace(
            go.Heatmap(
                z=cm.values,
                x=cm.columns,
                y=cm.index,
                colorscale='RdBu_r', # Equivalent to 'coolwarm'
                zmin=-1, 
                zmax=1,
                text=cm.values,
                texttemplate="%{z:.2f}", # Mimics annot=True
                textfont={"size": 10},
                showscale=show_scale
            ),
            row=row_idx + 1,
            col=col_idx + 1
        )
        # Invert Y-axis to match standard matrix orientation (top-down)
        fig.update_yaxes(autorange="reversed", row=row_idx + 1, col=col_idx + 1)

# Update layout size dynamically based on the number of plots
# Assuming approx 500px per subplot for readability
fig.update_layout(
    height=1000 * n_rows,
    width=1000 * n_cols,
    title_text="Combined Correlation Matrix Heatmaps"
)

# Save combined files
filename = "Combined_Correlation_Matrices"
fig.write_html(f"{output_dir}/{filename}.html")
fig.write_image(f"{output_dir}/{filename}.png")


# %%
# Define output directory
output_dir = globals.PLOT_PATH + "/CORRELATION-MATRIX"
os.makedirs(output_dir, exist_ok=True)

# Convert iterables to lists
esn_list = list(globals.ESN)
snap_list = list(globals.Snapshot)
n_rows = len(esn_list)
n_cols = len(snap_list)

# Create a grid of subplots
# Adjust figsize: 15 inches per column, 15 inches per row to ensure readability of numbers
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15 * n_cols, 15 * n_rows))

# Ensure axes is always a 2D array for consistent indexing, 
# even if there is only 1 row or 1 column.
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

for i, esn_id in enumerate(esn_list):
    for j, snapshot in enumerate(snap_list):
        ax = axes[i, j]
        
        # Filter data
        subset = train[(train["ESN"] == esn_id) & (train["Snapshot"] == snapshot)]
        
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        cm = subset.corr()

        # Create Heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            ax=ax, 
            cbar=True,
            square=True
        )
        
        ax.set_title(f'ESN {esn_id} - Snapshot {snapshot}')

# Adjust layout to prevent overlap
plt.tight_layout()

# Define filename
filename_base = "Combined_Correlation_Matrices_Matplotlib"
png_path = f"{output_dir}/{filename_base}.png"
html_path = f"{output_dir}/{filename_base}.html"

# 1. Save PNG
plt.savefig(png_path)
plt.close()

# 2. Save HTML (Wrapper for the PNG)
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Combined Correlation Matrices</title>
</head>
<body>
    <h1>Combined Correlation Matrix View</h1>
    <img src="{filename_base}.png" alt="Correlation Matrices" style="max-width: 100%; height: auto;">
</body>
</html>
"""

with open(html_path, "w") as f:
    f.write(html_content)

print(f"Saved combined view to:\n{png_path}\n{html_path}")


# %%
# Grafico andamento
# Tutti i Sensori in una singola immagine per ESN

check_init()

dirname = "ROLLING-WINDOWS"
# wsize viene impostato nel loop finale

esns = generals.ESN.copy()
sensors = generals.SENSORS.tolist() 
colors = cm.get_cmap('tab10', 8)

def ROLLING_WINDOWS_BY_SNAPSHOT_ALL_SENSORS():
    check_init()
    
    # Assumiamo che ci siano al massimo 16 sensori per una griglia 4x4.
    # Se ce ne sono di più, verranno plottati solo i primi 16.
    max_plots = 16 

    for esn in esns:
        # Correctly prepare columns for filtering the main dataframe for this ESN
        cols_for_esn_df = sensors + ['Snapshot']
        esn_df_all_data = generals.filter(train, "ESN", esn, cols_for_esn_df)

        if esn_df_all_data.empty: 
            print(f"Skipping ESN {esn}: No data found.")
            continue

        for pl, p_data in points.items(): 
            # Create output directory for current ESN and point type
            path = generals.plot_path(dirname, str(esn), pl, wsize)
            
            # --- MODIFICA: Creazione della griglia 4x4 ---
            fig, axes = plt.subplots(4, 4, figsize=(24, 20)) # Dimensioni grandi per leggibilità
            axes_flat = axes.flatten() # Appiattiamo la matrice 4x4 in un array 1D per iterare facilmente
            
            plotted_count = 0

            for idx, s in enumerate(sensors):
                # Se superiamo i 16 sensori, interrompiamo il loop per non rompere la griglia
                if idx >= max_plots:
                    break

                ax = axes_flat[idx] # Selezioniamo l'asse corrente

                # Filter data specific to the current sensor and ESN
                if s not in esn_df_all_data.columns or esn_df_all_data[s].dropna().empty:
                    print(f"Skipping Sensor {s} for ESN {esn}: No valid data.")
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center') # Scriviamo "No Data" nel plot vuoto
                    continue

                current_sensor_raw_df = esn_df_all_data[[s, 'Snapshot']].copy()
                
                # Calculate rolling means for each snapshot
                rolling_means_by_snapshot = {}
                for snap_idx in range(1, 9):
                    df_snap_s = generals.filter(current_sensor_raw_df, "Snapshot", snap_idx)
                    if not df_snap_s.empty and not df_snap_s[s].dropna().empty:
                        if len(df_snap_s[s].dropna()) >= wsize:
                            rolling_means_by_snapshot[snap_idx] = df_snap_s[s].rolling(window=wsize).mean()
                        elif len(df_snap_s[s].dropna()) > 0:
                            rolling_means_by_snapshot[snap_idx] = df_snap_s[s].rolling(window=wsize, min_periods=1).mean()
                
                valid_rolling_means_series = [
                    rm for rm in rolling_means_by_snapshot.values() 
                    if not rm.empty and not rm.isnull().all()
                ]

                if not valid_rolling_means_series:
                    ax.text(0.5, 0.5, 'No Valid Rolling Data', ha='center', va='center')
                    continue

                # Determine ymin and ymax
                ymin_plot = min(rm.min() for rm in valid_rolling_means_series)
                ymax_plot = current_sensor_raw_df[s].max() 
                
                # --- PLOTTING SULL'ASSE 'ax' ---
                
                # Vertical lines
                relevant_point_indices = p_data.loc[p_data['ESN'] == esn].index
                if not relevant_point_indices.empty:
                    plot_x_min = current_sensor_raw_df.index.min()
                    plot_x_max = current_sensor_raw_df.index.max()
                    points_in_x_range = relevant_point_indices[(relevant_point_indices >= plot_x_min) & (relevant_point_indices <= plot_x_max)]

                    if not points_in_x_range.empty:
                        ax.vlines(
                            points_in_x_range,
                            ymin=ymin_plot, 
                            ymax=ymax_plot,
                            colors='red', 
                            linestyles='dashed', 
                            label=pl if idx == 0 else "", # Metti label solo nel primo plot per non affollare la legenda
                            alpha=0.7
                        )
                
                # Plot rolling means
                for snap_idx in range(1, 9):
                    if snap_idx in rolling_means_by_snapshot:
                        rm_series = rolling_means_by_snapshot[snap_idx]
                        if not rm_series.empty and not rm_series.isnull().all():
                            ax.plot(rm_series.index, rm_series.values, label=f"Snap {snap_idx}", color=colors(snap_idx - 1))
                
                ax.set_title(f'Sensor: {s}') # Titolo del singolo subplot
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # Gestione legenda per ogni subplot o ridotta
                # ax.legend(fontsize='x-small') 
                
                plotted_count += 1

            # --- PULIZIA E SALVATAGGIO ---
            
            # Nascondi gli assi vuoti (se sensors < 16)
            for j in range(len(sensors), max_plots):
                axes_flat[j].axis('off')

            # Titolo generale dell'immagine
            fig.suptitle(f'ESN {esn} - Rolling Window (Size: {wsize}) - Event: {pl}', fontsize=16)
            
            # Aggiusta il layout per non sovrapporre i titoli
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            
            # Salva una sola immagine per ESN
            final_save_path = f"{path}/ALL_SENSORS_ESN_{esn}_{pl}_w{wsize}.png"
            plt.savefig(final_save_path)
            plt.close(fig) # Importante: chiude la figura per liberare memoria
            print(f"Saved grid plot to {final_save_path}")

# Esecuzione del loop sulle window sizes
for m in [10, 25, 50, 100, 200, 500]:
    wsize = m
    print(f"Processing window size: {wsize}")
    ROLLING_WINDOWS_BY_SNAPSHOT_ALL_SENSORS()

# %%
