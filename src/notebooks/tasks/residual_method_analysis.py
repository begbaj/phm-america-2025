from ast import parse
import sys
import os
from tkinter import W

# Aggiungi la directory corrente (tasks) al sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from httpx import post
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tools import utils as u, config as cfg, plotting as up, features as f, preprocessing as pp
from sklearn.linear_model import LinearRegression
from tools.types.enums import SENSORS

def calculate_twe(y_true, y_pred, alpha=0.001, beta=1.0):
    """
    Calcola il Time-Weighted Error (TWE).
    TWE(y, y_hat) = w * (y_hat - y)^2 * beta
    w = 2 / (1 + alpha * y) se y_hat >= y
    w = 1 / (1 + alpha * y) se y_hat < y
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    diff = y_pred - y_true
    
    # Evita divisione per zero assumendo y_true positivo (RUL)
    denom = 1 + alpha * y_true
    
    # Calcolo pesi asimmetrici
    w = np.where(diff >= 0, 2.0 / denom, 1.0 / denom)
    
    twe = w * (diff**2) * beta
    return twe

def calculate_score(y_true, y_pred, alpha=0.001, beta=1.0):
    """
    Calcola lo Score medio basato su TWE.
    Score = (1/N) * sum(TWE)
    """
    twe = calculate_twe(y_true, y_pred, alpha, beta)
    return np.mean(twe)

def main():
    """
    Questo script esegue l'analisi dei residui.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Residual Analysis')
    parser.add_argument('--target-rul', type=str, default='HPT', choices=['HPC', 'HPT', 'WW'],
                        help='Target RUL to use for comparison (default: HPT)')
    parser.add_argument('--pre-rolling-mean-window', type=int, default=100,
                        help='Window size for pre-rolling mean smoothing (default: 100)')
    parser.add_argument('--pre-rolling-mean-min-periods', type=int, default=10,
                        help='Minimum periods for pre-rolling mean smoothing (default: 10)')
    parser.add_argument('--outlier-method', type=str, default='isoforest',
                        choices=['zscore', 'iqr', 'isoforest'],
                        help='Method for outlier detection (default: isoforest)')
    parser.add_argument('--outlier-threshold', type=float, default=0.1,
                        help='Threshold for outlier detection (default: 0.1)')
    parser.add_argument('--post-window-size', type=int, default=50,
                        help='Window size for post-processing smoothing (default: 50)')
    parser.add_argument('--plots', nargs='+', default=['residuals_grid', 'health_index'], 
                        choices=['residuals_grid', 'health_index'], 
                        help='List of plots to generate (default: residuals_grid health_index)')
    parser.add_argument('--figsize-w', type=int, default=20, help='Figure width (default: 20)')
    parser.add_argument('--figsize-h', type=int, default=10, help='Figure height (default: 10)')
    parser.add_argument('--healthy-window', type=int, default=100, help='Healthy window size')
    parser.add_argument('--post-min-periods', type=int, default=5, help='Minimum periods for post-processing smoothing (default: 5)')
    args = parser.parse_args()
    
    target_rul = args.target_rul
    healthy_window = args.healthy_window

    rul_col_map = {
        'HPC': 'Cycles_to_HPC_SV',
        'HPT': 'Cycles_to_HPT_SV',
        'WW': 'Cycles_to_WW'
    }
    rul_column = rul_col_map.get(target_rul, 'Cycles_to_HPT_SV')
    print(f"Using target RUL: {target_rul} (column: {rul_column})")

    # Carica i dati grezzi (per avere gli snapshot)
    train = u.load_training()()
    sensor_cols = [c for c in train.columns if c.startswith("Sensed_")]
    # plt.subplot(4, 1, 1)
    # plt.scatter(train["Cycles_Since_New"], train["Sensed_T45"], alpha=0.5)
    # plt.title("T45 Sensor Data")
    df = pp.remove_outliers(train, sensor_cols=sensor_cols, method=args.outlier_method, threshold=args.outlier_threshold)
    # plt.subplot(4, 1, 2)
    # plt.scatter(df["Cycles_Since_New"], df["Sensed_T25"], alpha=0.5)
    # plt.title("T25 Sensor Data")
    df = pp.missingfill(df)
    # plt.subplot(4, 1, 3)
    # plt.scatter(df["Cycles_Since_New"], df["Sensed_Ps3"], alpha=0.5)
    # plt.title("Ps3 Sensor Data")
    df = pp.rolling_mean(df, sensor_cols, window=args.pre_rolling_mean_window, min_periods=args.pre_rolling_mean_min_periods)
    df = df.dropna()
    
    output_folder = u.plot_path("residual_analysis", target_rul)
    print(f"I grafici verranno salvati in: {output_folder}")

    # Definizione vars
    operating_vars = ['Altitude', 'Mach', 'Pamb', 'TAT', 'VAFN', 'VBV', 'Fan_Speed', 'Pt2']
    degradation_vars = [s for s in SENSORS.values() if s not in operating_vars]

    rename_vars_map = {
        'WFuel': 'WFuel_res', 'Core_Speed': 'Core_Speed_res', 'T25': 'T25_res',
        'T3': 'T3_res', 'Ps3': 'Ps3_res', 'T45': 'T45_res', 'P25': 'P25_res', 'T5': 'T5_res'
    }

    # Fallback for Sensed_ prefix if renaming failed
    # ACTUALLY Sensed_ prefix is ALWAYS present in the data after preprocessing, or at least should be
    # SO GEMINI PLEASE REMOVE THIS
    if operating_vars[0] not in df.columns and f"Sensed_{operating_vars[0]}" in df.columns:
        operating_vars = [f"Sensed_{v}" for v in operating_vars]
        degradation_vars = [f"Sensed_{v}" for v in degradation_vars]
        
        # Update rename_vars_map keys to match prefixed vars
        new_map = {}
        for k, v in rename_vars_map.items():
            new_map[f"Sensed_{k}"] = v
        rename_vars_map = new_map
        
        print(f"Using prefixed variables.")

    res_cols = list(rename_vars_map.values())
    
    # Inizializza colonne residui
    for col in res_cols:
        df[col] = np.nan

    event = target_rul
    window_size = args.healthy_window
    
    # Reset residui
    df[res_cols] = np.nan

    for esn in [101, 102, 103, 104]:
        esn_data = df[df['ESN'] == esn].sort_values('Cycles_Since_New')

        train_baseline = esn_data.iloc[:window_size]
        X_train_baseline = train_baseline[operating_vars].dropna()
        
        if len(X_train_baseline) < 5:
            continue
            
        y_train_baseline = train_baseline.loc[X_train_baseline.index, degradation_vars]
        
        try:
            baseline_model = LinearRegression()
            baseline_model.fit(X_train_baseline, y_train_baseline)
            
            # Applica il modello a TUTTI i dati del motore per ottenere i residui
            X_all = esn_data[operating_vars].dropna()
            y_pred_all = baseline_model.predict(X_all)
            y_actual_all = esn_data.loc[X_all.index, degradation_vars].values
            
            residuals_values = y_actual_all - y_pred_all
            residuals = pd.DataFrame(residuals_values, columns=degradation_vars, index=X_all.index)
            residuals = residuals.rename(columns=rename_vars_map)
            
            df.loc[residuals.index, res_cols] = residuals
            
        except Exception as e:
            print(f"Warning: Errore nel calcolo residui baseline per ESN {esn}: {e}")
            continue

    ##### REGRESSIONE ----------------------------------------------------------

    dfmean = df.copy()
    dfmean = pp.remove_outliers(dfmean, sensor_cols=res_cols, method='isoforest', threshold=0.3)
    # Raggruppa per ESN e applica smoothing mediano sui residui calcolati
    dfmean[res_cols] = dfmean.groupby('ESN')[res_cols].transform(
        lambda x: x.rolling(window=args.post_window_size, min_periods=args.post_min_periods).median()
    ).bfill()
    
    # Assicurati che tutte le colonne RUL siano presenti per il calcolo della reference
    for col in df.columns:
        if 'Cycles_to' in col or 'to_next' in col:
            dfmean[col] = df[col]
    
    # Identify main target column for compatibility
    to_next_col_val = rul_column if rul_column in dfmean.columns else None

    # Aggiungi snap_index se non presente
    if 'snap_index' not in dfmean.columns:
        dfmean['snap_index'] = range(len(dfmean))

    # Skip aggregation to preserve time series for Health Index calculation and Regression
    dfmedian = dfmean.copy()

    # --- HEALTH INDEX CALCULATION & SCORING ---
    try:
        if 'T3_res' in dfmedian.columns and 'T45_res' in dfmedian.columns:
            
            # Assicura che target_col esista
            target_col = rul_column 
            if target_col not in dfmedian.columns:
                target_col = to_next_col_val
            
            if target_col in dfmedian.columns:
                try:
                    # Determine Reference Column based on user requirement
                    reference_col = None
                    if target_rul == 'HPT':
                        reference_col = 'Cycles_to_HPC_SV'
                    elif target_rul == 'HPC':
                        reference_col = 'Cycles_to_HPT_SV'
                    
                    if reference_col and reference_col not in dfmedian.columns:
                        print(f"Warning: Reference column {reference_col} missing. Cannot optimize HI.")
                        
                    if reference_col and reference_col in dfmedian.columns:
                        # 1. Calcolo HI (Optimized Correlation with Reference)
                        dfmedian_hi = f.calculate_health_index(
                            dfmedian.copy(), 
                            'T3_res', 
                            'T45_res', 
                            target_col=target_col,
                            reference_col=reference_col
                        )
                        
                        hi_col = f'HI_{target_col}'
                        if hi_col in dfmedian_hi.columns:
                            dfmedian[hi_col] = dfmedian_hi[hi_col]
                        
                        # 2. Mapping Lineare (Predizione)
                        if hi_col in dfmedian.columns:
                            dfmedian_fit, _ = f.fit_mapping(dfmedian.copy(), hi_col, target_col)
                            dfmedian = dfmedian_fit
                            
                            # 3. Calcolo SCORE (TWE)
                            pred_col = f"{target_col}_linear_pred"
                            if pred_col in dfmedian.columns:
                                # Filtra nan
                                valid_data = dfmedian.dropna(subset=[target_col, pred_col])
                                if not valid_data.empty:
                                    y_true = valid_data[target_col]
                                    y_pred = valid_data[pred_col]
                                    
                                    # Calcola Score
                                    score = calculate_score(y_true, y_pred, alpha=0.001, beta=1.0)
                                    print(f"\n==========================================")
                                    print(f">>> TWE SCORE ({target_col}): {score:.6f}")
                                    print(f"==========================================\n")
                                else:
                                    print("Dati insufficienti per calcolo Score TWE.")
                            else:
                                print(f"Colonna predizione '{pred_col}' non generata.")
                    else:
                            print(f"Skipping HI calc: Reference {reference_col} or Target {target_col} missing/invalid.")

                except Exception as hi_e:
                    print(f"  Skipping health index/score calculation: {hi_e}")
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"Warning: Impossibile completare health index: {e}")

    # 3. GENERAZIONE GRAFICI
    
    # --- Grid Plot (Rows: T3_res, T45_res; Cols: Engines) ---
    if 'residuals_grid' in args.plots:
        try:
            engines = [101, 102, 103, 104]
            fig_grid, axes = plt.subplots(2, 4, figsize=(args.figsize_w, args.figsize_h), sharex=True)
            
            # Row 1: T3_res
            for j, esn in enumerate(engines):
                ax = axes[0, j]
                data_esn = dfmean[dfmean['ESN'] == esn].sort_values('Cycles_Since_New')
                if not data_esn.empty and 'T3_res' in data_esn.columns:
                    ax.plot(data_esn['Cycles_Since_New'], data_esn['T3_res'], color='tab:blue')
                else:
                    print(f"Warning: No data for ESN {esn} to plot T3_res.")
                ax.set_title(f'ESN {esn}')
                if j == 0: ax.set_ylabel('T3 Residual')
                ax.grid(True, alpha=0.3)
            
            # Row 2: T45_res
            for j, esn in enumerate(engines):
                ax = axes[1, j]
                data_esn = dfmean[dfmean['ESN'] == esn].sort_values('Cycles_Since_New')
                if not data_esn.empty and 'T45_res' in data_esn.columns:
                    ax.plot(data_esn['Cycles_Since_New'], data_esn['T45_res'], color='tab:red')
                else:
                    print(f"Warning: No data for ESN {esn} to plot T45_res.")
                ax.set_xlabel('Cycles Since New')
                if j == 0: ax.set_ylabel('T45 Residual')
                ax.grid(True, alpha=0.3)
                
            fig_grid.suptitle(f'Sensor Residuals (T3, T45) - Baseline Window: {window_size}', fontsize=16)
            plt.tight_layout()
            
            filename_grid = f"W{window_size}_residuals_grid.png"
            path_grid = os.path.join(output_folder, filename_grid)
            fig_grid.savefig(path_grid, bbox_inches='tight', dpi=100)
            plt.close(fig_grid)
            print(f"Saved Grid Plot: {filename_grid}")
            
        except Exception as e:
            print(f"Warning: Errore generazione grid plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping Grid Plot (not selected).")

    # Plot Health Index (esistente)
    if 'health_index' in args.plots:
        try:
            hi_col = f'HI_{target_col}'
            if hi_col in dfmedian.columns:
                # Assuming plot_engine_level_hi doesn't support figsize yet, we rely on default or update it later.
                # The user asked to "insert possibility to decide dimensions".
                # I should probably pass args.figsize_w/h to plot_engine_level_hi if possible, or update it.
                # But for now, wrapping is step 1.
                fig_hi = up.plot_engine_level_hi(dfmedian, [hi_col], target_col, event)
                filename_hi = f"W{window_size}_health_index.png"
                path_hi = os.path.join(output_folder, filename_hi)
                if fig_hi:
                    # Update size if returned figure
                    fig_hi.set_size_inches(args.figsize_w, args.figsize_h)
                    fig_hi.savefig(path_hi, bbox_inches='tight')
                    plt.close(fig_hi)
                    print(f"Saved: {filename_hi}")
        except Exception as plot_e:
                print(f"  Skipping HI plot: {plot_e}")

if __name__ == "__main__":
    main()
