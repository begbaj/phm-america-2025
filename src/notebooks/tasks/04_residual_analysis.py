import sys
import os

# Aggiungi la directory corrente (tasks) al sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

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
    parser.add_argument('--healthy-window', type=int, default=100,
                        help='Number of initial cycles to consider as healthy baseline for regression (default: 100)')
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
    print(f"Using healthy window: {healthy_window} cycles")
    # Carica i dati grezzi (per avere gli snapshot)
    train = u.load_training()()
    
    # Preprocessa mantenendo gli snapshot (non aggregare)
    df, history, sensors = pp.preprocess_data(train, do_missing_fill=False, do_smoothing=False) 
    df = df.dropna()

    # --- CONFIGURAZIONE CARTELLA OUTPUT ---
    output_folder = u.plot_path("residual_analysis")
    print(f"I grafici verranno salvati in: {output_folder}")

    # Definizione vars
    operating_vars = ['Altitude', 'Mach', 'Pamb', 'TAT', 'VAFN', 'VBV', 'Fan_Speed', 'Pt2']
    degradation_vars = [s for s in SENSORS.values() if s not in operating_vars]

    rename_vars_map = {
        'WFuel': 'WFuel_res', 'Core_Speed': 'Core_Speed_res', 'T25': 'T25_res',
        'T3': 'T3_res', 'Ps3': 'Ps3_res', 'T45': 'T45_res', 'P25': 'P25_res', 'T5': 'T5_res'
    }
    res_cols = list(rename_vars_map.values())
    
    # Inizializza colonne residui
    for col in res_cols:
        df[col] = np.nan

    event = target_rul
    windows_to_test = [healthy_window] 
    
    for window_size in windows_to_test:
        print(f"--- Elaborazione Baseline Healthy Window: {window_size} ---")
        
        # Reset residui
        df[res_cols] = np.nan

        # 1. CALCOLO RESIDUI (Logic confirmed: per-engine linear regression)
        for esn in [101, 102, 103, 104]:
            esn_data = df[df['ESN'] == esn].sort_values('Cycles_Since_New')
            # if len(esn_data) < window_size + 2:
            #     continue
            
            # # FIT DEL MODELLO SULLA FINESTRA SANA (Iniziale)
            # train_baseline = esn_data.iloc[:window_size]
            # X_train_baseline = train_baseline[operating_vars].dropna()
            # if len(X_train_baseline) < 5:
            #     print(f"  ESN {esn}: Troppi pochi dati validi nella healthy window ({len(X_train_baseline)})")
            #     continue

            train_baseline = esn_data
            X_train_baseline = train_baseline[operating_vars].dropna()
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
                # print(f"  ESN {esn}: Residui calcolati rispetto alla healthy window (0-{window_size})")
                
            except Exception as e:
                print(f"Warning: Errore nel calcolo residui baseline per ESN {esn}: {e}")
                continue

        # 2. PREPARAZIONE DATI PER PLOTTING (Smoothing)
        dfmean = df.copy()
        # Raggruppa per ESN e applica smoothing sui residui calcolati
        dfmean[res_cols] = dfmean.groupby('ESN')[res_cols].transform(
            lambda x: x.rolling(window=100, min_periods=1).mean()
        ).bfill()
        
        # Assicurati che to_next_col sia presente
        to_next_col_val = None
        for col in df.columns:
            if 'to_next' in col or 'Cycles_to' in col:
                to_next_col_val = col
                if to_next_col_val in df.columns:
                    dfmean[to_next_col_val] = df[to_next_col_val]
                break

        # 3. GENERAZIONE GRAFICI
        
        # --- NEW: Grid Plot (Rows: T3_res, T45_res; Cols: Engines) ---
        try:
            engines = [101, 102, 103, 104]
            fig_grid, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
            
            # Row 1: T3_res
            for j, esn in enumerate(engines):
                ax = axes[0, j]
                data_esn = dfmean[dfmean['ESN'] == esn].sort_values('Cycles_Since_New')
                if not data_esn.empty and 'T3_res' in data_esn.columns:
                    ax.plot(data_esn['Cycles_Since_New'], data_esn['T3_res'], color='tab:blue')
                ax.set_title(f'ESN {esn}')
                if j == 0: ax.set_ylabel('T3 Residual')
                ax.grid(True, alpha=0.3)
            
            # Row 2: T45_res
            for j, esn in enumerate(engines):
                ax = axes[1, j]
                data_esn = dfmean[dfmean['ESN'] == esn].sort_values('Cycles_Since_New')
                if not data_esn.empty and 'T45_res' in data_esn.columns:
                    ax.plot(data_esn['Cycles_Since_New'], data_esn['T45_res'], color='tab:red')
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

        # Aggiungi snap_index se non presente
        if 'snap_index' not in dfmean.columns:
            dfmean['snap_index'] = range(len(dfmean))

        # Skip aggregation to preserve time series for Health Index calculation and Regression
        # We want to fit the trend over time, not on a single aggregated point per engine.
        dfmedian = dfmean.copy()

        # Health index calculation & SCORING
        try:
            if 'T3_res' in dfmedian.columns and 'T45_res' in dfmedian.columns:
                # Assicura che snap_index esista
                if 'snap_index' not in dfmedian.columns:
                    dfmedian['snap_index'] = range(len(dfmedian))
                
                # Assicura che target_col esista
                target_col = rul_column # Usa la colonna passata come argomento (es. Cycles_to_HPT_SV)
                if target_col not in dfmedian.columns:
                    # Cerca fallback
                    target_col = to_next_col_val
                
                if target_col in dfmedian.columns:
                    try:
                        # 1. Calcolo HI
                        dfmedian_hi = f.calculate_hpt_health_index(dfmedian.copy(), 'T3_res', 'T45_res', to_next_hpc_col=target_col)
                        if 'HI_HPT' in dfmedian_hi.columns:
                            dfmedian['HI_HPT'] = dfmedian_hi['HI_HPT']
                        
                        # 2. Mapping Lineare (Predizione)
                        if 'HI_HPT' in dfmedian.columns:
                            dfmedian_fit, _ = f.fit_hpt_mapping(dfmedian.copy(), target_col)
                            dfmedian = dfmedian_fit
                            
                            # 3. Calcolo SCORE (TWE)
                            if 'hpt_linear_pred' in dfmedian.columns:
                                # Filtra nan
                                valid_data = dfmedian.dropna(subset=[target_col, 'hpt_linear_pred'])
                                if not valid_data.empty:
                                    y_true = valid_data[target_col]
                                    y_pred = valid_data['hpt_linear_pred']
                                    
                                    # Calcola Score
                                    score = calculate_score(y_true, y_pred, alpha=0.001, beta=1.0)
                                    print(f"\n==========================================")
                                    print(f">>> TWE SCORE ({target_rul}): {score:.6f}")
                                    print(f"==========================================\n")
                                else:
                                    print("Dati insufficienti per calcolo Score TWE.")
                            else:
                                print("Colonna predizione 'hpt_linear_pred' non generata.")
                                
                    except Exception as hi_e:
                        print(f"  Skipping health index/score calculation: {hi_e}")
                        import traceback
                        traceback.print_exc()
                        
                    # Plot Health Index (esistente)
                    try:
                        if 'HI_HPT' in dfmedian.columns:
                            fig_hi = up.plot_engine_level_hi(dfmedian, ['HI_HPT'], target_col, event)
                            filename_hi = f"W{window_size}_health_index.png"
                            path_hi = os.path.join(output_folder, filename_hi)
                            if fig_hi:
                                fig_hi.savefig(path_hi, bbox_inches='tight')
                                plt.close(fig_hi)
                                print(f"Saved: {filename_hi}")
                    except Exception as plot_e:
                         print(f"  Skipping HI plot: {plot_e}")

        except Exception as e:
            print(f"Warning: Impossibile completare health index: {e}")

if __name__ == "__main__":
    main()
