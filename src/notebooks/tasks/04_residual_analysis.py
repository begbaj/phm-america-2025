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
    df, history, sensors = pp.preprocess_data(train)
    df = df.dropna()

    # --- CONFIGURAZIONE CARTELLA OUTPUT ---
    output_folder = "output_plots"
    os.makedirs(output_folder, exist_ok=True)
    print(f"I grafici verranno salvati in: {os.path.abspath(output_folder)}")

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

    event = 'hpt'
    to_next_col = f'to_next_{event}_cycle'

    # OTTIMIZZAZIONE: Usa solo uno o pochi window_size per ridurre il tempo
    # windows_to_test = range(5, 150, 22)  # Troppo lento: 7 window_size * 4 ESN * ~1000 sliding windows = 28k+ fits!
    windows_to_test = [healthy_window]  # Usa il healthy_window definito dall'utente
    
    # Alternativamente, se vuoi più window_size, aumenta lo step del sliding window
    sliding_window_step = 50  # Invece di 1, usa step più grande per ridurre iterazioni

    for window_size in windows_to_test:
        print(f"--- Elaborazione Baseline Healthy Window: {window_size} ---")
        
        # Reset residui
        df[res_cols] = np.nan

        # 1. CALCOLO RESIDUI
        for esn in [101, 102, 103, 104]:
            esn_data = df[df['ESN'] == esn].sort_values('Cycles_Since_New')
            if len(esn_data) < window_size + 2:
                continue
            
            # FIT DEL MODELLO SULLA FINESTRA SANA (Iniziale)
            train_baseline = esn_data.iloc[:window_size]
            X_train_baseline = train_baseline[operating_vars].dropna()
            if len(X_train_baseline) < 5: # Servono almeno alcuni punti per un fit decente
                print(f"  ESN {esn}: Troppi pochi dati validi nella healthy window ({len(X_train_baseline)})")
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
                print(f"  ESN {esn}: Residui calcolati rispetto alla healthy window (0-{window_size})")
                
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

        # 3. GENERAZIONE E SALVATAGGIO GRAFICI
        # Skip plot_residuals_dashboard - ha dipendenze complesse con colonne che non esistono
        # try:
        #     fig_res = up.plot_residuals_dashboard(dfplot, res_cols)
        # Invece, salva i residui a file CSV
        
        try:
            residuals_file = os.path.join(output_folder, f"W{window_size}_residuals_data.csv")
            dfmean[['ESN', 'Cycles_Since_New'] + res_cols].to_csv(residuals_file, index=False)
            print(f"Saved residuals data: {residuals_file}")
        except Exception as e:
            print(f"Warning: Impossibile salvare residuals data: {e}")

        # Aggregazione a livello motore
        agg_logic = {col: 'median' for col in res_cols if col in dfmean.columns}
        if to_next_col_val and to_next_col_val in dfmean.columns: 
            agg_logic[to_next_col_val] = 'first'
        
        # Aggiungi snap_index prima dell'aggregazione
        if 'snap_index' not in dfmean.columns:
            dfmean['snap_index'] = range(len(dfmean))
            agg_logic['snap_index'] = 'first'
        
        dfmedian = dfmean.groupby(['ESN']).agg(agg_logic).reset_index()
        
        print(f"  dfmedian shape: {dfmedian.shape}, columns: {dfmedian.columns.tolist()[:5]}...")
        print(f"  Residuals available: {[c for c in res_cols if c in dfmedian.columns]}")

        try:
            # Crea un plot semplice per verificare i dati
            if len(dfmedian) > 0 and any(c in dfmedian.columns for c in res_cols):
                # Scegli la prima colonna residuo disponibile
                res_col_to_plot = next((c for c in res_cols if c in dfmedian.columns), None)
                if res_col_to_plot:
                    # Crea snap_index se non esiste (per il plotting)
                    if 'snap_index' not in dfmedian.columns:
                        dfmedian['snap_index'] = range(len(dfmedian))
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    for esn in [101, 102, 103, 104]:
                        esn_data = dfmedian[dfmedian['ESN'] == esn].sort_values('snap_index')
                        if len(esn_data) > 0:
                            ax.plot(esn_data['snap_index'], esn_data[res_col_to_plot], 
                                   label=f'ESN {esn}', linewidth=2, marker='o')
                    ax.set_xlabel('Snap Index')
                    ax.set_ylabel(f'{res_col_to_plot}')
                    ax.set_title(f'Engine-Level Residuals: {res_col_to_plot}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    filename_eng = f"W{window_size}_engine_residuals.png"
                    path_eng = os.path.join(output_folder, filename_eng)
                    fig.savefig(path_eng, bbox_inches='tight', dpi=100)
                    plt.close(fig)
                    print(f"Saved: {filename_eng}")
                else:
                    print(f"  No residual columns found in dfmedian")
            else:
                print(f"  Skipping engine residuals plot: empty data or no residuals")
        except Exception as e:
            print(f"Warning: Impossibile salvare engine residuals: {e}")
            import traceback
            traceback.print_exc()

        # Plot RUL Comparison (Residuals vs Actual RUL)
        try:
            if rul_column in dfmean.columns:
                # Crea aggregazione con RUL per il confronto
                dfrul = dfmean.copy()
                if 'snap_index' not in dfrul.columns:
                    dfrul['snap_index'] = range(len(dfrul))
                
                # Aggregazione
                agg_rul = {col: 'median' for col in res_cols if col in dfrul.columns}
                agg_rul['snap_index'] = 'first'
                agg_rul[rul_column] = 'mean'  # RUL medio per motore
                
                dfrul_agg = dfrul.groupby(['ESN']).agg(agg_rul).reset_index()
                
                if len(dfrul_agg) > 0:
                    # Seleziona una colonna residuo per il confronto
                    res_col_compare = next((c for c in res_cols if c in dfrul_agg.columns), None)
                    if res_col_compare and rul_column in dfrul_agg.columns:
                        fig, ax1 = plt.subplots(figsize=(14, 7))
                        
                        # Plot residui sull'asse sinistro
                        color_residual = 'tab:blue'
                        ax1.set_xlabel('Engine (ESN)', fontsize=12, fontweight='bold')
                        ax1.set_ylabel(f'Residuals ({res_col_compare})', color=color_residual, fontsize=12, fontweight='bold')
                        ax1.tick_params(axis='y', labelcolor=color_residual)
                        
                        bars1 = ax1.bar([x - 0.2 for x in range(len(dfrul_agg))], dfrul_agg[res_col_compare], 
                                       width=0.35, label=f'{res_col_compare}', color=color_residual, alpha=0.7)
                        ax1.grid(True, alpha=0.3, axis='y')
                        
                        # Plot RUL sull'asse destro
                        ax2 = ax1.twinx()
                        color_rul = 'tab:green'
                        ax2.set_ylabel(f'Actual RUL ({target_rul})', color=color_rul, fontsize=12, fontweight='bold')
                        ax2.tick_params(axis='y', labelcolor=color_rul)
                        
                        bars2 = ax2.bar([x + 0.2 for x in range(len(dfrul_agg))], dfrul_agg[rul_column], 
                                       width=0.35, label=f'{target_rul} RUL', color=color_rul, alpha=0.7)
                        
                        # Etichette X-axis
                        ax1.set_xticks(range(len(dfrul_agg)))
                        ax1.set_xticklabels([f"ESN {int(esn)}" for esn in dfrul_agg['ESN']])
                        
                        ax1.set_title(f'Residuals vs Actual RUL Comparison ({target_rul})', 
                                     fontsize=14, fontweight='bold', pad=20)
                        
                        # Combine legends
                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
                        
                        filename_rul = f"W{window_size}_residuals_vs_rul_{target_rul}.png"
                        path_rul = os.path.join(output_folder, filename_rul)
                        fig.savefig(path_rul, bbox_inches='tight', dpi=100)
                        plt.close(fig)
                        print(f"Saved: {filename_rul}")
                    else:
                        print(f"  Missing columns for RUL comparison plot")
                else:
                    print(f"  No data for RUL comparison")
            else:
                print(f"  RUL column '{rul_column}' not found in data")
        except Exception as e:
            print(f"Warning: Impossibile salvare RUL comparison plot: {e}")
            import traceback
            traceback.print_exc()

        # Health index calculation
        try:
            if 'T3_res' in dfmedian.columns and 'T45_res' in dfmedian.columns:
                # Assicura che snap_index esista
                if 'snap_index' not in dfmedian.columns:
                    dfmedian['snap_index'] = range(len(dfmedian))
                
                # Assicura che target_col esista
                target_col = to_next_col_val
                if not target_col or target_col not in dfmedian.columns:
                    dfmedian['target_rul'] = 100
                    target_col = 'target_rul'
                
                try:
                    # Chiama calculate_hpt_health_index con try-except interno
                    dfmedian_hi = f.calculate_hpt_health_index(dfmedian.copy(), 'T3_res', 'T45_res', to_next_hpc_col=target_col)
                    if 'HI_HPT' in dfmedian_hi.columns:
                        dfmedian['HI_HPT'] = dfmedian_hi['HI_HPT']
                except Exception as hi_e:
                    print(f"  Skipping health index calculation: {hi_e}")
                    dfmedian_hi = dfmedian
                
                # Se HI_HPT esiste, prova a plottare
                if 'HI_HPT' in dfmedian.columns:
                    try:
                        dfmedian_fit, _ = f.fit_hpt_mapping(dfmedian.copy(), 'HI_HPT')
                        dfmedian = dfmedian_fit
                    except Exception as fit_e:
                        print(f"  Skipping HI mapping: {fit_e}")
                    
                    try:
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
