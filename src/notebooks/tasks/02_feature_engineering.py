import sys
import os

# Aggiungi la directory corrente (tasks) al sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import argparse
from tools import utils as u, config as cfg, features as f

def main(target_arg=None, statistical_features_arg=None, window_arg=None, step_arg=None):
    """
    Questo script esegue la feature engineering.
    I parametri possono venire da CLI o GUI.
    """
    # Carica i dati aggregati
    path_avg = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="averaged_final.csv")
    df_averaged = pd.read_csv(path_avg)

    # Definisce le feature da calcolare
    to_calc = df_averaged.copy()
    features = [] # [f.FThermalEfficiency.DELTA_PR_TH_HPC_2]
    target = target_arg if target_arg is not None else 'HPC'
    
    # Statistical features (from GUI or defaults)
    if statistical_features_arg:
        statistical_features = [s.strip() for s in statistical_features_arg.split(',') if s.strip()]
    else:
        statistical_features = ['mean', 'rms']
    
    # Window and step parameters
    window = window_arg if window_arg is not None else 100
    step = step_arg if step_arg is not None else 25
    
    # Mapping dei target ai nomi originali delle colonne
    target_mapping = {
        'HPC': 'Cycles_to_HPC_SV',
        'HPT': 'Cycles_to_HPT_SV',
        'WW': 'Cycles_to_WW'
    }
    fulltarget = target_mapping.get(target, f'to_next_{target.lower()}_cycle')
    colname = f.get_all_performance_colnames()

    print(f"Target: {target}")
    print(f"Statistical features: {statistical_features}")
    print(f"Window: {window}, Step: {step}")

    if target == "HPC":
        dff, val = f.pipeline_hpc(to_calc, features, colname, statistical_features, window=window, step=step, stat_groupby=["ESN"], stat_sortby=["ESN", "esn_index"], target=fulltarget)
    elif target == "HPT":
        dff, val = f.pipeline_hpt(df_averaged, features, colname, statistical_features, window=window, step=step, stat_groupby=["ESN"], stat_sortby=["ESN", "esn_index"], target=fulltarget)
    elif target == "WW":
        dff, val = f.pipeline_ww(df_averaged, features, colname, statistical_features, window=window, step=step, stat_groupby=["ESN"], stat_sortby=["ESN", "esn_index"], target=fulltarget)
    else:
        print("Target non valido.")
        return

    # Estrazione delle migliori feature
    tot = val.sort_values(by='tot_val', key=abs, ascending=False).head(10)
    print("Migliori feature:")
    print(tot)

    # Salvataggio delle feature
    run = u.get_timestamp()
    path_feat_meta = u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"training_feature_{target}_metadata.csv")
    tot.to_csv(path_feat_meta, index=False)
    print(f"Metadati delle feature salvati in {path_feat_meta}")

    path_feat_data = u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"training_feature_{target}_data.csv")
    
    # Includi le colonne essenziali (ESN, indici, target RUL)
    essential_cols = ['ESN', 'Cycles_Since_New', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV', 'Cycles_to_WW']
    cols_to_save = []
    
    # Aggiungi le feature selezionate
    for feat in tot["feature"].tolist():
        if feat in dff.columns:
            cols_to_save.append(feat)
    
    # Aggiungi le colonne essenziali se presenti
    for col in essential_cols:
        if col in dff.columns and col not in cols_to_save:
            cols_to_save.append(col)
    
    # Se ci sono colonne mancanti, avvisa ma continua
    if not cols_to_save:
        print("Warning: No columns found to save, using default feature columns")
        cols_to_save = tot["feature"].tolist()
    
    dff[cols_to_save].to_csv(path_feat_data, index=False)
    print(f"Dati delle feature salvati in {path_feat_data}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run feature engineering task')
    parser.add_argument('--target', choices=['HPC', 'HPT', 'WW'], help='Target event to compute features for')
    parser.add_argument('--statistical-features', type=str, help='Comma-separated statistical features: mean,rms,std,min,max')
    parser.add_argument('--pipeline-window', type=int, help='Rolling window size')
    parser.add_argument('--pipeline-step', type=int, help='Minimum period')
    args = parser.parse_args()
    main(args.target, args.statistical_features, args.pipeline_window, args.pipeline_step)
