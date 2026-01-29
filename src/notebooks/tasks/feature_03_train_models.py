import sys
import os

# Aggiungi la directory corrente (tasks) al sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from tools import utils as u, config as cfg, algorithms as alg, plotting as up

import argparse

def main():
    """
    Questo script esegue il training dei modelli. Ora accetta --models (comma-separated)
    es: --models linear,rf,xgb,transformer
    Accetta anche --target-training per scegliere il target event (HPC, HPT, WW)
    """
    parser = argparse.ArgumentParser(description="Train selected models")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated models: linear,rf,xgb,transformer")
    parser.add_argument("--target-training", type=str, default="HPC", choices=['HPC', 'HPT', 'WW'],
                        help="Target event for training (default: HPC)")
    
    # Model specific parameters
    parser.add_argument("--rf-n-estimators", type=int, default=200, help="Random Forest: n_estimators")
    parser.add_argument("--rf-max-depth", type=int, default=15, help="Random Forest: max_depth")
    
    parser.add_argument("--xgb-n-estimators", type=int, default=600, help="XGBoost: n_estimators")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05, help="XGBoost: learning_rate")
    parser.add_argument("--xgb-max-depth", type=int, default=5, help="XGBoost: max_depth")
    
    parser.add_argument("--trans-epochs", type=int, default=400, help="Transformer: epochs")
    parser.add_argument("--trans-batch-size", type=int, default=64, help="Transformer: batch_size")
    parser.add_argument("--trans-learning-rate", type=float, default=0.0005, help="Transformer: learning_rate")

    parser.add_argument("--healthy-window", type=int, default=None, help="Window of cycles to consider as healthy baseline")

    args = parser.parse_args()

    selected = None
    if args.models:
        selected = set([m.strip().lower() for m in args.models.split(',') if m.strip()])

    def want(aliases):
        # if no selection specified, run all
        if selected is None:
            return True
        return any(a in selected for a in aliases)

    # Seleziona il target per il training
    target_training = args.target_training
    print(f"\n{'='*50}")
    print(f"Training Target: {target_training}")
    print(f"{'='*50}\n")
    
    # Carica i dati delle feature in base al target scelto
    data_filename = f"training_feature_{target_training}"
    dfm = pd.read_csv(u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"{data_filename}_metadata.csv"))
    dfd = pd.read_csv(u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"{data_filename}_data.csv"))
    
    print(f"Loading feature metadata from: {data_filename}_metadata.csv")
    print(f"Metadata columns: {dfm.columns.tolist()}")
    print(f"Metadata shape: {dfm.shape}")
    print(f"First few rows of metadata:\n{dfm.head()}\n")
    
    # Evitiamo colonne duplicate (es. sensori presenti sia in data che in metadata):
    meta_only_cols = [c for c in dfm.columns if c not in dfd.columns]
    dff = pd.concat([dfd, dfm[meta_only_cols]], axis=1)

    print(f"Modelli selezionati: {sorted(selected) if selected is not None else 'tutti'}")

    all_results = {}

    if want(['linear', 'lr', 'linear_regression']):
        print("Training del modello di regressione lineare...")
        model, y_pred = alg.train_linear_regression(dff.copy(), dfm, target=target_training, 
                                                    filename=f"LR_{target_training}_prediction.png",
                                                    show_plot=False)
        
        test_df = dff[dff["ESN"] == 104]
        if len(test_df) == 0:
            split_idx = int(len(dff) * 0.8)
            test_df = dff.iloc[split_idx:]
        
        target_map = {'HPC': 'Cycles_to_HPC_SV', 'HPT': 'Cycles_to_HPT_SV', 'WW': 'Cycles_to_WW'}
        target_col = target_map.get(target_training, 'RUL')
        if target_col in test_df.columns:
            y_test = test_df[target_col].dropna()
            all_results['Linear Regression'] = (y_test, y_pred)

    if want(['rf', 'random_forest', 'randomforest']):
        print(f"\nTraining del modello Random Forest (n_estimators={args.rf_n_estimators}, max_depth={args.rf_max_depth})...")
        model, y_pred, _ = alg.train_random_forest(dff.copy(), dfm, target=target_training, 
                                n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
                                filename=f"RF_{target_training}_prediction.png",
                                show_plot=False)
        
        test_df = dff[dff["ESN"] == 104]
        if len(test_df) == 0: 
            split_idx = int(len(dff) * 0.8)
            test_df = dff.iloc[split_idx:]
        
        target_map = {'HPC': 'Cycles_to_HPC_SV', 'HPT': 'Cycles_to_HPT_SV', 'WW': 'Cycles_to_WW'}
        target_col = target_map.get(target_training, 'RUL')
        if target_col in test_df.columns:
            y_test = test_df[target_col].dropna()
            all_results['Random Forest'] = (y_test, y_pred)

    if want(['xgb', 'xgboost']):
        print(f"\nTraining del modello XGBoost (n_estimators={args.xgb_n_estimators}, lr={args.xgb_learning_rate}, max_depth={args.xgb_max_depth})...")
        model, y_pred = alg.train_xgboost(dff.copy(), dfm, target=target_training,
                          n_estimators=args.xgb_n_estimators, learning_rate=args.xgb_learning_rate, max_depth=args.xgb_max_depth,
                          filename=f"XGB_{target_training}_prediction.png",
                          show_plot=False)
        
        test_df = dff[dff["ESN"] == 104]
        if len(test_df) == 0:
            split_idx = int(len(dff) * 0.8)
            test_df = dff.iloc[split_idx:]
            
        target_map = {'HPC': 'Cycles_to_HPC_SV', 'HPT': 'Cycles_to_HPT_SV', 'WW': 'Cycles_to_WW'}
        target_col = target_map.get(target_training, 'RUL')
        if target_col in test_df.columns:
            y_test = test_df[target_col].dropna()
            all_results['XGBoost'] = (y_test, y_pred)

    if want(['transformer', 'trans']):
        print(f"\nTraining del modello Transformer (epochs={args.trans_epochs}, batch_size={args.trans_batch_size}, lr={args.trans_learning_rate})...")
        model, y_pred = alg.train_transformer(dff.copy(), dfm, target=target_training,
                              epochs=args.trans_epochs, batch_size=args.trans_batch_size, lr=args.trans_learning_rate,
                              filename=f"Transformer_{target_training}_prediction.png",
                              show_plot=False)
        
        test_df = dff[dff["ESN"] == 104]
        if len(test_df) == 0:
            split_idx = int(len(dff) * 0.8)
            test_df = dff.iloc[split_idx:]
            
        target_map = {'HPC': 'Cycles_to_HPC_SV', 'HPT': 'Cycles_to_HPT_SV', 'WW': 'Cycles_to_WW'}
        target_col = target_map.get(target_training, 'RUL')
        if target_col in test_df.columns:
            y_test_full = test_df[target_col].dropna().values
            y_test_seq = y_test_full[-len(y_pred):]
            all_results['Transformer'] = (y_test_seq, y_pred)

    if all_results:
        print("\nGenerazione Dashboard comparativa...")
        up.plot_training_dashboard(all_results, target=target_training, 
                                   filename=f"Training_Dashboard_{target_training}.png")

if __name__ == "__main__":
    main()