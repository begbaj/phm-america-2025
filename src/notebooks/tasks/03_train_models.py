import sys
import os

# Aggiungi la directory corrente (tasks) al sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from tools import utils as u, config as cfg, algorithms as alg

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
    args = parser.parse_args()

    selected = None
    if args.models:
        selected = set([m.strip().lower() for m in args.models.split(",") if m.strip()])

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

    if want(['linear', 'lr', 'linear_regression']):
        print("Training del modello di regressione lineare...")
        alg.train_linear_regression(dff.copy(), dfm, target=target_training)

    if want(['rf', 'random_forest', 'randomforest']):
        print("\nTraining del modello Random Forest...")
        alg.train_random_forest(dff.copy(), dfm, target=target_training)

    if want(['xgb', 'xgboost']):
        print("\nTraining del modello XGBoost...")
        alg.train_xgboost(dff.copy(), dfm, target=target_training)

    if want(['transformer', 'trans']):
        print("\nTraining del modello Transformer...")
        alg.train_transformer(dff.copy(), dfm, target=target_training)
    
if __name__ == "__main__":
    main()
