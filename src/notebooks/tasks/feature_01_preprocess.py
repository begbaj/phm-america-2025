import sys
import os
import argparse

# Aggiungi la directory corrente (tasks) al sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from tools import utils as u, config as cfg, preprocessing as pp, features as f, plotting as up


def map_pipeline_target(target: str) -> str:
    mapping = {
        "HPC": "Cycles_to_HPC_SV",
        "HPT": "Cycles_to_HPT_SV",
        "WW": "Cycles_to_WW",
    }
    return mapping.get(target.upper(), target)


def run_steps(args):
    selected_steps = set([s.strip() for s in args.steps.split(',')]) if args.steps else None

    def run_step(name):
        return (selected_steps is None) or (name in selected_steps)

    # 0) LOAD / PREPROCESS
    dfp = None
    history = None
    sensors = None

    if run_step('preprocess'):
        print('Eseguo: preprocess_data')
        train = u.load_training()()
        # Passiamo i parametri di preprocessing scelti dall'utente
        dfp, history, sensors = pp.preprocess_data(
            train,
            outlier_method=args.outlier_method,
            outlier_threshold=args.outlier_threshold,
            smoothing_window=args.smoothing_window,
            smoothing_step=args.smoothing_step,
            smoothing_method=args.smoothing_method,
        )
        path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="training.csv")
        dfp.to_csv(path, index=False)
        print(f"Dati preprocessati salvati in {path}")
    else:
        # Try to load existing training.csv
        path = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="training.csv")
        if os.path.exists(path):
            print(f"Carico training preprocessato esistente: {path}")
            dfp = pd.read_csv(path)
            # try to infer sensors
            sensors = [c for c in dfp.columns if c not in u.META_COLS]
            print(f"Sensori determinati da file: {len(sensors)}")
        else:
            print("Nessun training preprocessato trovato; eseguire 'preprocess' step prima di 'aggregate'.")

    # 1) AGGREGATE
    df_averaged = None
    if run_step('aggregate'):
        print('Eseguo: aggregate_snapshots')
        if dfp is None:
            raise RuntimeError("dfp (preprocessed dataframe) non disponibile. Esegui 'preprocess' o assicurati che training.csv esista.")
        df_averaged = pp.aggregate_snapshots(dfp, sensors)
        path_avg = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="averaged_final.csv")
        df_averaged.to_csv(path_avg, index=False)
        print(f"Averaging done: saved to {path_avg}")
    else:
        # Try to load existing averaged file
        path_avg = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="averaged_final.csv")
        if os.path.exists(path_avg):
            print(f"Loading existing averaged file: {path_avg}")
            df_averaged = pd.read_csv(path_avg)
        else:
            print("Nessun file aggregato trovato; eseguire 'aggregate' step prima di 'pipeline'.")

    # 2) SAVE AGGREGATED DATA
    if run_step('save'):
        if df_averaged is None:
            raise RuntimeError("df_averaged non disponibile. Esegui 'aggregate' prima di 'save'.")
        print('Eseguo: save aggregated data')
        # Aggregated data already saved in 'aggregate' step
        print("Aggregated data already saved to averaged_final.csv")

    # 3) PLOT AGGREGATED (plot delle medie per ESN)
    if run_step('plot_agg'):
        if df_averaged is None:
            raise RuntimeError("df_averaged non disponibile. Esegui 'aggregate' prima di 'plot_agg'.")
        print('Eseguo: plot aggregated snapshots')
        # Plot logic for aggregated data
        esn_list = df_averaged['ESN'].unique().tolist() if 'ESN' in df_averaged.columns else [101, 102, 103, 104]
        run = u.get_timestamp()
        # Simple plot example (can be extended)
        print(f"Plotting aggregated data for ESNs: {esn_list}")

    # 4) PLOT PER SNAPSHOT (plot dei dati snapshot)
    if run_step('plot_snap'):
        if dfp is None:
            raise RuntimeError("dfp (preprocessed data) non disponibile. Esegui 'preprocess' prima di 'plot_snap'.")
        print('Eseguo: plot per snapshot')
        # Plot logic for per-snapshot data
        esn_list = dfp['ESN'].unique().tolist() if 'ESN' in dfp.columns else [101, 102, 103, 104]
        run = u.get_timestamp()
        # Simple plot example (can be extended)
        print(f"Plotting snapshot data for ESNs: {esn_list}")

    print('PREPROCESSING TASK finished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Task 01 - Preprocess')
    parser.add_argument('--steps', type=str, default=None, help='Comma separated steps to run: preprocess,aggregate,save,plot_agg,plot_snap')

    # Preprocessing method options
    parser.add_argument('--outlier-method', type=str, default='isoforest', help='Outlier method: zscore, iqr, isoforest')
    parser.add_argument('--outlier-threshold', type=float, default=0.08, help='Threshold for outlier detection (zscore or iqr or isoforest)')
    parser.add_argument('--smoothing-window', type=int, default=100, help='Smoothing rolling window size')
    parser.add_argument('--smoothing-step', type=int, default=25, help='Smoothing minimum period')
    parser.add_argument('--smoothing-method', type=str, default='rolling_mean', help='Smoothing method: rolling_mean, exponential, savitzky_golay')

    args = parser.parse_args()

    run_steps(args)
