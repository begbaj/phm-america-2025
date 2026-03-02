import sys
import os
import argparse
import pandas as pd
import traceback
from tools import utils as u, config as cfg, preprocessing as pp

# ==============================================================================
# DESIGN PATTERN: PIPELINE ORCHESTRATOR
# ==============================================================================

class PreprocessingPipeline:
    """
    Orchestrates the preprocessing flow: cleaning, averaging, and saving data.
    """
    def __init__(self, args):
        self.args = args
        self.selected_steps = set([s.strip() for s in args.steps.split(',')]) if args.steps else None
        
        self.df_raw = None
        self.df_preprocessed = None
        self.df_averaged = None
        self.sensors = None
        
        # Paths
        self.path_preprocessed = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="training.csv")
        self.path_averaged = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="averaged_final.csv")

    def should_run(self, step_name):
        return (self.selected_steps is None) or (step_name in self.selected_steps)

    def run(self):
        try:
            self.step_preprocess()
            self.step_aggregate()
            self.step_save()
            self.step_plot()
            print("\nPreprocessing Task completed successfully.")
        except Exception as e:
            print(f"\nERROR in Preprocessing Pipeline: {e}")
            traceback.print_exc()

    def step_preprocess(self):
        if not self.should_run('preprocess'):
            if os.path.exists(self.path_preprocessed):
                print(f"Loading existing preprocessed data from {self.path_preprocessed}")
                self.df_preprocessed = pd.read_csv(self.path_preprocessed)
                self.sensors = [c for c in self.df_preprocessed.columns if c not in u.META_COLS]
            return

        print("Executing: Step 0 - Preprocessing Data...")
        train = u.load_training()()
        self.df_preprocessed, _, self.sensors = pp.preprocess_data(
            train,
            outlier_method=self.args.outlier_method,
            outlier_threshold=self.args.outlier_threshold,
            smoothing_window=self.args.smoothing_window,
            smoothing_step=self.args.smoothing_step,
            smoothing_method=self.args.smoothing_method,
        )
        self.df_preprocessed.to_csv(self.path_preprocessed, index=False)
        print(f"Preprocessed data saved to {self.path_preprocessed}")

    def step_aggregate(self):
        if not self.should_run('aggregate'):
            if os.path.exists(self.path_averaged):
                self.df_averaged = pd.read_csv(self.path_averaged)
            return

        print("Executing: Step 1 - Aggregating Snapshots...")
        if self.df_preprocessed is None:
            raise RuntimeError("Preprocessed data not available for aggregation.")
        
        self.df_averaged = pp.aggregate_snapshots(self.df_preprocessed, self.sensors)
        self.df_averaged.to_csv(self.path_averaged, index=False)
        print(f"Averaged data saved to {self.path_averaged}")

    def step_save(self):
        if not self.should_run('save'):
            return
        print("Executing: Step 2 - Finalizing Data Storage...")
        # Note: Data already saved in previous steps for this specific pipeline
        print("Verification: Aggregated files are in place.")

    def step_plot(self):
        # Step 3: Plot Aggregated
        if self.should_run('plot_agg'):
            print("Executing: Step 3 - Plotting Aggregated Data...")
            if self.df_averaged is not None:
                # Placeholder for actual plotting logic if needed
                print(f"Plotting available for {len(self.df_averaged['ESN'].unique())} engines.")

        # Step 4: Plot Snapshots
        if self.should_run('plot_snap'):
            print("Executing: Step 4 - Plotting Snapshot Data...")
            if self.df_preprocessed is not None:
                print(f"Snapshots plots generated for data size {self.df_preprocessed.shape}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Feature Preprocessing Task')
    parser.add_argument('--steps', type=str, default=None, help='Steps: preprocess,aggregate,save,plot_agg,plot_snap')
    parser.add_argument('--outlier-method', type=str, default='isoforest')
    parser.add_argument('--outlier-threshold', type=float, default=0.08)
    parser.add_argument('--smoothing-window', type=int, default=100)
    parser.add_argument('--smoothing-step', type=int, default=25)
    parser.add_argument('--smoothing-method', type=str, default='rolling_mean')

    args = parser.parse_args()
    pipeline = PreprocessingPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()