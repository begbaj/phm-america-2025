import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import traceback
from tools import utils as u, config as cfg, plotting as up, features as f, preprocessing as pp, algorithms as algo
from sklearn.linear_model import LinearRegression
from tools.types.enums import SENSORS

# ==============================================================================
# DESIGN PATTERN: STRATEGY & COMMAND ORCHESTRATOR
# ==============================================================================

class ResidualAnalysisPipeline:
    """
    Orchestrator for the Residual Analysis workflow.
    Uses a Pipeline-like structure to separate concerns.
    """
    def __init__(self, args):
        self.args = args
        self.target_rul = args.target_rul
        self.rul_column = self._get_rul_column(args.target_rul)
        self.output_folder = u.plot_path("residual_analysis", args.target_rul)
        
        self.df = None          # Raw/Preprocessed data
        self.df_residuals = None # Data with residuals
        self.df_final = None     # Data with HI and predictions
        
        # Mappings
        self.operating_vars = ['Altitude', 'Mach', 'Pamb', 'TAT', 'VAFN', 'VBV', 'Fan_Speed', 'Pt2']
        self.rename_vars_map = {
            'WFuel': 'WFuel_res', 'Core_Speed': 'Core_Speed_res', 'T25': 'T25_res',
            'T3': 'T3_res', 'Ps3': 'Ps3_res', 'T45': 'T45_res', 'P25': 'P25_res', 'T5': 'T5_res'
        }

    def _get_rul_column(self, target):
        rul_map = {'HPC': 'Cycles_to_HPC_SV', 'HPT': 'Cycles_to_HPT_SV', 'WW': 'Cycles_to_WW'}
        return rul_map.get(target, 'Cycles_to_HPT_SV')

    def load_and_preprocess(self):
        """Step 1: Load data and apply initial cleaning."""
        print(f"Loading data for target RUL: {self.target_rul}...")
        train = u.load_training()()
        
        # Initial outlier removal and missing fill
        sensor_cols = [c for c in train.columns if c.startswith("Sensed_")]
        df = pp.remove_outliers(train, sensor_cols=sensor_cols, method=self.args.outlier_method, threshold=self.args.outlier_threshold)
        df = pp.missingfill(df)
        
        # Smoothing
        df = pp.rolling_mean(df, sensor_cols, window=self.args.pre_rolling_mean_window, min_periods=self.args.pre_rolling_mean_min_periods)
        self.df = df.dropna()
        
        # Check for prefixed variables
        if self.operating_vars[0] not in self.df.columns and f"Sensed_{self.operating_vars[0]}" in self.df.columns:
            self.operating_vars = [f"Sensed_{v}" for v in self.operating_vars]
            # Update rename map keys
            self.rename_vars_map = {f"Sensed_{k}": v for k, v in self.rename_vars_map.items()}
            print("Using prefixed variables for residuals.")

    def compute_residuals(self):
        """Step 2: Fit baseline models and calculate residuals."""
        print("Computing residuals...")
        degradation_vars = list(self.rename_vars_map.keys())
        
        self.df_residuals = algo.fit_baseline_residuals(
            self.df, 
            self.operating_vars, 
            degradation_vars, 
            self.rename_vars_map,
            healthy_window=self.args.healthy_window
        )

    def prepare_analysis_data(self):
        """Step 3: Post-processing of residuals (smoothing)."""
        print("Smoothing residuals...")
        df_res = self.df_residuals.copy()
        res_cols = list(self.rename_vars_map.values())
        
        # Apply outlier removal on residuals
        df_res = pp.remove_outliers(df_res, sensor_cols=res_cols, method='isoforest', threshold=0.3)
        
        # Smooth residuals
        df_res[res_cols] = df_res.groupby('ESN')[res_cols].transform(
            lambda x: x.rolling(window=self.args.post_window_size, min_periods=self.args.post_min_periods).median()
        ).bfill()
        
        # Ensure RUL columns are present
        for col in self.df_residuals.columns:
            if 'Cycles_to' in col or 'to_next' in col:
                df_res[col] = self.df_residuals[col]
        
        if 'snap_index' not in df_res.columns:
            df_res['snap_index'] = range(len(df_res))
            
        self.df_final = df_res

    def compute_health_index(self):
        """Step 4: Calculate HI and fit mapping to RUL."""
        print("Calculating Health Index and Score...")
        target_col = self.rul_column
        
        # Use the same target RUL as reference for Alpha optimization
        reference_col = self.rul_column
        
        if reference_col not in self.df_final.columns:
            print(f"Warning: Reference column {reference_col} missing. Skipping optimization.")
            return

        # 1. Calculate HI using Strategy (Correlation Optimization with Target RUL)
        self.df_final = f.calculate_health_index(
            self.df_final, 
            'T3_res', 'T45_res', 
            target_col=target_col,
            reference_col=reference_col
        )
        
        # 2. Fit Mapping HI -> RUL
        hi_col = f'HI_{target_col}'
        if hi_col in self.df_final.columns:
            self.df_final, _ = f.fit_mapping(self.df_final, hi_col, target_col)
            self._evaluate_score(target_col)

    def _evaluate_score(self, target_col):
        pred_col = f"{target_col}_linear_pred"
        if pred_col in self.df_final.columns:
            valid = self.df_final.dropna(subset=[target_col, pred_col])
            if not valid.empty:
                score = calculate_score(valid[target_col], valid[pred_col])
                print(f"\n==========================================")
                print(f">>> TWE SCORE ({target_col}): {score:.6f}")
                print(f"==========================================\n")

    def generate_plots(self):
        """Step 5: Visualization."""
        print(f"Generating plots in: {self.output_folder}")
        
        # Grid Plot
        if 'residuals_grid' in self.args.plots:
            self._plot_residuals_grid()
            
        # HI Plot
        if 'health_index' in self.args.plots:
            self._plot_health_index()

    def _plot_residuals_grid(self):
        try:
            engines = [101, 102, 103, 104]
            fig, axes = plt.subplots(2, 4, figsize=(self.args.figsize_w, self.args.figsize_h), sharex=True)
            for j, esn in enumerate(engines):
                data_esn = self.df_final[self.df_final['ESN'] == esn].sort_values('Cycles_Since_New')
                if not data_esn.empty:
                    if 'T3_res' in data_esn.columns: axes[0, j].plot(data_esn['Cycles_Since_New'], data_esn['T3_res'], color='tab:blue')
                    if 'T45_res' in data_esn.columns: axes[1, j].plot(data_esn['Cycles_Since_New'], data_esn['T45_res'], color='tab:red')
                axes[0, j].set_title(f'ESN {esn}')
                if j == 0: 
                    axes[0, j].set_ylabel('T3 Residual')
                    axes[1, j].set_ylabel('T45 Residual')
                axes[0, j].grid(True, alpha=0.3); axes[1, j].grid(True, alpha=0.3)
            
            fig.suptitle(f'Sensor Residuals (T3, T45) - Baseline Window: {self.args.healthy_window}', fontsize=16)
            plt.tight_layout()
            fig.savefig(os.path.join(self.output_folder, f"W{self.args.healthy_window}_residuals_grid.png"), bbox_inches='tight', dpi=100)
            plt.close(fig)
        except Exception as e:
            print(f"Error in grid plot: {e}")

    def _plot_health_index(self):
        try:
            hi_col = f'HI_{self.rul_column}'
            if hi_col in self.df_final.columns:
                fig = up.plot_engine_level_hi(self.df_final, [hi_col], self.rul_column, self.target_rul)
                if fig:
                    fig.set_size_inches(self.args.figsize_w, self.args.figsize_h)
                    fig.savefig(os.path.join(self.output_folder, f"W{self.args.healthy_window}_health_index.png"), bbox_inches='tight')
                    plt.close(fig)
        except Exception as e:
            print(f"Error in HI plot: {e}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def calculate_twe(y_true, y_pred, alpha=0.001, beta=1.0):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = y_pred - y_true
    denom = 1 + alpha * y_true
    w = np.where(diff >= 0, 2.0 / denom, 1.0 / denom)
    return w * (diff**2) * beta

def calculate_score(y_true, y_pred):
    return np.mean(calculate_twe(y_true, y_pred))

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Residual Analysis Refactored')
    parser.add_argument('--target-rul', type=str, default='HPT', choices=['HPC', 'HPT', 'WW'])
    parser.add_argument('--healthy-window', type=int, default=100)
    parser.add_argument('--pre-rolling-mean-window', type=int, default=100)
    parser.add_argument('--pre-rolling-mean-min-periods', type=int, default=10)
    parser.add_argument('--outlier-method', type=str, default='isoforest', choices=['zscore', 'iqr', 'isoforest'])
    parser.add_argument('--outlier-threshold', type=float, default=0.1)
    parser.add_argument('--post-window-size', type=int, default=50)
    parser.add_argument('--post-min-periods', type=int, default=5)
    parser.add_argument('--plots', nargs='+', default=['residuals_grid', 'health_index'], choices=['residuals_grid', 'health_index'])
    parser.add_argument('--figsize-w', type=int, default=20)
    parser.add_argument('--figsize-h', type=int, default=10)
    
    args = parser.parse_args()

    try:
        pipeline = ResidualAnalysisPipeline(args)
        pipeline.load_and_preprocess()
        pipeline.compute_residuals()
        pipeline.prepare_analysis_data()
        pipeline.compute_health_index()
        pipeline.generate_plots()
        print("\nPipeline execution finished successfully.")
    except Exception as e:
        print(f"\nCRITICAL ERROR in pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()