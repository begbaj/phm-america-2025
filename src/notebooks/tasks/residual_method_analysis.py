import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import traceback
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from tools import utils as u, config as cfg, plotting as up, features as f, preprocessing as pp, algorithms as algo

# ==============================================================================
# LOGIC IMPLEMENTATION
# ==============================================================================

class ResidualAnalysisPipeline:
    def __init__(self, args):
        self.args = args
        self.output_folder = u.plot_path("residual_analysis", args.target_rul)
        self.operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 
                               'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
        self.degradation_vars = [] # Populated dynamically
        
        self.df_train = None
        self.df_test = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        
        self.model = None
        self.Y_pred = None
        self.residuals = None
        
        self.hi_hpt = None
        self.hi_hpc = None
        self.hpt_rul = None
        self.hpc_rul = None
        self.alpha_hpt = -2.75
        self.alpha_hpc = 9.0

    def load_and_preprocess(self):
        """Step 1: Load data, specific preprocessing and split."""
        print("Loading and Preprocessing data...")
        u.set_debug(self.args.debug_mode)
        
        # Load Data
        df = u.load_training()()
        
        # Define degradation vars based on available sensors minus operating vars
        self.degradation_vars = [s for s in u.SENSORS if s not in self.operating_vars]
        
        # 1. Preprocessing (Outliers & Missing)
        if self.args.remove_outliers:
             # The snippet uses pp.remove_outliers(df, u.SENSORS) - assuming default method or zscore
            df = pp.remove_outliers(df, u.SENSORS)
            
        if self.args.fill_missing:
             df = pp.missingfill(df).dropna()
        else:
             df = df.dropna()

        # 2. Split into Train (Others) and Test (Specific ESN)
        testing_esn = self.args.testing_esn
        print(f"Splitting data. Testing ESN: {testing_esn}")
        
        test_mask = df["ESN"] == testing_esn
        train_mask = df["ESN"].isin([x for x in [101, 102, 103, 104] if x != testing_esn])
        
        self.df_test = df[test_mask].reset_index(drop=True)
        self.df_train = df[train_mask].reset_index(drop=True)
        
        self.X_test = self.df_test[self.operating_vars]
        self.Y_test = self.df_test[self.degradation_vars]
        
        self.X_train = self.df_train[self.operating_vars]
        self.Y_train = self.df_train[self.degradation_vars]
        
        # Prepare RUL for testing ESN (for optimization later)
        self.hpt_rul = self.df_test["Cycles_to_HPT_SV"].reset_index(drop=True)
        self.hpc_rul = self.df_test["Cycles_to_HPC_SV"].reset_index(drop=True)

    def train_and_predict(self):
        """Step 2: Train Model and Predict."""
        print("Training models and predicting...")
        
        # MIMO Linear Regression (predicts all degradation vars from operating vars)
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.Y_train)
        
        # Predict on Test Data
        # Logic from snippet: Y_pred = model.predict(np.roll(X_test, model_i, axis=1)) 
        # With model_i=0, np.roll is identity.
        self.Y_pred = self.model.predict(self.X_test)
        
        # Convert to DataFrame for easier handling
        self.Y_pred = pd.DataFrame(self.Y_pred, columns=self.degradation_vars, index=self.Y_test.index)

    def compute_residuals(self):
        """Step 3: Calculate and Process Residuals."""
        print("Processing residuals...")
        
        # Raw Residuals
        res = self.Y_test - self.Y_pred
        
        # Post-processing: Remove Outliers
        res = pp.remove_outliers(res, u.SENSORS, threshold=self.args.res_outlier_threshold)
        res = res.dropna()
        
        # Rolling Median
        window = self.args.rolling_window
        step = window // self.args.rolling_step_div
        res = res.rolling(window, step).median()
        
        # Subtract Median of each column (Centering)
        # Note: res is likely a DataFrame. The snippet iterates by index, but vectorization is better.
        # "for i in range(0,7): m = res.iloc[:,i].median(); res.iloc[:,i] -= m"
        for col in res.columns:
            m = res[col].median()
            res[col] -= m
            
        self.residuals = res

    def optimize_and_calculate_hi(self):
        """Step 4: Optimize Alpha and Calculate HI."""
        print("Optimizing Alpha and calculating HI...")
        
        # Prepare Inputs
        # Ensure alignment (residuals might have dropped rows due to rolling/dropna)
        common_index = self.residuals.index
        
        # Align RULs
        hpt_rul_aligned = self.hpt_rul.loc[common_index]
        hpc_rul_aligned = self.hpc_rul.loc[common_index]
        
        # Get T3 and T45 residuals
        t3_res = self.residuals["Sensed_T3"]
        t45_res = self.residuals["Sensed_T45"]
        
        # Scaling RULs (Logic from snippet)
        hpt_target = hpt_rul_aligned / 100.0
        hpc_target = hpc_rul_aligned / 100.0 - 40.0
        
        # Optimization Objective
        def objective(alpha, t3, t45, y_true):
            y_pred = -alpha * t3 - t45
            mse = np.mean((y_pred - y_true)**2)
            return mse
            
        # Optimize HPT
        print(f"Optimizing HPT (Initial alpha: {self.alpha_hpt})...")
        res_hpt = minimize(
            objective, 
            x0=self.alpha_hpt, 
            args=(t3_res, t45_res, hpt_target), 
            method='BFGS'
        )
        self.alpha_hpt = res_hpt.x[0]
        print(f"Optimal Alpha HPT: {self.alpha_hpt}")
        
        # Optimize HPC
        print(f"Optimizing HPC (Initial alpha: {self.alpha_hpc})...")
        res_hpc = minimize(
            objective, 
            x0=self.alpha_hpc, 
            args=(t3_res, t45_res, hpc_target), 
            method='BFGS'
        )
        self.alpha_hpc = res_hpc.x[0]
        print(f"Optimal Alpha HPC: {self.alpha_hpc}")
        
        # Calculate Final HI
        def HI(t3, t45, alpha):
            return -alpha * t3 - t45
            
        self.hi_hpt = HI(t3_res, t45_res, self.alpha_hpt)
        self.hi_hpc = HI(t3_res, t45_res, self.alpha_hpc)
        
        # Store aligned RULs for plotting
        self.hpt_rul_plot = hpt_target
        self.hpc_rul_plot = hpc_target

    def generate_plots(self):
        """Step 5: Generate Plots."""
        print(f"Generating plots in: {self.output_folder}")
        
        # 1. Residuals Grid Plot (Korean Style)
        try:
            fig, axs = plt.subplots(2, 3, figsize=(15, 8))
            plot_vars = [v for v in self.degradation_vars if v in self.residuals.columns]
            
            # Use flattened iterator but handle if fewer vars than axes
            for i, ax in enumerate(axs.flat):
                if i < len(plot_vars):
                    var = plot_vars[i]
                    ax.plot(self.residuals[var], linewidth=1)
                    ax.set_title(var)
                    ax.set_ylabel("Residuals")
                    ax.set_xlabel(f"{var}_res")
                    ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off') # Hide unused subplots
            
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            fig.suptitle(f'Residuals for ESN {self.args.testing_esn}', fontsize=16)
            fig.savefig(os.path.join(self.output_folder, "residuals_korean_style.png"), bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting residuals: {e}")
            traceback.print_exc()

        # 2. HI vs RUL Plot
        try:
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            
            # HPT
            axs[0].plot(self.hi_hpt, color='tab:blue', label='Health Index (HPT)')
            axs[0].set_title(f'HPT Health Index (Alpha={self.alpha_hpt:.4f})')
            axs[0].grid(True, alpha=0.3)
            
            ax0_rul = axs[0].twinx()
            ax0_rul.plot(self.hpt_rul_plot, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale (Scaled)')
            
            lines0, labels0 = axs[0].get_legend_handles_labels()
            lines0r, labels0r = ax0_rul.get_legend_handles_labels()
            axs[0].legend(lines0 + lines0r, labels0 + labels0r, loc='upper right')

            # HPC
            axs[1].plot(self.hi_hpc, color='tab:green', label='Health Index (HPC)')
            axs[1].set_title(f'HPC Health Index (Alpha={self.alpha_hpc:.4f})')
            axs[1].grid(True, alpha=0.3)
            
            ax1_rul = axs[1].twinx()
            ax1_rul.plot(self.hpc_rul_plot, color='tab:orange', linewidth=2, linestyle='--', label='RUL Reale (Scaled)')
            
            lines1, labels1 = axs[1].get_legend_handles_labels()
            lines1r, labels1r = ax1_rul.get_legend_handles_labels()
            axs[1].legend(lines1 + lines1r, labels1 + labels1r, loc='upper right')

            fig.tight_layout()
            fig.savefig(os.path.join(self.output_folder, "HI_vs_RUL_optimized.png"), bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Error plotting HI: {e}")
            traceback.print_exc()

# ==============================================================================
# MAIN
# ==============================================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Residual Analysis Specific Implementation')
    
    # Standard args (kept for compatibility with orchestration if needed, but mostly logic is hardcoded)
    parser.add_argument('--target-rul', type=str, default='HPT', choices=['HPC', 'HPT', 'WW'])
    parser.add_argument('--debug-mode', type=str2bool, default=False)
    
    # Specific args for this logic
    parser.add_argument('--testing-esn', type=int, default=102, help='ESN to use for testing/calibration')
    parser.add_argument('--remove-outliers', type=str2bool, default=True)
    parser.add_argument('--fill-missing', type=str2bool, default=True)
    
    parser.add_argument('--res-outlier-threshold', type=float, default=3.0)
    parser.add_argument('--rolling-window', type=int, default=370)
    parser.add_argument('--rolling-step-div', type=int, default=5, help='Divisor for rolling step (window//div)')
    
    # Dummy args to satisfy potential external callers (GUI) if they pass them
    parser.add_argument('--healthy-window', type=int, default=0)
    parser.add_argument('--pre-rolling-mean-window', type=int, default=0)
    parser.add_argument('--pre-rolling-mean-min-periods', type=int, default=0)
    parser.add_argument('--outlier-method', type=str, default='isoforest')
    parser.add_argument('--outlier-threshold', type=float, default=0.1)
    parser.add_argument('--post-window-size', type=int, default=50)
    parser.add_argument('--post-min-periods', type=int, default=5)
    parser.add_argument('--plots', nargs='+', default=[])
    parser.add_argument('--figsize-w', type=int, default=20)
    parser.add_argument('--figsize-h', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0)
    
    args = parser.parse_args()

    try:
        pipeline = ResidualAnalysisPipeline(args)
        pipeline.load_and_preprocess()
        pipeline.train_and_predict()
        pipeline.compute_residuals()
        pipeline.optimize_and_calculate_hi()
        pipeline.generate_plots()
        print("\nPipeline execution finished successfully.")
    except Exception as e:
        print(f"\nCRITICAL ERROR in pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
