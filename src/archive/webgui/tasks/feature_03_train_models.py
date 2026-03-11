import sys
import os
import argparse
import pandas as pd
import numpy as np
import traceback
from tools import utils as u, config as cfg, algorithms as alg, plotting as up

# ==============================================================================
# DESIGN PATTERN: STRATEGY & ORCHESTRATOR
# ==============================================================================

class ModelStrategy:
    """Base interface for model training strategies."""
    def __init__(self, name, filename_prefix):
        self.name = name
        self.filename = f"{filename_prefix}_{name}.png"

    def train(self, df_data, df_meta, target, args):
        raise NotImplementedError

class LinearRegressionStrategy(ModelStrategy):
    def __init__(self, target): super().__init__("LR", target)
    def train(self, df_data, df_meta, target, args):
        return alg.train_linear_regression(df_data, df_meta, target=target, filename=self.filename, show_plot=False)

class RandomForestStrategy(ModelStrategy):
    def __init__(self, target): super().__init__("RF", target)
    def train(self, df_data, df_meta, target, args):
        return alg.train_random_forest(df_data, df_meta, target=target, 
                                       n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
                                       filename=self.filename, show_plot=False)

class XGBoostStrategy(ModelStrategy):
    def __init__(self, target): super().__init__("XGB", target)
    def train(self, df_data, df_meta, target, args):
        return alg.train_xgboost(df_data, df_meta, target=target,
                                 n_estimators=args.xgb_n_estimators, learning_rate=args.xgb_learning_rate, max_depth=args.xgb_max_depth,
                                 filename=self.filename, show_plot=False)

class TransformerStrategy(ModelStrategy):
    def __init__(self, target): super().__init__("Transformer", target)
    def train(self, df_data, df_meta, target, args):
        return alg.train_transformer(df_data, df_meta, target=target,
                                     epochs=args.trans_epochs, batch_size=args.trans_batch_size, lr=args.trans_learning_rate,
                                     filename=self.filename, show_plot=False)

class TrainingPipeline:
    """
    Orchestrates the model training process using Strategy pattern for different models.
    """
    def __init__(self, args):
        self.args = args
        self.target = args.target_training
        self.selected_models = set([m.strip().lower() for m in args.models.split(',') if m.strip()]) if args.models else None
        
        self.df_data = None
        self.df_meta = None
        self.all_results = {}
        
        # Strategy Registry
        self.strategies = {
            'linear': LinearRegressionStrategy(self.target),
            'rf': RandomForestStrategy(self.target),
            'xgb': XGBoostStrategy(self.target),
            'transformer': TransformerStrategy(self.target)
        }

    def run(self):
        try:
            self.load_data()
            self.train_selected_models()
            self.generate_dashboard()
            print("\nModel Training Task completed successfully.")
        except Exception as e:
            print(f"\nERROR in Training Pipeline: {e}")
            traceback.print_exc()

    def load_data(self):
        print(f"Loading feature data for {self.target}...")
        data_filename = f"training_feature_{self.target}"
        self.df_meta = pd.read_csv(u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"{data_filename}_metadata.csv"))
        df_d = pd.read_csv(u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"{data_filename}_data.csv"))
        
        # Combine
        meta_only_cols = [c for c in self.df_meta.columns if c not in df_d.columns]
        self.df_data = pd.concat([df_d, self.df_meta[meta_only_cols]], axis=1)

    def train_selected_models(self):
        for key, strategy in self.strategies.items():
            if self.selected_models is None or key in self.selected_models:
                print(f"\n>>> Executing Strategy: {strategy.name}...")
                try:
                    # Model Training returns (model, y_pred) or (model, y_pred, importances)
                    result = strategy.train(self.df_data.copy(), self.df_meta, self.target, self.args)
                    y_pred = result[1]
                    self._collect_results(strategy.name, y_pred)
                except Exception as e:
                    print(f"Failed to train {strategy.name}: {e}")

    def _collect_results(self, name, y_pred):
        # Extract true values for comparison
        target_map = {'HPC': 'Cycles_to_HPC_SV', 'HPT': 'Cycles_to_HPT_SV', 'WW': 'Cycles_to_WW'}
        target_col = target_map.get(self.target, 'RUL')
        
        test_df = self.df_data[self.df_data["ESN"] == 104]
        if len(test_df) == 0:
            split_idx = int(len(self.df_data) * 0.8)
            test_df = self.df_data.iloc[split_idx:]
            
        if target_col in test_df.columns:
            y_test = test_df[target_col].dropna().values
            # Handle sequence length differences (e.g. for Transformer)
            if len(y_test) != len(y_pred):
                y_test = y_test[-len(y_pred):]
            self.all_results[name] = (y_test, y_pred)

    def generate_dashboard(self):
        if self.all_results:
            print("\nGenerating comparative dashboard...")
            up.plot_training_dashboard(self.all_results, target=self.target, 
                                       filename=f"Training_Dashboard_{self.target}.png")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Model Training Pipeline")
    parser.add_argument("--models", type=str, help="Comma-separated: linear,rf,xgb,transformer")
    parser.add_argument("--target-training", type=str, default="HPC", choices=['HPC', 'HPT', 'WW'])
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=15)
    parser.add_argument("--xgb-n-estimators", type=int, default=600)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=5)
    parser.add_argument("--trans-epochs", type=int, default=400)
    parser.add_argument("--trans-batch-size", type=int, default=64)
    parser.add_argument("--trans-learning-rate", type=float, default=0.0005)
    parser.add_argument("--healthy-window", type=int, default=None)

    args = parser.parse_args()
    pipeline = TrainingPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()
