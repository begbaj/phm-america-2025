import sys
import os
import argparse
import pandas as pd
import traceback
from tools import utils as u, config as cfg, features as f

# ==============================================================================
# DESIGN PATTERN: PIPELINE ORCHESTRATOR
# ==============================================================================

class FeatureEngineeringPipeline:
    """
    Orchestrates the Feature Engineering flow: calculation and selection of best features.
    """
    def __init__(self, args):
        self.args = args
        self.target = args.target if args.target else 'HPC'
        self.window = args.pipeline_window if args.pipeline_window else 100
        self.step = args.pipeline_step if args.pipeline_step else 25
        self.stat_features = [s.strip() for s in args.statistical_features.split(',') if s.strip()] if args.statistical_features else ['mean', 'rms']
        
        # Mappings
        self.target_mapping = {
            'HPC': 'Cycles_to_HPC_SV',
            'HPT': 'Cycles_to_HPT_SV',
            'WW': 'Cycles_to_WW'
        }
        
        self.df_averaged = None
        self.df_features = None
        self.df_correlation = None
        self.best_features_meta = None

    def run(self):
        try:
            self.load_data()
            self.calculate_features()
            self.select_best_features()
            self.save_results()
            print(f"\nFeature Engineering for {self.target} completed successfully.")
        except Exception as e:
            print(f"\nERROR in Feature Engineering Pipeline: {e}")
            traceback.print_exc()

    def load_data(self):
        print(f"Loading aggregated data for {self.target}...")
        path_avg = u.pathfinder(cfg.DATA_BASE_PATH, "snapshot_tables", filename="averaged_final.csv")
        self.df_averaged = pd.read_csv(path_avg)

    def calculate_features(self):
        print(f"Calculating features (Window: {self.window}, Step: {self.step})...")
        full_target_col = self.target_mapping.get(self.target, f'to_next_{self.target.lower()}_cycle')
        performance_cols = f.get_all_performance_colnames()
        
        pipeline_map = {
            "HPC": f.pipeline_hpc,
            "HPT": f.pipeline_hpt,
            "WW": f.pipeline_ww
        }
        
        if self.target not in pipeline_map:
            raise ValueError(f"Invalid target: {self.target}")
            
        self.df_features, self.df_correlation = pipeline_map[self.target](
            self.df_averaged, [], performance_cols, self.stat_features, 
            window=self.window, step=self.step, 
            stat_groupby=["ESN"], stat_sortby=["ESN", "esn_index"], 
            target=full_target_col
        )

    def select_best_features(self):
        print("Selecting best features based on correlation...")
        self.best_features_meta = self.df_correlation.sort_values(
            by='tot_val', key=abs, ascending=False
        ).head(10)
        print("Top Features Selected:")
        print(self.best_features_meta[["feature", "tot_val"]])

    def save_results(self):
        print("Saving results...")
        # 1. Save Metadata
        path_meta = u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"training_feature_{self.target}_metadata.csv")
        self.best_features_meta.to_csv(path_meta, index=False)
        
        # 2. Save Data
        path_data = u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"training_feature_{self.target}_data.csv")
        
        # Columns to keep
        essential_cols = ['ESN', 'Cycles_Since_New', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV', 'Cycles_to_WW']
        cols_to_save = [f for f in self.best_features_meta["feature"].tolist() if f in self.df_features.columns]
        
        for col in essential_cols:
            if col in self.df_features.columns and col not in cols_to_save:
                cols_to_save.append(col)
                
        self.df_features[cols_to_save].to_csv(path_data, index=False)
        print(f"Metadata: {path_meta}")
        print(f"Feature Matrix: {path_data}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Feature Engineering Task')
    parser.add_argument('--target', choices=['HPC', 'HPT', 'WW'], default='HPC')
    parser.add_argument('--statistical-features', type=str, help='Comma-separated: mean,rms,std,min,max')
    parser.add_argument('--pipeline-window', type=int, default=100)
    parser.add_argument('--pipeline-step', type=int, default=25)
    
    args = parser.parse_args()
    pipeline = FeatureEngineeringPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()