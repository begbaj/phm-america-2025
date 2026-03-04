
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from tools import utils as u, config as cfg, preprocessing as pp

# --- Helper Functions ---
def train_model(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def train_models(df, operating_vars, degradation_vars) -> dict[int, dict[str,LinearRegression]]:
    X_train = df[operating_vars]
    Y_train = df[degradation_vars]
    models = {}
    for i in range(0,8):
        # Note: simplistic simulation of the rolling logic from original script
        X_temp = pd.DataFrame(np.roll(X_train, i, axis=1))
        models[i] = {}
        models[i]["model"] = train_model(X_temp, Y_train)
    return models

def HIE(params, vars):
    # Health Index Estimation
    return vars.dot(-np.array(params))

def get_rolling_slope_intercept(series, window):
    slopes = []
    intercepts = []
    # Optimization: Using numpy for faster rolling window polyfit might be better, 
    # but sticking to loop for fidelity to original logic or simplicity
    vals = series.values
    x = np.arange(window)
    
    for i in range(len(series)):
        if i < window:
            slopes.append(0)
            intercepts.append(0)
        else:
            y = vals[i-window:i]
            # Fit polynomial degree 1 (line) -> returns [slope, intercept]
            # Handling singular matrix issues if y is constant
            if np.all(y == y[0]):
                 slopes.append(0)
                 intercepts.append(y[0])
            else:
                poly = np.polyfit(x, y, 1)
                slopes.append(poly[0])
                intercepts.append(poly[1])
    return np.array(slopes), np.array(intercepts)

class ErrorCorrector:
    def __init__(self, window_size=800):
        self.mapper = LinearRegression()
        self.corrector = lgb.LGBMRegressor(n_estimators=20000, learning_rate=0.001, verbose=-1)
        self.window_size = window_size
        
    def fit(self, hi_series, rul_target):
        # 1. Fit Base Mapper (HI -> RUL)
        X_base = hi_series.values.reshape(-1, 1)
        self.mapper.fit(X_base, rul_target)
        pred_base = self.mapper.predict(X_base)
        
        # 2. Calculate Residuals (True Gap)
        gap_true = rul_target - pred_base
        
        # 3. Prepare Features for LightGBM
        slope, intercept = get_rolling_slope_intercept(hi_series, self.window_size)
        
        X_lgbm = pd.DataFrame({
            'HI': hi_series.values,
            'Slope': slope,
            'Intercept': intercept
        })
        
        # Filter for training (remove initial 0 slopes if any, or just where slope != 0 to match logic)
        mask = X_lgbm['Slope'] != 0
        
        if mask.sum() > 0:
            X_train = X_lgbm[mask]
            y_train = gap_true[mask]
            self.corrector.fit(X_train, y_train)
        else:
            print("Warning: Not enough data to train LightGBM corrector.")
            
    def predict(self, hi_series):
        # 1. Base Prediction
        X_base = hi_series.values.reshape(-1, 1)
        pred_base = self.mapper.predict(X_base)
        
        # 2. Features for Correction
        slope, intercept = get_rolling_slope_intercept(hi_series, self.window_size)
        X_lgbm = pd.DataFrame({
            'HI': hi_series.values,
            'Slope': slope,
            'Intercept': intercept
        })
        
        # 3. Predict Gap
        pred_gap = self.corrector.predict(X_lgbm)
        
        # 4. Corrected RUL
        return pred_base + pred_gap

# --- Configuration & Coefficients ---
coefs_hpt = (850.7167357577274, -15.072134101084401, -24.57278901752793, 102.1439502693563, -119.99243468799672, 11.396869998772502, -917.4479008347435, -24.27692144851878)
coefs_hpc = (363.09492555179804, 2.0186697716587463, 13.743038965485157, 11.855221907006188, 69.72211072964963, 0.42476560122623985, -989.8022299435106, 28.63715373059266)
coefs_ww  = (850.7167357577274, -15.072134101084401, -24.57278901752793, 102.1439502693563, -119.99243468799672, 11.396869998772502, -917.4479008347435, -24.27692144851878)

operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
degradation_vars = [s for s in u.SENSORS if s not in operating_vars]
model_i = 0

# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing Pipeline...")
    
    # 1. TRAIN PHASE (Models + Correctors)
    print("Loading training data (ESN 102) to setup Error Correctors...")
    df_train = u.load_training()()
    df_train = pp.remove_outliers(df_train, u.SENSORS)
    df_train = pp.missingfill(df_train).dropna()
    
    # Train Base Residual Models
    testing_esn = 102
    train_subset = df_train[df_train["ESN"].isin([x for x in [101,102,103,104] if x != testing_esn])]
    models = train_models(train_subset, operating_vars, degradation_vars)
    model = models[model_i]['model']
    
    # Process ESN 102 for LightGBM Training
    esn102 = df_train[df_train["ESN"] == testing_esn].reset_index(drop=True)
    X_102 = esn102[operating_vars]
    Y_102 = esn102[degradation_vars]
    
    # Get Residuals for 102
    Y_pred_102 = model.predict(np.roll(X_102, model_i, axis=1))
    res_102 = Y_102 - Y_pred_102
    
    # Smooth residuals (logic from notebook)
    window = 100
    res_102 = pp.remove_outliers(res_102, u.SENSORS, threshold=3)
    esn102[degradation_vars] = res_102
    
    # We need to maintain RUL alignment after dropping NaNs
    temp_102 = esn102.dropna()
    res_102_smoothed = temp_102[degradation_vars].rolling(window, 1).median()
    
    # Median Norm
    for i in range(0, 7):
        m = res_102_smoothed.iloc[:, i].median()
        res_102_smoothed.iloc[:, i] -= m
        
    final_train_data = res_102_smoothed.dropna()
    # Re-align RUL
    final_indices = final_train_data.index
    hpt_rul_train = temp_102.loc[final_indices, "Cycles_to_HPT_SV"]
    hpc_rul_train = temp_102.loc[final_indices, "Cycles_to_HPC_SV"]
    ww_rul_train  = temp_102.loc[final_indices, "Cycles_to_WW"]
    
    print("Training Error Correctors (LightGBM)...")
    corrector_hpt = ErrorCorrector(window_size=800)
    hi_hpt_train = HIE(coefs_hpt, final_train_data)
    corrector_hpt.fit(hi_hpt_train, hpt_rul_train)
    
    corrector_hpc = ErrorCorrector(window_size=800)
    hi_hpc_train = HIE(coefs_hpc, final_train_data)
    corrector_hpc.fit(hi_hpc_train, hpc_rul_train)
    
    corrector_ww = ErrorCorrector(window_size=1000) # Note: 1000 window in notebook for WW
    hi_ww_train = HIE(coefs_ww, final_train_data)
    corrector_ww.fit(hi_ww_train, ww_rul_train)
    
    # 2. TESTING PHASE
    print("Starting Testing Phase on load_testing()...")
    df_test = u.load_testing()()
    
    results = []
    
    for eng in df_test["ESN"].unique():
        print(f"Processing Engine {eng}...")
        test_data = df_test[df_test["ESN"] == eng].reset_index(drop=True)
        
        if test_data.empty: continue
            
        X_test = test_data[operating_vars]
        Y_test = test_data[degradation_vars]
        
        # Predict Residuals
        Y_pred = model.predict(X_test) # Assuming model_i=0 so no roll needed, or roll=0
        
        res = Y_test - Y_pred
        
        # Apply same smoothing/preprocessing as training
        # Note: In strict pipeline, we shouldn't use future information (rolling), 
        # but replicating the notebook logic which uses rolling on the whole series.
        test_data[degradation_vars] = res
        res_processed = test_data.dropna() # Drop initial if any
        
        # Note: If test set is shorter than window, this will be empty.
        # The notebook uses huge windows (100, 800, 1000). 
        # If load_testing() data is streaming or short, this will fail or return empty.
        # Assuming the test files are complete flights/histories.
        
        # Smooth
        res_rolling = res_processed[degradation_vars].rolling(window, 1).median()
        # Norm
        for i in range(0, 7):
            if res_rolling.shape[0] > 0:
                m = res_rolling.iloc[:, i].median()
                res_rolling.iloc[:, i] -= m
        
        res_final = res_rolling.dropna()
        
        if res_final.empty:
            print(f"  -> Skipping {eng}: Not enough data for rolling window.")
            continue
            
        # Calculate HIs
        hi_hpt = HIE(coefs_hpt, res_final)
        hi_hpc = HIE(coefs_hpc, res_final)
        hi_ww  = HIE(coefs_ww, res_final)
        
        # Predict Corrected RULs
        pred_rul_hpt = corrector_hpt.predict(hi_hpt)
        pred_rul_hpc = corrector_hpc.predict(hi_hpc)
        pred_rul_ww  = corrector_ww.predict(hi_ww)
        
        # Store/Display last predicted RUL (most current)
        print(f"  -> Final Predicted RULs | HPT: {pred_rul_hpt.iloc[-1]:.2f} | HPC: {pred_rul_hpc.iloc[-1]:.2f} | WW: {pred_rul_ww.iloc[-1]:.2f}")
        
        results.append({
            "ESN": eng,
            "RUL_HPT": pred_rul_hpt.iloc[-1],
            "RUL_HPC": pred_rul_hpc.iloc[-1],
            "RUL_WW": pred_rul_ww.iloc[-1]
        })

    print("\n--- Final Results Summary ---")
    print(pd.DataFrame(results))

