# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: phm-america-2025
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Modellazione RUL (Remaining Useful Life)
# In questo notebook implementiamo la stima della RUL per i tre eventi di manutenzione:
# - Water Wash (WW)
# - HPC Shop Visit
# - HPT Shop Visit
#
# Utilizziamo una strategia di validazione **Leave-One-Engine-Out (LOGO)**.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Aggiunge la directory corrente al path per importare tools
sys.path.append(os.getcwd())

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tools import config as cfg, utils as u

# %% [markdown]
# ## Configurazione e Caricamento Dati

# %%
TARGETS = {
    # 'to_next_ww_cycle': 'RUL Water Wash',
    # 'to_next_hpc_cycle': 'RUL HPC Shop Visit',
    'to_next_hpt_cycle': 'RUL HPT Shop Visit'
}

# Features da escludere (ID, target, metadata)
DROP_COLS = [
    'esn', 'snap', 'global_index', 'esn_index',
    'ww_cycle', 'hpc_cycle', 'hpt_cycle',
    'ww_cycle_index', 'hpc_cycle_index', 'hpt_cycle_index',
    'fault_ww_cycle', 'fault_hpc_cycle', 'fault_hpt_cycle',
    'to_next_ww_cycle', 'to_next_hpc_cycle', 'to_next_hpt_cycle'
] + u.SENSORS

RESULTS_DIR = os.path.join("img", "MODEL_RESULTS")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    """Carica il dataset aggregato."""
    path = os.path.join(cfg.DATA_BASE_PATH, "snapshot_tables", "feature_table.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File non trovato: {path}. Esegui PREPROCESSING.py prima.")
    
    df = pd.read_csv(path)
    print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    return df

def get_features(df):
    """Restituisce la lista delle feature e i target."""
    features = [c for c in df.columns if c not in DROP_COLS]
    # Rimuove eventuali colonne non numeriche se presenti per errore
    features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    return features

# %% [markdown]
# ## Logica di Training e Validazione (LOGO)

# %%
def train_evaluate_logo(df, features, target_col, model_type='rf'):
    """
    Esegue Leave-One-Group-Out (LOGO) su ESN.
    """
    esn_list = df['esn'].unique()
    results = []
    
    print(f"\n--- Training per Target: {target_col} ({model_type}) ---")
    
    for test_esn in esn_list:
        # Split Train/Test
        train_data = df[df['esn'] != test_esn]
        test_data = df[df['esn'] == test_esn]
        
        X_train = train_data[features]
        y_train = train_data[target_col]
        X_test = test_data[features]
        y_test = test_data[target_col]

        # Imputazione
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        # Modello
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        elif model_type == 'xgb':
            model = XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        else:
            model = LinearRegression()
            
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Salvataggio risultati per questo fold
        fold_res = pd.DataFrame({
            'esn': test_data['esn'],
            'esn_index': test_data['esn_index'],      # Indice progressivo per ESN
            'True_RUL': y_test,
            'Pred_RUL': preds
        })
        results.append(fold_res)
        
        # Metriche fold corrente
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"Test ESN {test_esn}: RMSE={rmse:.2f}, R2={r2:.4f}")

    # Concatenazione di tutti i risultati
    all_results = pd.concat(results, ignore_index=True)
    return all_results

# %% [markdown]
# ## Visualizzazione Risultati

# %%
def plot_predictions(df_results, target_name, model_name):
    """Scatter plot: True vs Predicted RUL"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_results, x='True_RUL', y='Pred_RUL', hue='esn', palette='viridis', alpha=0.6)
    
    # Linea ideale
    min_val = min(df_results['True_RUL'].min(), df_results['Pred_RUL'].min())
    max_val = max(df_results['True_RUL'].max(), df_results['Pred_RUL'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    rmse = np.sqrt(mean_squared_error(df_results['True_RUL'], df_results['Pred_RUL']))
    mae = mean_absolute_error(df_results['True_RUL'], df_results['Pred_RUL'])
    r2 = r2_score(df_results['True_RUL'], df_results['Pred_RUL'])
    
    plt.title(f"{model_name} - {target_name}\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.grid(True, alpha=0.3)
    
    filename = f"{target_name}_{model_name}_scatter.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.show()

def plot_rul_trajectory(df_results, target_name, model_name):
    """Plot temporale per ogni ESN: Ciclo vs RUL"""
    esns = df_results['esn'].unique()
    
    # Una figura per tutti i motori
    fig, axes = plt.subplots(len(esns), 1, figsize=(12, 4*len(esns)), sharex=False)
    if len(esns) == 1: axes = [axes]
    
    for ax, esn in zip(axes, esns):
        subset = df_results[df_results['esn'] == esn].sort_values('esn_index')
        
        ax.plot(subset['esn_index'], subset['True_RUL'], label='True RUL', color='black', lw=2)
        ax.plot(subset['esn_index'], subset['Pred_RUL'], label='Predicted RUL', color='orange', alpha=0.8)
        
        ax.set_title(f"ESN {esn} - {target_name} ({model_name})")
        ax.set_ylabel("RUL (Cycles)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Cycle Index")
    plt.tight_layout()
    
    filename = f"{target_name}_{model_name}_trajectory.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.show()

# %% [markdown]
# ## Esecuzione del Training

# %%
df = load_data()
features = get_features(df)

print(f"Features utilizzate ({len(features)}): {features}")

models = ['linear', 'rf', 'xgb']

for target_col, target_desc in TARGETS.items():
    for model_name in models:
        # Addestra e Valuta (Cross Validation)
        results = train_evaluate_logo(df, features, target_col, model_type=model_name)
        
        # Genera Grafici
        plot_predictions(results, target_col, model_name)
        plot_rul_trajectory(results, target_col, model_name)
        
        # Metrica finale globale
        global_rmse = np.sqrt(mean_squared_error(results['True_RUL'], results['Pred_RUL']))
        print(f"Global RMSE per {target_desc} ({model_name}): {global_rmse:.2f}")
