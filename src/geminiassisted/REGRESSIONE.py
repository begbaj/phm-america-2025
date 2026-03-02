# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Regressione

# %%
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %load_ext autoreload
# %autoreload 2
from tools import (
    utils as u,
    config as cfg,
    algorithms as alg,
    plotting as up,
    features as f,
)

# %%
data_filename = "training_feature_HPT_26011956"

dfm = pd.read_csv(
    u.pathfinder(
        cfg.DATA_BASE_PATH, "features", filename=f"{data_filename}_metadata.csv"
    )
)  # METADATI
dfd = pd.read_csv(
    u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"{data_filename}_data.csv")
)  # DATI
# dft = pd.read_csv(u.pathfinder(cfg.DATA_BASE_PATH, "PHM2025_validation_data", filename=f"val_0.csv")) # TEST

# Uniamo dati e metadati per avere l'ESN e il target insieme
df_full = pd.concat(
    [dfd, dfm[["esn", "cycle", "fault_hpt_cycle", "to_next_hpt_cycle"]]], axis=1
)

# Definiamo le feature (tutte le colonne di dfd)
feature_cols = dfd.columns.tolist()

# Eseguiamo il LOGO con XGBoost e 5 componenti PCA
final_preds = u.train_evaluate_logo_pca(
    df_full,
    features=feature_cols,
    target_col="to_next_hpt_cycle",
    n_components=5,
    model_type="xgb",
)

# Calcolo metriche
mae = mean_absolute_error(final_preds["True_RUL"], final_preds["Pred_RUL"])
rmse = np.sqrt(mean_squared_error(final_preds["True_RUL"], final_preds["Pred_RUL"]))
r2 = r2_score(final_preds["True_RUL"], final_preds["Pred_RUL"])

print(f"--- Performance Globali (LOGO CV) ---")
print(f"MAE:  {mae:.2f} cicli")
print(f"RMSE: {rmse:.2f} cicli")
print(f"R^2:  {r2:.4f}")

plt.figure(figsize=(16, 6))
x_axis = np.arange(len(final_preds))

plt.plot(
    x_axis,
    final_preds["True_RUL"].values,
    label="Target Reale",
    color="black",
    alpha=0.7,
    linewidth=1,
)
plt.plot(
    x_axis,
    final_preds["Pred_RUL"].values,
    label="Predizione LOGO",
    color="red",
    linestyle="--",
    alpha=0.6,
)

# Linee verticali per separare i motori (ESN)
cambio_motore = final_preds[final_preds["esn"] != final_preds["esn"].shift()].index
for idx in cambio_motore:
    plt.axvline(x=idx, color="gray", linestyle=":", alpha=0.4)

plt.title("Risultati LOGO Cross-Validation: Tutti i Motori Concatenati")
plt.xlabel("Campioni sequenziali (Unità dopo Unità)")
plt.ylabel("HPT Cycle State")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(final_preds["True_RUL"], final_preds["Pred_RUL"], alpha=0.3, s=10, c="blue")

# Linea di riferimento a 45 gradi
lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
plt.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="Ideale")

plt.title("Parity Plot: Reale vs Predetto")
plt.xlabel("Valore Reale (Target)")
plt.ylabel("Valore Predetto (Model)")
plt.legend()
plt.grid(True)
plt.show()


# %%
# PCA + Regressione Lineare su tutto il dataset (senza separare i motori)


target_col = "to_next_hpt_cycle"

X_train, X_test, y_train, y_test = train_test_split(
    dfd,
    dfm[[target_col]],  # <--- Doppie quadre qui
    test_size=0.2,
    shuffle=False,  # Mantiene l'ordine temporale (fondamentale per PHM)
)
# --- 2. Scaling e PCA ---
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

pca = PCA(n_components=2)  # Con 2 componenti puoi visualizzare bene il piano
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)


# %%

# --- 3. Regressione ---
classifier = LinearRegression()
classifier.fit(X_train_pca, y_train[target_col])

plt.figure(figsize=(15, 6))

# Usiamo il range dell'indice invece di y_test['cycle']
x_axis = np.arange(len(y_test))

# Valori Reali (Target)
plt.plot(
    x_axis,
    y_test[target_col].values,
    label="Reale (Tutti i motori)",
    color="black",
    alpha=0.7,
)

# Predizione PCA
plt.plot(
    x_axis,
    y_pred,
    label="Predizione PCA (Tutti i motori)",
    color="red",
    linestyle="--",
    alpha=0.6,
)

plt.title("Performance PCA Regression su tutto il Test Set (Sequenziale)")
plt.xlabel("Campioni totali nel Test Set")
plt.ylabel("Stato Guasto (HPT Cycle)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%

# 1. Utilizziamo i dati scalati ma NON trasformati dalla PCA
# X_train_sc e X_test_sc contengono tutte le feature originali (standardizzate)
mlr_raw = LinearRegression()
mlr_raw.fit(X_train_sc, y_train[target_col])

# 2. Predizione
y_pred_raw = mlr_raw.predict(X_test_sc)

# 3. Metriche
mae_raw = mean_absolute_error(y_test[target_col], y_pred_raw)
r2_raw = r2_score(y_test[target_col], y_pred_raw)

print(f"--- Risultati MLR Senza PCA (Dati Raw) ---")
print(f"Mean Absolute Error: {mae_raw:.2f} cicli")
print(f"R^2 Score: {r2_raw:.4f}")

# 4. Plot Sequenziale
plt.figure(figsize=(15, 6))
x_axis = np.arange(len(y_test))

plt.plot(
    x_axis, y_test[target_col].values, label="Reale (Target)", color="black", alpha=0.8
)
plt.plot(
    x_axis,
    y_pred_raw,
    label="Predizione MLR (Senza PCA)",
    color="green",
    linestyle="--",
    alpha=0.7,
)

plt.title("Multiple Linear Regression su Dati Originali (Senza PCA)")
plt.xlabel("Indice temporale (Test Set)")
plt.ylabel("Stato Guasto (HPT Cycle)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%

# 1. Inizializzazione del modello
# n_estimators: numero di alberi (100 è un buon default)
# max_depth: profondità degli alberi (evita l'overfitting)
# random_state: per rendere i risultati riproducibili
rf_model = RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)

# 2. Addestramento sui dati scalati (non serve la PCA)
rf_model.fit(X_train_sc, y_train[target_col])

# 3. Predizione
y_pred_rf = rf_model.predict(X_test_sc)

# 4. Metriche
mae_rf = mean_absolute_error(y_test[target_col], y_pred_rf)
r2_rf = r2_score(y_test[target_col], y_pred_rf)

print(f"--- Risultati Random Forest ---")
print(f"Mean Absolute Error: {mae_rf:.2f} cicli")
print(f"R^2 Score: {r2_rf:.4f}")

plt.figure(figsize=(15, 6))
x_axis = np.arange(len(y_test))

plt.plot(
    x_axis, y_test[target_col].values, label="Reale (Target)", color="black", alpha=0.8
)
plt.plot(
    x_axis,
    y_pred_rf,
    label="Predizione Random Forest",
    color="blue",
    linestyle="--",
    alpha=0.7,
)

plt.title("Random Forest Regressor: Fault Detection Globale")
plt.xlabel("Indice temporale (Test Set)")
plt.ylabel("Stato Guasto (HPT Cycle)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Creiamo un grafico delle 10 feature più importanti
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]  # Prendi le ultime 10 (le più grandi)

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Top 10 Sensori)")
plt.barh(range(len(indices)), importances[indices], color="b", align="center")
plt.yticks(range(len(indices)), [dfd.columns[i] for i in indices])
plt.xlabel("Importanza Relativa")
plt.show()

# %%

# 1. Preparazione delle componenti PCA (già calcolate nei passi precedenti)
# Usiamo X_train_pca (che ha n_components=10 o 2) e X_test_pca

# 2. Inizializzazione del Random Forest
# Nota: Su poche componenti (es. 2 o 5), il RF è velocissimo
rf_pca = RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)

# 3. Addestramento sulle componenti artificiali
rf_pca.fit(X_train_pca, y_train[target_col])

# 4. Predizione
y_pred_rf_pca = rf_pca.predict(X_test_pca)

# 5. Metriche
mae_rf_pca = mean_absolute_error(y_test[target_col], y_pred_rf_pca)
r2_rf_pca = r2_score(y_test[target_col], y_pred_rf_pca)

print(f"--- Risultati Random Forest su componenti PCA ---")
print(f"Mean Absolute Error: {mae_rf_pca:.2f} cicli")
print(f"R^2 Score: {r2_rf_pca:.4f}")
plt.figure(figsize=(15, 6))
x_axis = np.arange(len(y_test))

# Target reale
plt.plot(
    x_axis, y_test[target_col].values, label="Reale (Target)", color="black", alpha=0.8
)

# Predizione RF + PCA
plt.plot(
    x_axis,
    y_pred_rf_pca,
    label="Predizione RF su PCA",
    color="magenta",
    linestyle="--",
    alpha=0.7,
)

plt.title(f"Fault Detection: Random Forest su {pca.n_components} Componenti PCA")
plt.xlabel("Indice temporale (Test Set)")
plt.ylabel("Stato Guasto (HPT Cycle)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%

# 1. Configurazione del modello XGBoost
# objective='reg:squarederror' indica che vogliamo fare una regressione
# n_estimators: numero di iterazioni di boosting
# learning_rate (eta): quanto "velocemente" il modello impara dai residui
# max_depth: profondità degli alberi
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

# 2. Addestramento sulle componenti PCA
xgb_model.fit(X_train_pca, y_train[target_col])

# 3. Predizione
y_pred_xgb = xgb_model.predict(X_test_pca)

# 4. Metriche di valutazione
mae_xgb = mean_absolute_error(y_test[target_col], y_pred_xgb)
r2_xgb = r2_score(y_test[target_col], y_pred_xgb)

print(f"--- Risultati XGBoost su PCA ---")
print(f"Mean Absolute Error: {mae_xgb:.2f} cicli")
print(f"R^2 Score: {r2_xgb:.4f}")

plt.figure(figsize=(15, 6))
x_axis = np.arange(len(y_test))

# Target reale
plt.plot(
    x_axis, y_test[target_col].values, label="Reale (Target)", color="black", alpha=0.8
)

# Predizione XGBoost + PCA
plt.plot(
    x_axis,
    y_pred_xgb,
    label="Predizione XGBoost su PCA",
    color="orange",
    linestyle="-",
    alpha=0.7,
)

plt.title(f"Fault Detection: XGBoost su {pca.n_components} Componenti PCA")
plt.xlabel("Indice temporale (Test Set)")
plt.ylabel("Stato Guasto (HPT Cycle)")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()
