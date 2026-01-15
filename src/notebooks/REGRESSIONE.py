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
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %load_ext autoreload
# %autoreload 2
from tools import utils as u, config as cfg, algorithms as alg, plotting as up, features as f

# %%
data_filename = "training_feature_HPT_26011917"

dfm = pd.read_csv(u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"{data_filename}_metadata.csv")) # METADATI
dfd = pd.read_csv(u.pathfinder(cfg.DATA_BASE_PATH, "features", filename=f"{data_filename}_data.csv")) # DATI
# dft = pd.read_csv(u.pathfinder(cfg.DATA_BASE_PATH, "PHM2025_validation_data", filename=f"val_0.csv")) # TARGETS

X_train, X_test, y_train, y_test = train_test_split(
    dfd, 
    dfm[dfm['fault_hpt_cycle', "cycle", "esn"]], 
    test_size=0.2, 
    shuffle=False
)

# sc = StandardScaler() # permette di normalizzare i valori delle features così che abbiano media 0 e deviazione standard 1
# X_train = sc.fit_transform(X_train, y_train)
# X_test = sc.transform(X_test)

# pca = PCA(n_components=10)
# X_train = pca.fit_transform(X_train, y_train)
# X_test = pca.transform(X_test)

# explained_variance = pca.explained_variance_ratio_
# print(f"{explained_variance}")



# %%

classifier = LinearRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

plt.figure(figsize=(8,6)) 
plt.scatter(X_test, y_test, color='blue', label='Data Points') 
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line') 
plt.title('Linear Regression on Random Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# %%
