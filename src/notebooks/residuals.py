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
#     display_name: phm-america-2025 (3.10.19)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
# %load_ext autoreload
# %autoreload 2
from tools import utils as u, config as cfg, plotting as up, preprocessing as pp

# %%
def train_models(df) -> dict[int, dict[str,LinearRegression]]:
    X_train = df[operating_vars]
    Y_train = df[degradation_vars]
    models = {}
    for i in range(0,8):
        X_temp = pd.DataFrame(np.roll(X_train, i, axis=1))
        models[i] = {}
        models[i]["model"] = train_model(X_temp, Y_train)
    return models

def train_model(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def s_pred(s_o, model):
    return model.predict(s_o)

def residual(s_d, s_o, model):
    return s_d - s_pred(s_o, model)

def w(y_p, y, a):
    diff = y - y_p
    num = np.where(diff >= 0, 2.0, 1.0)
    if isinstance(y_p, pd.DataFrame) or isinstance(y_p, pd.Series):
        y_p = y_p.values
    return num / (1 + a * y_p)

def TWE(y_p, y, a, b):
    if isinstance(y_p, pd.DataFrame): y_p = y_p.values
    weight = w(y_p, y, a)
    squared_error = (y - y_p) ** 2
    return weight * squared_error * b



# %%
from pyparsing import line


df = u.load_training()()
operating_vars = ['Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_TAT', 'Sensed_VAFN', 'Sensed_VBV', 'Sensed_Fan_Speed', 'Sensed_Pt2']
degradation_vars = [s for s in u.SENSORS if s not in operating_vars]

df = pp.remove_outliers(df, u.SENSORS)
df = pp.missingfill(df).dropna()
test_data = df[df["ESN"] == 104].reset_index()

X_test = test_data[operating_vars]
Y_test = test_data[degradation_vars]
models = train_models(df[df["ESN"].isin([101, 102, 103])])

model_i = 0
model = models[model_i]['model']
Y_pred = model.predict(np.roll(X_test, model_i, axis=1))

twes = TWE(Y_pred, Y_test, 10, 3)

res = Y_test - Y_pred
res = pp.remove_outliers(res, u.SENSORS, threshold=3)
res = res.dropna()
window = 370
step = window//5
res = res.rolling(window, step).median()

for i in range(0,7):
    m = res.iloc[:,i].median()
    res.iloc[:,i] -= m

fig, axs = plt.subplots(2,3, figsize=(15,8))
for i, ax in enumerate(axs.flat):
    if isinstance(ax, plt.Axes):
        ax.plot(res.iloc[:,i], linewidth=1)
        ax.set_title(degradation_vars[i])
        ax.set_ylabel("Residuals")
        ax.set_xlabel(f"{res.iloc[:,i].index.name}_res")
        ax.grid()

fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.show()

# col = 1
# plt.figure()
# plt.plot(test_data["Cycles_Since_New"],Y_pred[:,col], linewidth=0.5, label="Predicted")
# plt.plot(test_data["Cycles_Since_New"],Y_test.iloc[:,col], linewidth=0.5, label="Observed")
# plt.show()

# a = 0.01
# t3_res = np.array(df["T3_res"])
# t45_res = np.array(df["T45_res"])
# hi_target = -a * t3_res - t45_res
# print(df)
# plt.figure()
# plt.plot(df["Cycles_since_new"], hi_target)
# plt.show()
