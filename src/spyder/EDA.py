# %%
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import iplot
from importlib import reload
import sys
import subprocess
import os
import json
import plotly
from plotview import show_plotly


# %% 
# Load Data
train = None
with open("../../Data/PHM2025_training_data/training_data.csv", "r") as f:
    train = pd.read_csv(f)
# %%
# Add a cycle index 'i' for each ESN
train['i'] = train.groupby('ESN').cumcount()

# %%
# List Sensors
sensors = [
    "Sensed_Altitude",
    "Sensed_Mach",
    "Sensed_Pamb",
    "Sensed_Pt2",
    "Sensed_TAT",
    "Sensed_WFuel",
    "Sensed_VAFN",
    "Sensed_VBV",
    "Sensed_Fan_Speed",
    "Sensed_Core_Speed",
    "Sensed_T25",
    "Sensed_T3",
    "Sensed_Ps3",
    "Sensed_T45",
    "Sensed_P25",
    "Sensed_T5",
]
# %%
# define plot function

def plot(df, sensor):
  fig = px.line(
        df,
        x=df['i'],
        y=sensor,
        color='ESN',
        title=f'Andamento del Sensore {sensor} Suddiviso per ESN',
        labels={'x':'Indice del Dato / Tempo (Sequenza)', 'y':f'Valore di {sensor}'},
        height=1200,
        line_group='ESN',
  )

  fig.update_xaxes(rangeslider_visible=True) # Aggiunge uno slider in basso per navigare nel tempo
  return fig
# %%
# Plot 
fig = plot(train[train["ESN"] == 101], sensors[5])
show_plotly(fig)

# %%
cm = train.corr()

plt.figure(figsize=(15,15))
sns.heatmap(cm, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
# Assuming your DataFrame is 'df' and the column is 'sensor_value'
wws_points = train[train['Cumulative_WWs'] > train['Cumulative_WWs'].shift(1)]
hpc_points = train[train['Cumulative_HPC_SVs'] > train['Cumulative_HPC_SVs'].shift(1)]
hpt_points = train[train['Cumulative_HPT_SVs'] > train['Cumulative_HPT_SVs'].shift(1)]

# %%

