clear; clc; close all;

TRAINING_DATA = 'data/PHM2025_training_data/training_data.csv';

ESNS = [101, 102, 103, 104];
SNAPSHOTS = [1, 2, 3, 4, 5, 6, 7, 8]; 
FEATURES_NAMES = {'mean', 'std', 'rms', 'kurtosis', 'skewness', 'shape_factor'};

SENSORS = {'Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_Pt2', ...
           'Sensed_TAT', 'Sensed_WFuel', 'Sensed_VAFN', 'Sensed_VBV', ...
           'Sensed_Fan_Speed', 'Sensed_Core_Speed', 'Sensed_T25', 'Sensed_T3', ...
           'Sensed_Ps3', 'Sensed_T45', 'Sensed_P25', 'Sensed_T5'};

MAINTAINANCE_TYPE = ["Cumulative_WWs", "Cumulative_HPT_SVs", "Cumulative_HPC_SVs"];
WINDOW_SIZE = 10;
STEP_SIZE = 3;