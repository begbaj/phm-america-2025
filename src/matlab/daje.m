clear; clc; close all;

%% 1. CONFIGURATION
DATA_PATH = 'data/refactor/';
EVENTS_PATH = 'data/events'
ESNS = [101, 102, 103, 104];
SNAPSHOTS_TO_ANALYZE = [1, 2, 3, 4, 5, 6, 7, 8]; 
FEATURES_NAMES = {'mean', 'std', 'rms', 'kurtosis', 'skewness', 'shape_factor'};