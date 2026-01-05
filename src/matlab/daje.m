clear; clc; close all;

%% 1. CONFIGURATION
DATA_PATH = 'data/refactor/';
EVENTS_PATH = 'data/events/';
ESNS = [101, 102, 103, 104];
SNAPSHOTS_TO_ANALYZE = [1, 2, 3, 4, 5, 6, 7, 8]; 
FEATURES_NAMES = {'mean', 'std', 'kurtosis', 'skewness'};
SENSORS = {'Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_Pt2', ...
           'Sensed_TAT', 'Sensed_WFuel', 'Sensed_VAFN', 'Sensed_VBV', ...
           'Sensed_Fan_Speed', 'Sensed_Core_Speed', 'Sensed_T25', 'Sensed_T3', ...
           'Sensed_Ps3', 'Sensed_T45', 'Sensed_P25', 'Sensed_T5'};

%% 2.1 DATA LOADING
data_files = dir(fullfile(DATA_PATH, '*.csv'));
Data = struct();
for i = 1:length(data_files)
    filename = data_files(i).name;
    fullpath = fullfile(DATA_PATH, filename);
    varName = genvarname(strrep(filename, '.csv', ''));
    Data.(varName) = readtable(fullpath);
    fprintf('Dataset caricato: %s\n', varName);
end

%% 2.2 EVENTS LOADING
event_files = dir(fullfile(EVENTS_PATH, '*.csv'));
EventData = struct();
for i = 1:length(event_files)
    filename = event_files(i).name;
    fullpath = fullfile(EVENTS_PATH, filename);
    varName = genvarname(strrep(filename, '.csv', ''));
    EventData.(varName) = readtable(fullpath);
    fprintf('Eventi caricati: %s\n', varName);
end

%% 3. FEATURE CALCULATION



%% 4. METRICS