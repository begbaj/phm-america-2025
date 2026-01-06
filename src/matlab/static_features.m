clear; clc; close all;
SENSORS = ["Altitude", "Mach", "Pamb", "Pt2", "TAT", "WFuel", "VAFN", ...
    "VBV","Fan_Speed", "Core_Speed", "T25", "T3", "Ps3", "T45", "P25", "T5"];
T = readtable('data/snapshot_tables/reformatted.csv');
T = stc(T); % stc fa la pulizia dei dati (però è da rivedere come funzione)
% ho scritto una nuova funzione perchè il modo in cui trattiamo i dati
% adesso è differente da prima e ho pensato dovesse essere opportuno fare
% la pulizia in modo diverso, ma non ne sono certo ;)

EVENT = 'hpcCycle';

%% FEATURES FEATURES
% questo blocco serve a calcolare le features e inserirle nel workspace,
% non fa altro

FeatureTable = groupsummary(T, {'esn', 'snap', EVENT},...
    {@mean, @std, @rms, @kurtosis, @skewness },...
    SENSORS);
FeatureTable = renamevars(FeatureTable, 'GroupCount', 'Cycle_Life');
% features calcolate a seconda del raggruppamento esn -> snap -> ciclo di 
% appartenenza del dato

%% MOVING FEATURES
K = 10; 
% creiamo delle "finestre" hardcoded nella tabella
T = sortrows(T, {'esn', 'snap', EVENT});
T.wid = floor((0:height(T)-1)' / K);
MovingFeatures = groupsummary(T, {'esn', 'snap', EVENT, 'wid'}, ...
    {@mean, @std, @kurtosis, @skewness}, SENSORS);

%% MONOTONICITY, PROGNOSABILITY E TRENDABILITY
% 1. Organize data: Create a cell array where each cell is one engine's history
uniqueESNs = unique(FeatureTable.esn);
ensembleData = cell(length(uniqueESNs), 1);

for i = 1:length(uniqueESNs)
    % Extract data for one engine and sort by time/cycle
    idx = FeatureTable.esn == uniqueESNs(i);
    ensembleData{i} = sortrows(FeatureTable(idx, :), EVENT); 
end

% 2. Evaluate Features (this will auto-plot a bar chart if no output is assigned)
figure;
subplot(3,1,1); monotonicity(ensembleData);   title('Monotonicity');
subplot(3,1,2); trendability(ensembleData);   title('Trendability');
subplot(3,1,3); prognosability(ensembleData); title('Prognosability');
