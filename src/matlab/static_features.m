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
MF = groupsummary(T, {'esn', 'snap', EVENT, 'wid'}, ...
    {@mean, @std, @kurtosis, @skewness}, SENSORS);

G = findgroups(MF.esn, MF.snap, MF.hpcCycle);
MF.index = cell2mat(splitapply(@(x) {(0:numel(x)-1)'}, G, G));

G = findgroups(MF.esn, MF.hpcCycle);
MF.grouping = G;

%% MONOTONICITY, PROGNOSABILITY E TRENDABILITY

%function mtpplot(data)
%    figure;
%    subplot(3,1,1); monotonicity(data);   title('Monotonicity');
%    subplot(3,1,2); trendability(data);   title('Trendability');
%    subplot(3,1,3); prognosability(data); title('Prognosability');
%end

features = e(:, varfun(@isnumeric, e, 'OutputFormat', 'uniform')); 
predictors = removevars(features, {'hpcCycle', 'index'});
target = e.hpcCycle;

% 2. Train a Random Forest to see which features "explain" the cycle best
model = TreeBagger(50, predictors, target, 'Method', 'regression', 'OOBPredictorImportance', 'on');

% 3. Extract and Sort Importance
importance = model.OOBPredictorImportance;
[sortedImp, idx] = sort(importance, 'descend');
featureNames = predictors.Properties.VariableNames(idx);

% 4. Visualize the top 10 relevant features
figure;
bar(sortedImp(1:min(10, end)));
set(gca, 'XTick', 1:min(10, end), 'XTickLabel', featureNames(1:min(10, end)), 'TickLabelInterpreter', 'none');
title('Top Features Relevant to Condition Monitoring');
ylabel('Importance Score');

