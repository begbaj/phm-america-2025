clear; clc; close all;
SENSORS = ["Altitude", "Mach", "Pamb", "Pt2", "TAT", "WFuel", "VAFN", ...
    "VBV","Fan_Speed", "Core_Speed", "T25", "T3", "Ps3", "T45", "P25", "T5"];
T = readtable('data/snapshot_tables/snapshot_1.csv');
T = stc(T); % stc fa la pulizia dei dati (però è da rivedere come funzione)
% ho scritto una nuova funzione perchè il modo in cui trattiamo i dati
% adesso è differente da prima e ho pensato dovesse essere opportuno fare
% la pulizia in modo diverso, ma non ne sono certo ;)

EVENT = 'hpc_cycle';

%% FEATURES FEATURES
% questo blocco serve a calcolare le features e inserirle nel workspace,
% non fa altro

FeatureTable = groupsummary(T, {'esn', EVENT},...
    {@mean, @std, @rms, @kurtosis, @skewness },...
    SENSORS);
FeatureTable = renamevars(FeatureTable, 'GroupCount', 'Cycle_Life');
% features calcolate a seconda del raggruppamento esn -> snap -> ciclo di 
% appartenenza del dato

%% MOVING FEATURES
K = 10; 
% creiamo delle "finestre" hardcoded nella tabella
T = sortrows(T, {'esn', EVENT});
T.wid = floor((0:height(T)-1)' / K);
tempFeatureTable = groupsummary(T, {'esn', EVENT, 'wid'}, ...
    {@mean, @std, @rms, @kurtosis, @skewness}, SENSORS);
MovingFeatures = fillFeatureNaN(tempFeatureTable);

% 1. Converti EVENT in double (per le metriche)
if iscategorical(MovingFeatures.(EVENT))
    MovingFeatures.(EVENT) = double(string(MovingFeatures.(EVENT)));
end

% 2. Converti esn in double (per evitare categorie vuote nel loop unique)
if iscategorical(MovingFeatures.esn)
    MovingFeatures.esn = double(string(MovingFeatures.esn));
end

%% MONOTONICITY, PROGNOSABILITY E TRENDABILITY
% 1. Organize data: Create a cell array where each cell is one engine's history
uniqueESNs = unique(MovingFeatures.esn);
ensembleData = {};

for i = 1:length(uniqueESNs)
    idx = MovingFeatures.esn == uniqueESNs(i);
    tempTable = MovingFeatures(idx, :);
    
    if height(tempTable) > 1
        % Ordiniamo prima per ciclo e poi per finestra (wid)
        tempTable = sortrows(tempTable, {EVENT, 'wid'});
        
        % CREAZIONE TEMPO STRETTAMENTE CRESCENTE
        % Creiamo un indice lineare (1, 2, 3, 4...) che rappresenta il tempo
        tempTable.TimeIndex = (1:height(tempTable))';
        
        % Rimuoviamo le variabili non necessarie
        % Teniamo TimeIndex come variabile temporale di riferimento
        temp = removevars(tempTable, {'esn', 'wid', EVENT}); 
        
        ensembleData{end+1} = temp;
    end
end

% 2. Evaluate Features (this will auto-plot a bar chart if no output is assigned)
figure;
subplot(3,1,1); monotonicity(ensembleData, 'TimeIndex');   title('Monotonicity');
subplot(3,1,2); trendability(ensembleData, 'TimeIndex');   title('Trendability');
subplot(3,1,3); prognosability(ensembleData, 'TimeIndex'); title('Prognosability');



%% HELPER FUNCTIONS
function T = fillFeatureNaN(T)
    % Identifica le colonne numeriche (escludendo esn e EVENT se necessario)
    % ma solitamente agire su tutta la tabella è sicuro
    numericCols = varfun(@isnumeric, T, 'OutputFormat', 'uniform');
    colNames = T.Properties.VariableNames(numericCols);
    for i = 1:length(colNames)
        col = colNames{i};
        nanIdx = isnan(T.(col));
        if any(nanIdx)
            T.(col)(nanIdx) = 0;
        end
    end
end
