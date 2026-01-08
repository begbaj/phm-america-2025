clear; clc; close all;
SENSORS = ["Altitude", "Mach", "Pamb", "Pt2", "TAT", "WFuel", "VAFN", ...
    "VBV","Fan_Speed", "Core_Speed", "T25", "T3", "Ps3", "T45", "P25", "T5"];
T = readtable('data/snapshot_tables/snapshot_1.csv');
T = stc(T); % stc fa la pulizia dei dati (però è da rivedere come funzione)
% ho scritto una nuova funzione perchè il modo in cui trattiamo i dati
% adesso è differente da prima e ho pensato dovesse essere opportuno fare
% la pulizia in modo diverso, ma non ne sono certo ;)
% Creiamo un indice temporale cumulativo per ogni motore
EVENT = 'hpc_cycle';
EVENT_INDEX = EVENT + "_index" ;
EVENT_RUL = "to_next_" + EVENT;
EVENT_FAULT = "fault_" + EVENT;


%% FEATURES
% questo blocco serve a calcolare le features e inserirle nel workspace,
% non fa altro
FeatureTable = groupsummary(T, {'esn', EVENT, char(EVENT_INDEX), char(EVENT_RUL), char(EVENT_FAULT)},...
    {@rms, @std, @kurtosis, @skewness },...
    SENSORS);
FeatureTable = renamevars(FeatureTable, 'GroupCount', 'Cycle_Life');
% features calcolate a seconda del raggruppamento esn -> snap -> ciclo di 
% appartenenza del dato

%% MOVING FEATURES
K = 10; 
% creiamo delle "finestre" hardcoded nella tabella
T = sortrows(T, {'esn', EVENT, char(EVENT_INDEX), char(EVENT_RUL), char(EVENT_FAULT)});
T.wid = floor((0:height(T)-1)' / K);
tempFeatureTable = groupsummary(T, {'esn', EVENT, char(EVENT_INDEX), char(EVENT_RUL), char(EVENT_FAULT), 'wid'}, ...
    {@rms, @std, @kurtosis, @skewness}, SENSORS);
MovingFeatures = fillFeatureNaN(tempFeatureTable);
% 1. Converti EVENT in double (per le metriche)
if iscategorical(MovingFeatures.(EVENT))
    MovingFeatures.(EVENT) = double(string(MovingFeatures.(EVENT)));
end
% 2. Converti esn in double (per evitare categorie vuote nel loop unique)
if iscategorical(MovingFeatures.esn)
    MovingFeatures.esn = double(string(MovingFeatures.esn));
end


%% RANKING CON ONE-WAY ANOVA
[featureTab,ranking,outputTable] = featureRanking(MovingFeatures,EVENT);
ranking(:,"One-way ANOVA") = fillmissing(ranking(:,"One-way ANOVA"), 'constant', 0);
ranking = sortrows(ranking,"One-way ANOVA","descend");


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
        temp = removevars(tempTable, {'esn', 'wid', EVENT, char(EVENT_INDEX), ...
            char(EVENT_RUL), char(EVENT_FAULT), 'GroupCount'}); 
        
        ensembleData{end+1} = temp;
    end
end

%% CALCOLO CONDITION INDICATOR E TOP 10 FEATURES
% 1. Calcola i valori (Assicurati di assegnare l'output per avere i dati numerici)
m_table = monotonicity(ensembleData, 'TimeIndex');
t_table = trendability(ensembleData, 'TimeIndex');
p_table = prognosability(ensembleData, 'TimeIndex');

% 2. Converti le tabelle di output in vettori numerici (double)
% MATLAB restituisce i valori come variabili all'interno di una tabella
m_vals = table2array(m_table);
t_vals = table2array(t_table);
p_vals = table2array(p_table);

% 3. Ora puoi usare isnan perché m_vals, t_vals e p_vals sono double
m_vals(isnan(m_vals)) = 0;
t_vals(isnan(t_vals)) = 0;
p_vals(isnan(p_vals)) = 0;

% 4. Calcolo del CI Index (Media semplice)
ci_index = (m_vals + t_vals + p_vals) / 3;

% 4. Recuperiamo i nomi delle feature (escludendo TimeIndex)
featureNames = ensembleData{1}.Properties.VariableNames;
featureNames = featureNames(~strcmp(featureNames, 'TimeIndex'));

% 5. Creiamo una tabella riassuntiva per facilitare l'ordinamento
ResultTable = table(featureNames', m_vals', t_vals', p_vals', ci_index', ...
    'VariableNames', {'Feature', 'Monotonicity', 'Trendability', 'Prognosability', 'CI_Index'});

% 6. Ordiniamo per CI_Index decrescente
ResultTable = sortrows(ResultTable, 'CI_Index', 'descend');

% 7. Estraiamo e stampiamo le Top 10
Top10Features = ResultTable(1:min(10, height(ResultTable)), :);
disp('--- TOP 10 FEATURES BY CONDITION INDICATOR INDEX ---');
disp(Top10Features);

% 8. Visualizzazione Grafica delle Top 10
figure('Name', 'Top 10 Condition Indicators');
bar(Top10Features.CI_Index);
set(gca, 'XTick', 1:10, 'XTickLabel', Top10Features.Feature, 'TickLabelInterpreter', 'none');
xtickangle(45);
ylabel('CI Index Score (Average of M, T, P)');
title('Top 10 Features per Manutenzione Predittiva');
grid on;



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
