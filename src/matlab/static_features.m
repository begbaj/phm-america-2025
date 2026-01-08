clear; clc; close all;
SENSORS = ["Altitude", "Mach", "Pamb", "Pt2", "TAT", "WFuel", "VAFN", ...
    "VBV","Fan_Speed", "Core_Speed", "T25", "T3", "Ps3", "T45", "P25", "T5"];
[scriptPath, ~, ~] = fileparts(mfilename('fullpath'));
[parentPath, ~, ~] = fileparts(scriptPath);
[granParentPath, ~, ~] = fileparts(parentPath);
baseSavePath = fullfile(granParentPath, 'CONDITION_INDICATORS');

for i = {'hpc_cycle', 'hpt_cycle', 'ww_cycle'}
    EVENT = i{1};
    for SNAPSHOT = 1:8
        T = readtable("data/snapshot_tables/snapshot_" + string(SNAPSHOT) + ".csv");
        T = stc(T); % stc fa la pulizia dei dati (però è da rivedere come funzione)
        % ho scritto una nuova funzione perchè il modo in cui trattiamo i dati
        % adesso è differente da prima e ho pensato dovesse essere opportuno fare
        % la pulizia in modo diverso, ma non ne sono certo ;)
        % Creiamo un indice temporale cumulativo per ogni motore
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
        % 1. Converte EVENT in double (per le metriche)
        if iscategorical(MovingFeatures.(EVENT))
            MovingFeatures.(EVENT) = double(string(MovingFeatures.(EVENT)));
        end
        % 2. Converte esn in double (per evitare categorie vuote nel loop unique)
        if iscategorical(MovingFeatures.esn)
            MovingFeatures.esn = double(string(MovingFeatures.esn));
        end
        
        
        %% RANKING CON ONE-WAY ANOVA
        [featureTab,ranking,outputTable] = featureRanking(MovingFeatures,EVENT);
        ranking(:,"One-way ANOVA") = fillmissing(ranking(:,"One-way ANOVA"), 'constant', 0);
        ranking = sortrows(ranking,"One-way ANOVA","descend");
        top10Ranking = ranking(1:min(10, height(ranking)), :);
        
        
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
        m_table = monotonicity(ensembleData, 'TimeIndex');
        t_table = trendability(ensembleData, 'TimeIndex');
        p_table = prognosability(ensembleData, 'TimeIndex');
        m_vals = table2array(m_table);
        t_vals = table2array(t_table);
        p_vals = table2array(p_table);
        m_vals(isnan(m_vals)) = 0;
        t_vals(isnan(t_vals)) = 0;
        p_vals(isnan(p_vals)) = 0;
        % CI Index
        ci_index = (m_vals + t_vals + p_vals) / 3;
        featureNames = ensembleData{1}.Properties.VariableNames;
        featureNames = featureNames(~strcmp(featureNames, 'TimeIndex'));
        % Tabella riassuntiva
        ResultTable = table(featureNames', m_vals', t_vals', p_vals', ci_index', ...
            'VariableNames', {'Feature', 'Monotonicity', 'Trendability', 'Prognosability', 'CI_Index'});
        ResultTable = sortrows(ResultTable, 'CI_Index', 'descend');
        % Top 10
        top10Features = ResultTable(1:min(10, height(ResultTable)), :);
        
        
        %% PLOTTING COMPARATIVO
        fig = figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8], 'Name', 'Confronto Ranking Features');
        % --- PREPARAZIONE DATI PER UITABLE ---
        dataAnova = table2cell(top10Ranking);
        idxStringsAnova = cellfun(@isstring, dataAnova) | cellfun(@iscategorical, dataAnova);
        dataAnova(idxStringsAnova) = cellfun(@char, dataAnova(idxStringsAnova), 'UniformOutput', false);
        dataCI = table2cell(top10Features);
        idxStringsCI = cellfun(@isstring, dataCI) | cellfun(@iscategorical, dataCI);
        dataCI(idxStringsCI) = cellfun(@char, dataCI(idxStringsCI), 'UniformOutput', false);
        % --- LAYOUT ---
        t = tiledlayout(2, 2, 'TileSpacing', 'Loose', 'Padding', 'Compact');
        title(t, "Analisi Comparativa Ranking delle Features Snapshot " + string(SNAPSHOT), ...
            'FontSize', 16, 'FontWeight', 'bold');
        % 1. Grafico ANOVA
        nexttile(1);
        bar(top10Ranking.("One-way ANOVA"), 'FaceColor', [0.2 0.4 0.6]);
        set(gca, 'XTick', 1:height(top10Ranking), 'XTickLabel', top10Ranking.Features, 'TickLabelInterpreter', 'none');
        xtickangle(45);
        ylabel('F-Statistic (ANOVA)');
        title('Top 10: One-way ANOVA');
        grid on;
        % 2. Grafico CI
        nexttile(2);
        bar(top10Features.CI_Index, 'FaceColor', [0.6 0.2 0.2]);
        set(gca, 'XTick', 1:height(top10Features), 'XTickLabel', top10Features.Feature, 'TickLabelInterpreter', 'none');
        xtickangle(45);
        ylabel('CI Index Score');
        title('Top 10: Condition Indicators (M+T+P)/3');
        grid on;
        % 3. Tabella ANOVA
        nexttile(3);
        axis off;
        uit1 = uitable(fig, 'Data', dataAnova, ...
            'ColumnName', top10Ranking.Properties.VariableNames, ...
            'Units', 'Normalized', 'Position', [0.05, 0.05, 0.42, 0.38]);
        % 4. Tabella CI
        nexttile(4);
        axis off;
        uit2 = uitable(fig, 'Data', dataCI, ...
            'ColumnName', top10Features.Properties.VariableNames, ...
            'Units', 'Normalized', 'Position', [0.53, 0.05, 0.42, 0.38]);
        % SALVATAGGIO
        savePath = baseSavePath + "/" + string(EVENT) + "/"; 
        if ~exist(savePath, 'dir'), mkdir(savePath); end
        fileName = fullfile(savePath, string(EVENT) + "_Feature_Ranking_Comparison_Snapshot_" + string(SNAPSHOT) + ".png");
        
        frame = getframe(fig);
        img = frame2im(frame);
        imwrite(img, fileName);
        disp("Figura salvata correttamente in: " + fileName);
        close(fig);
    end
end



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
