clear; clc; close all;
SENSORS = ["Altitude", "Mach", "Pamb", "Pt2", "TAT", "WFuel", "VAFN", ...
    "VBV","Fan_Speed", "Core_Speed", "T25", "T3", "Ps3", "T45", "P25", "T5"];
[scriptPath, ~, ~] = fileparts(mfilename('fullpath'));
[parentPath, ~, ~] = fileparts(scriptPath);
[granParentPath, ~, ~] = fileparts(parentPath);
baseSavePath = fullfile(granParentPath, 'Data/AVERAGED');

for i = {'hpc_cycle', 'hpt_cycle', 'ww_cycle'}
    EVENT = i{1};
    T = readtable("data/snapshot_tables/averaged_final.csv");
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
    FeatureTable = groupsummary(T, {'esn', char(EVENT), char(EVENT_INDEX), char(EVENT_RUL), char(EVENT_FAULT)},...
        {@mean, @std, @rms, @kurtosis, @skewness },...
        SENSORS);
    FeatureTable = renamevars(FeatureTable, 'GroupCount', 'Cycle_Life');
    % features calcolate a seconda del raggruppamento esn -> snap -> ciclo di 
    % appartenenza del dato
        

    %% MOVING FEATURES
    K = 10; 
    T = sortrows(T, {'esn', EVENT, char(EVENT_INDEX)});
    % Reset WID per ogni motore
    T.wid = zeros(height(T), 1);
    uEsn = unique(T.esn);
    for idxE = 1:numel(uEsn)
        sel = T.esn == uEsn(idxE);
        T.wid(sel) = floor((0:sum(sel)-1)' / K);
    end
    % Specifica delle funzioni statistiche per i sensori
    tempFeatureTable = groupsummary(T, {'esn', char(EVENT), 'wid'}, ...
        {@mean, @std, @rms, @kurtosis, @skewness, @max}, [SENSORS, EVENT_FAULT]);
    % Rinominazione colonna fault
    oldFaultName = "fun6_" + EVENT_FAULT;
    if ismember(oldFaultName, tempFeatureTable.Properties.VariableNames)
        tempFeatureTable = renamevars(tempFeatureTable, oldFaultName, EVENT_FAULT);
    end
    % Rinominazione delle colonne dei sensori
    funNames = {'mean', 'std', 'rms', 'kurtosis', 'skewness'};
    for fn = 1:numel(funNames)
        oldNames = "fun" + string(fn) + "_" + SENSORS;
        newNames = string(funNames{fn}) + "_" + SENSORS;
        existIdx = ismember(oldNames, tempFeatureTable.Properties.VariableNames);
        tempFeatureTable = renamevars(tempFeatureTable, oldNames(existIdx), newNames(existIdx));
    end
    % drop di tutte le colonne inutili
    allVars = tempFeatureTable.Properties.VariableNames;
    varsToDrop = allVars(startsWith(allVars, "fun"));
    if ~isempty(varsToDrop)
        tempFeatureTable = removevars(tempFeatureTable, varsToDrop);
    end
    MovingFeatures = fillFeatureNaN(tempFeatureTable);
    if iscategorical(MovingFeatures.esn)
        MovingFeatures.esn = double(string(MovingFeatures.esn));
    end


    % %% PLOTTING FEATURES
    % for target_esn = [101, 102, 103, 104]
    %     for target_sensor = SENSORS
    %         plotData = MovingFeatures(MovingFeatures.esn == target_esn, :);
    %         colors = [0.00, 0.45, 0.74;   % Blu (Mean)
    %                   0.85, 0.33, 0.10;   % Arancio (Std)
    %                   0.47, 0.67, 0.19;   % Verde (Kurtosis)
    %                   0.64, 0.08, 0.18];  % Amaranto (Skewness)
    %         if ~isempty(plotData)
    %             hFig = figure('Units', 'normalized', 'Position', [0.1 0.1 0.5 0.8], 'Color', 'w');
    %             t = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'loose');
    %             txtTitle = "Feature Statistiche: " + target_sensor;
    %             txtSub   = "ESN: " + string(target_esn);
    %             title(t, {['\fontsize{14}', char(txtTitle)], ['\fontsize{10}\color[rgb]{0.4 0.4 0.4}', char(txtSub)]}, 'FontWeight', 'bold');
    %             x = plotData.('wid');
    %             % --- 1. TREND PRINCIPALE (Mean con Area di Varianza) ---
    %             ax1 = nexttile;
    %             hold on;
    %             mu = plotData.("mean_" + target_sensor);
    %             sigma = plotData.("std_" + target_sensor);
    %             % Ombreggiatura per la deviazione standard
    %             fill([x; flipud(x)], [mu-sigma; flipud(mu+sigma)], colors(1,:), ...
    %                 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    %             plot(x, mu, '-o', 'Color', colors(1,:), 'LineWidth', 1.8, ...
    %                 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', 'Moving Mean');
    %             grid on; ax1.GridAlpha = 0.4;
    %             ylabel('Mean', 'FontWeight', 'bold');
    %             legend('Location', 'best', 'Box', 'off');
    %             % --- 2. DISPERSIONE E RMS ---
    %             ax2 = nexttile;
    %             hold on;
    %             plot(x, sigma, '-d', 'Color', colors(2,:), 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Std Dev');
    %             plot(x, plotData.("rms_" + target_sensor), '--', 'Color', [0.4 0.4 0.4], 'DisplayName', 'RMS');
    %             grid on; ax2.GridAlpha = 0.4;
    %             ylabel('Std/RMS', 'FontWeight', 'bold');
    %             legend('Location', 'best', 'Box', 'off');
    %             % --- 3. FORMA DELLA DISTRIBUZIONE (Kurtosis & Skewness) ---
    %             ax3 = nexttile;
    %             yyaxis left
    %             s = stem(x, plotData.("kurtosis_" + target_sensor), 'filled', 'DisplayName', 'Kurtosis');
    %             s.Color = colors(3,:); s.MarkerSize = 4;
    %             ylabel('Kurtosis', 'FontWeight', 'bold');
    %             ax3.YColor = colors(3,:);
    %             yyaxis right
    %             plot(x, plotData.("skewness_" + target_sensor), '-p', 'Color', colors(4,:), ...
    %                 'MarkerFaceColor', colors(4,:), 'MarkerSize', 6, 'DisplayName', 'Skewness');
    %             ylabel('Skewness', 'FontWeight', 'bold');
    %             ax3.YColor = colors(4,:);
    %             grid on; 
    %             xlabel(['Ciclo (', char(EVENT_INDEX), ')'], 'FontWeight', 'bold');
    %             % Link degli assi per zoom sincronizzato
    %             linkaxes([ax1, ax2, ax3], 'x');
    %             drawnow;
    %             % SALVATAGGIO
    %             savePath = baseSavePath + "/FEATURE_STATISTICHE/" + string(EVENT) + "/"; 
    %             if ~exist(savePath, 'dir'), mkdir(savePath); end
    %             fileName = fullfile(savePath, string(EVENT) + "_Feature_Statistiche_ESN_" + target_esn + "_Sensore_" + target_sensor + ".png");
    %             frame = getframe(hFig);
    %             img = frame2im(frame);
    %             imwrite(img, fileName);
    %             disp("Figura salvata correttamente in: " + fileName);
    %             close(hFig);
    %         else
    %             warning('Nessun dato trovato per il motore %d', target_esn);
    %         end
    %     end
    % end
        
        
    %% RANKING CON ONE-WAY ANOVA
    [featureTab,ranking,outputTable] = featureRanking(MovingFeatures,EVENT);
    ranking(:,"One-way ANOVA") = fillmissing(ranking(:,"One-way ANOVA"), 'constant', 0);
    ranking = sortrows(ranking,"One-way ANOVA","descend");
    top10Ranking = ranking(1:min(10, height(ranking)), :);
        
        
    %% MONOTONICITY, PROGNOSABILITY E TRENDABILITY
    uniqueESNs = unique(MovingFeatures.esn);
    ensembleData = {};
    for i = 1:length(uniqueESNs)
        idx = MovingFeatures.esn == uniqueESNs(i);
        tempTable = MovingFeatures(idx, :);
        if height(tempTable) > 1
            % Ordiniamo prima per ciclo e poi per finestra (wid)
            tempTable = sortrows(tempTable, {EVENT, 'wid'});
            % CREAZIONE TEMPO STRETTAMENTE CRESCENTE
            tempTable.TimeIndex = (1:height(tempTable))';
            % Rimuoviamo le variabili non necessarie
            % Teniamo TimeIndex come variabile temporale di riferimento
            temp = removevars(tempTable, ["esn", "wid", string(EVENT), "GroupCount", string(EVENT_FAULT)]); 
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
    dataAnova = table2cell(top10Ranking);
    idxStringsAnova = cellfun(@isstring, dataAnova) | cellfun(@iscategorical, dataAnova);
    dataAnova(idxStringsAnova) = cellfun(@char, dataAnova(idxStringsAnova), 'UniformOutput', false);
    dataCI = table2cell(top10Features);
    idxStringsCI = cellfun(@isstring, dataCI) | cellfun(@iscategorical, dataCI);
    dataCI(idxStringsCI) = cellfun(@char, dataCI(idxStringsCI), 'UniformOutput', false);
    % --- LAYOUT ---
    t = tiledlayout(2, 2, 'TileSpacing', 'Loose', 'Padding', 'Compact');
    title(t, "Analisi Comparativa Ranking delle Features", ...
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
    % --- 3. Tabella ANOVA ---
    nexttile(3);
    axis off; 
    uit1 = uitable(fig, 'Data', dataAnova, ...
        'ColumnName', top10Ranking.Properties.VariableNames, ...
        'Units', 'Normalized', ...
        'Position', [0.05, 0.05, 0.43, 0.38], ... 
        'FontSize', 9, ...
        'ColumnWidth', {140, 80}); 
    % --- 4. Tabella CI ---
    nexttile(4);
    axis off;
    uit2 = uitable(fig, 'Data', dataCI, ...
        'ColumnName', top10Features.Properties.VariableNames, ...
        'Units', 'Normalized', ...
        'Position', [0.52, 0.05, 0.43, 0.38], ... 
        'FontSize', 9, ...
        'ColumnWidth', {140, 60, 60, 60, 60});
    % SALVATAGGIO
    savePath = baseSavePath + "/CONDITION_INDICATORS/" + string(EVENT) + "/"; 
    if ~exist(savePath, 'dir'), mkdir(savePath); end
    fileName = fullfile(savePath, string(EVENT) + "_Feature_Ranking_Comparison_total.png");
    frame = getframe(fig);
    img = frame2im(frame);
    imwrite(img, fileName);
    disp("Figura salvata correttamente in: " + fileName);
    close(fig);
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
