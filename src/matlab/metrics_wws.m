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
WINDOW_SIZE = 10; 
STEP_SIZE = 1;
TIPO_EVENTO = 'wws';


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
Results = struct();
for s = 1:length(SNAPSHOTS_TO_ANALYZE)
    snap_id = SNAPSHOTS_TO_ANALYZE(s);
    snap_field = sprintf('Snap%d', snap_id);
    snap_name = sprintf('training_data_snapshot_%d', snap_id);    
    if ~isfield(Data, snap_name), continue; end    
    curr_data = Data.(snap_name);    
    % Inizializziamo una cell array per accumulare i dati (più veloce di concatenare tabelle)
    all_rows = {}; 
    for e = 1:length(ESNS)
        esn = ESNS(e);
        event_field = sprintf('%s_%d', TIPO_EVENTO, esn);     
        % Trova limiti e stop
        if isfield(EventData, event_field) && ~isempty(EventData.(event_field))
            limit = EventData.(event_field).Index(end);
            intervalli_stop = EventData.(event_field).Index(1:end-1);
        else
            limit = height(curr_data); 
            intervalli_stop = [];
        end
        for sns = 1:length(SENSORS)
            sensor_name = SENSORS{sns};
            sensor_col = sprintf('%s_%d', sensor_name, esn);          
            if ismember(sensor_col, curr_data.Properties.VariableNames)
                actual_limit = min(limit, height(curr_data));
                signal = [curr_data.Cycles_Since_New(1:actual_limit), ...
                          curr_data.(sensor_col)(1:actual_limit)];              
                % Calcolo feature
                feat = moving_features(signal, intervalli_stop, WINDOW_SIZE, STEP_SIZE);              
                % Creazione dei dati per questo sensore
                n = length(feat.mean);
                esn_col = repmat(esn, n, 1);
                sensor_label = repmat({sensor_name}, n, 1);                
                % Accodiamo i dati alla lista generale
                new_data = table(esn_col, sensor_label, feat.Cycles', feat.Event_ID', ...
                                 feat.Event_Index', feat.mean', feat.std', ...
                                 feat.kurtosis', feat.skewness', ...
                                 'VariableNames', {'ESN', 'Sensor', 'Cycles', 'Event_Num', 'Data_Idx', 'Mean', 'Std', 'Kurtosis', 'Skewness'});
                all_rows{end+1} = new_data;
            end
        end
    end
    % "Vertcat" di tutte le tabelle in una sola (molto efficiente)
    Results.(snap_field) = vertcat(all_rows{:});
end

%% 4. METRICS
MetricsResults = struct();
for s = 1:length(SNAPSHOTS_TO_ANALYZE)
    snap_id = SNAPSHOTS_TO_ANALYZE(s);
    snap_field = sprintf('Snap%d', snap_id);
    if ~isfield(Results, snap_field), continue; end  
    T = Results.(snap_field);
    metrics_list = {};    
    % Iteriamo per ogni sensore e ogni motore
    unique_sensors = unique(T.Sensor);
    unique_esns = unique(T.ESN);    
    for sns = 1:length(unique_sensors)
        s_name = unique_sensors{sns};
        for e_idx = 1:length(unique_esns)
            esn_val = unique_esns(e_idx);            
            % Filtriamo i dati per lo specifico sensore e motore
            subT = T(strcmp(T.Sensor, s_name) & T.ESN == esn_val, :);
            if isempty(subT), continue; end            
            % Ogni "Event_Num" è un profilo Run-to-Failure
            unique_events = unique(subT.Event_Num);            
            for ev = 1:length(unique_events)
                ev_id = unique_events(ev);
                data_r2f = subT(subT.Event_Num == ev_id, :);               
                % Estraiamo le feature (usiamo la Media come indicatore principale, 
                % ma puoi estenderlo a Std, Kurt, etc.)
                % Calcoliamo le metriche sulla colonna 'Mean'
                val = data_r2f.Mean;
                n = length(val);                
                if n < 2
                    m = 0; p = 0; tr = 0;
                else
                    % 1. MONOTONICITY
                    % Differenza tra incrementi positivi e negativi
                    m = abs(sum(diff(val) > 0) - sum(diff(val) < 0)) / (n - 1);                    
                    % 2. PROGNOSABILITY
                    % Rapporto tra il range del degrado e la dispersione al failure
                    % Nota: Qui calcolata sul singolo evento è semplificata
                    p = exp(-std(val) / abs(val(end) - val(1)));                    
                    % 3. TRENDABILITY
                    % Correlazione di Spearman rispetto al tempo (Cycles)
                    tr = abs(corr(val, data_r2f.Cycles, 'Type', 'Spearman'));
                end                
                % Gestione NaN
                m(isnan(m)) = 0; p(isnan(p)) = 0; tr(isnan(tr)) = 0;                
                % Score Totale (Media semplice o pesata)
                total_score = (m + p + tr) / 3;                
                % Salvataggio riga
                metrics_list{end+1} = table(esn_val, {s_name}, ev_id, m, p, tr, total_score, ...
                    'VariableNames', {'ESN', 'Sensor', 'Event_Num', 'Monotonicity', 'Prognosability', 'Trendability', 'TotalScore'});
            end
        end
    end   
    MetricsResults.(snap_field) = vertcat(metrics_list{:});
    fprintf('Metriche calcolate per Snapshot %d\n', snap_id);
end


%% 5. BEST FEATURES IN TERMS OF MONOT. TREND. AND PROGN.
BestFeatures = struct();
for s = 1:length(SNAPSHOTS_TO_ANALYZE)
    snap_id = SNAPSHOTS_TO_ANALYZE(s);
    snap_field = sprintf('Snap%d', snap_id);
    if ~isfield(MetricsResults, snap_field), continue; end
    % Recuperiamo la tabella delle metriche dello snapshot corrente
    T_metrics = MetricsResults.(snap_field);    
    % 1. Media per Motore (ESN) all'interno di ogni Sensore
    % (Collassiamo gli eventi: un valore per ogni motore-sensore)
    T_esn_avg = groupsummary(T_metrics, {'Sensor', 'ESN'}, 'mean', ...
        {'Monotonicity', 'Prognosability', 'Trendability', 'TotalScore'});  
    % 2. Media Globale per Sensore (Media di tutti i motori)
    T_sensor_final = groupsummary(T_esn_avg, 'Sensor', 'mean', ...
        {'mean_Monotonicity', 'mean_Prognosability', 'mean_Trendability', 'mean_TotalScore'}); 
    % Rinominiamo le colonne per leggibilità
    T_sensor_final.Properties.VariableNames = ...
        {'Sensor', 'GroupCount', 'Monotonicity', 'Prognosability', 'Trendability', 'TotalScore'};  
    % 3. Ordiniamo per TotalScore decrescente
    T_sensor_final = sortrows(T_sensor_final, 'TotalScore', 'descend'); 
    % Salvataggio nella struct dei risultati
    BestFeatures.(snap_field) = T_sensor_final;   
    % --- STAMPA TOP 5 ---
    fprintf('\n--- TOP 5 SENSORS FOR SNAPSHOT %d ---\n', snap_id);
    if height(T_sensor_final) >= 8
        disp(T_sensor_final(1:8, {'Sensor', 'TotalScore', 'Monotonicity', 'Prognosability', 'Trendability'}));
    else
        disp(T_sensor_final(:, {'Sensor', 'TotalScore', 'Monotonicity', 'Prognosability', 'Trendability'}));
    end
    fprintf('--------------------------------------\n');
end


%% 6 IDENTIFICAZIONE DEI CONDITION INDICATORS
all_top_sensors = {};
% Recuperiamo i nomi dei sensori nelle Top 5 di ogni snapshot
snap_names = fieldnames(BestFeatures);
for i = 1:length(snap_names)
    T_snap = BestFeatures.(snap_names{i});
    % Prendiamo i primi 5 (o meno se lo snapshot ne ha meno)
    num_to_extract = min(5, height(T_snap));
    top_5_names = T_snap.Sensor(1:num_to_extract);
    % Accumuliamo i nomi in una lista unica
    all_top_sensors = [all_top_sensors; top_5_names];
end
% Creiamo una tabella di frequenza
[unique_sensors, ~, idx] = unique(all_top_sensors);
counts = accumarray(idx, 1);
FrequencyTable = table(unique_sensors, counts, ...
    'VariableNames', {'Sensor', 'Top5_Appearance_Count'});
% Ordiniamo per frequenza decrescente
FrequencyTable = sortrows(FrequencyTable, 'Top5_Appearance_Count', 'descend');
% Visualizzazione finale
fprintf('\n======================================================\n');
fprintf('   SENSORS CONSISTENCY ANALYSIS (Appearance in Top 5)\n');
fprintf('======================================================\n');
disp(FrequencyTable);
fprintf('======================================================\n');
% Identifichiamo i sensori "Leader" (quelli che appaiono in quasi tutti gli snapshot)
threshold = length(SNAPSHOTS_TO_ANALYZE) * 0.7; % Appare almeno nel 70% degli snapshot
leader_sensors = FrequencyTable.Sensor(FrequencyTable.Top5_Appearance_Count >= threshold);
fprintf('Sensori consigliati per il modello predittivo:\n');
disp(leader_sensors);

%% POSSIBILE 7: ANALISI SUI 5 SENSORI PIU FREQUENTI IN GENERALE? O DIVISI PER SNAPSHOT?

%% 7 SEPARAZIONE DEI FAULTY CYCLES PER I CI PER OGNI SNAPSHOT
FailurePoints = struct();   
DegradationData = struct(); 
for s = 1:length(SNAPSHOTS_TO_ANALYZE)
    snap_id = SNAPSHOTS_TO_ANALYZE(s);
    snap_field = sprintf('Snap%d', snap_id);
    % Verifichiamo che esistano sia i dati che le metriche
    if ~isfield(Results, snap_field) || ~isfield(BestFeatures, snap_field), continue; end   
    % 1. Identifichiamo i Top 3 sensori specifici per QUESTO snapshot
    T_best = BestFeatures.(snap_field);
    current_top_3 = T_best.Sensor(1:min(3, height(T_best)));    
    % 2. Filtriamo i dati originali di Results per questi 3 sensori
    T_full = Results.(snap_field);
    T_filtered = T_full(ismember(T_full.Sensor, current_top_3), :);    
    all_failures = {};
    all_degradation = {};    
    unique_esns = unique(T_filtered.ESN);    
    for e_idx = 1:length(unique_esns)
        esn_val = unique_esns(e_idx);
        T_esn = T_filtered(T_filtered.ESN == esn_val, :);        
        unique_evs = unique(T_esn.Event_Num);       
        for ev = 1:length(unique_evs)
            ev_id = unique_evs(ev);
            T_ev = T_esn(T_esn.Event_Num == ev_id, :);           
            % Per ogni sensore dei top 3, separiamo l'ultimo ciclo dal resto
            for sns_idx = 1:length(current_top_3)
                s_name = current_top_3{sns_idx};
                T_ev_sns = T_ev(strcmp(T_ev.Sensor, s_name), :);               
                if isempty(T_ev_sns), continue; end          
                % Identifichiamo l'ultimo ciclo (Failure Point)
                max_cycle = max(T_ev_sns.Cycles);                
                % Estrazione righe
                is_failure = (T_ev_sns.Cycles == max_cycle);                
                all_failures{end+1} = T_ev_sns(is_failure, :);
                all_degradation{end+1} = T_ev_sns(~is_failure, :);
            end
        end
    end    
    % Consolidamento dei dati dello snapshot
    FailurePoints.(snap_field) = vertcat(all_failures{:});
    DegradationData.(snap_field) = vertcat(all_degradation{:});    
    fprintf('Snapshot %d (Sensori: %s, %s, %s):\n', ...
        snap_id, current_top_3{1}, current_top_3{2}, current_top_3{3});
    fprintf('   -> Isolate %d righe di Failure e %d di Degrado.\n', ...
        height(FailurePoints.(snap_field)), height(DegradationData.(snap_field)));
end


%% 8. BOXPLOT DELLA MEDIA DI DEGRADATION DATA E FAULTY DATA
for s = 1:length(SNAPSHOTS_TO_ANALYZE)
    snap_id = SNAPSHOTS_TO_ANALYZE(s);
    snap_field = sprintf('Snap%d', snap_id);
    if ~isfield(DegradationData, snap_field), continue; end
    % Recuperiamo i dati e i sensori top per questo snapshot
    T_deg = DegradationData.(snap_field);
    T_fail = FailurePoints.(snap_field);
    current_sensors = unique(T_deg.Sensor); 
    % Creiamo una nuova figura per ogni snapshot
    figure('Name', sprintf('Comparison Snapshot %d', snap_id), 'NumberTitle', 'off', 'Color', 'w');
    sgtitle(sprintf('Distribuzione Feature: Degradation vs Failure (Snapshot %d)', snap_id));
    for i = 1:length(current_sensors)
        sensor_name = current_sensors{i};
        % Estraiamo i valori della media per le due classi
        val_deg = T_deg.Mean(strcmp(T_deg.Sensor, sensor_name));
        val_fail = T_fail.Mean(strcmp(T_fail.Sensor, sensor_name));
        % Prepariamo i dati per il boxplot (vettore unico + etichette)
        data_plot = [val_deg; val_fail];
        group_label = [repmat({'Degradation'}, length(val_deg), 1); ...
                       repmat({'Failure'}, length(val_fail), 1)];
        % Subplot per ogni sensore
        subplot(1, 3, i);
        boxplot(data_plot, group_label, 'Notch', 'on', 'Colors', 'rb');
        grid on;
        title(strrep(sensor_name, '_', ' '));
        ylabel('Mean Value');
        % Estetica: coloriamo i box per distinguerli meglio
        h = findobj(gca,'Tag','Box');
        if length(h) >= 2
            patch(get(h(1),'XData'), get(h(1),'YData'), 'b', 'FaceAlpha', 0.3); % Failure
            patch(get(h(2),'XData'), get(h(2),'YData'), 'r', 'FaceAlpha', 0.3); % Degradation
        end
    end
end



%% --- HELPER FUNCTIONS ---
function res = moving_features(signal, stop, N, step)
    stop = sort(stop);
    L = size(signal, 1);
    i = 1; 
    stop_ptr = 1;
    group_id = 1;
    
    % Inizializziamo i vettori per accumulare tutto il segnale dell'ESN
    res = struct('mean', [], 'std', [], 'kurtosis', [], 'skewness', [], ...
                 'Event_ID', [], 'Event_Index', [], 'Cycles', []);
    
    while i + N - 1 <= L
        % Se superiamo un punto di stop, incrementiamo il group_id
        if stop_ptr <= length(stop) && i >= stop(stop_ptr)
            stop_ptr = stop_ptr + 1;
            group_id = group_id + 1;
        end
        
        window_data = signal(i : i+N-1, 2);
        window_cycles = signal(i + N - 1, 1); % Ciclo alla fine della finestra
        
        res.mean(end+1)     = mean(window_data);
        res.std(end+1)      = std(window_data, 1);
        res.kurtosis(end+1) = kurtosis(window_data, 1) - 3; 
        res.skewness(end+1) = skewness(window_data, 1);
        res.Event_ID(end+1) = group_id;
        res.Event_Index(end+1) = i;
        res.Cycles(end+1)   = window_cycles;
        
        i = i + step;
    end
end

