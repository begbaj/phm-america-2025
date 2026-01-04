clear; clc; close all;

%% 1. CONFIGURATION
DATA_PATH = 'data/PHM2025_training_data/training_data.csv';
ESNS = [101, 102, 103, 104];
% Analyzing just the first few snapshots to ensure data availability
SNAPSHOTS_TO_ANALYZE = [1, 2, 3, 4, 5, 6, 7, 8]; 
FEATURES_NAMES = {'mean', 'std', 'rms', 'kurtosis', 'skewness', 'shape_factor'};
SENSORS = {'Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_Pt2', ...
           'Sensed_TAT', 'Sensed_WFuel', 'Sensed_VAFN', 'Sensed_VBV', ...
           'Sensed_Fan_Speed', 'Sensed_Core_Speed', 'Sensed_T25', 'Sensed_T3', ...
           'Sensed_Ps3', 'Sensed_T45', 'Sensed_P25', 'Sensed_T5'};
WINDOW_SIZE = 10;
STEP_SIZE = 3;
CUMULATIVE_TYPE = ["Cumulative_WWs", "Cumulative_HPT_SVs", "Cumulative_HPC_SVs"];

%% 2. PRE-PROCESSING (Global Moving Mean)
T_raw = readtable(DATA_PATH);
T_movmean = T_raw;
fprintf('Applying pre-processing moving mean...\n');
for esn = ESNS
    idx = T_movmean.ESN == esn;
    for sensor = SENSORS
        T_movmean{idx, sensor} = movmean(T_movmean{idx, sensor}, WINDOW_SIZE);
    end
end
clearvars idx esn sensor

%% 3. COLLECTION PHASE
% Organize data into an Ensemble format: {SnapID}.(Sensor).(Feature) = {Traj1, Traj2...}
fprintf('Collecting trajectories...\n');
DataStore = struct();

for esn = ESNS
    df = T_movmean(T_movmean.ESN == esn, :);
    df.Index = (1:height(df))';
    
    stops_table = get_step_points(df, CUMULATIVE_TYPE(1));
    stop_indices = stops_table.Index;
    
    for s_idx = 1:length(SENSORS)
        sens_name = SENSORS{s_idx};
        safe_sens = regexprep(sens_name, '\W', ''); % Field name safe

        signal = [df.Index, df.(sens_name)];
        segments = moving_features(signal, stop_indices, WINDOW_SIZE, STEP_SIZE);
        
        for snap_id = SNAPSHOTS_TO_ANALYZE
            if snap_id > length(segments), continue; end
            
            seg_data = segments(snap_id);
            if length(seg_data.mean) < 5, continue; end % Skip tiny segments
            
            % Initialize DataStore nodes if they don't exist
            if ~isfield(DataStore, 'Snap') || length(DataStore) < snap_id
                DataStore(snap_id).Index = snap_id;
            end
            if ~isfield(DataStore(snap_id), safe_sens)
                DataStore(snap_id).(safe_sens) = struct();
            end
            
            for f_name = FEATURES_NAMES
                feat = f_name{1};
                traj = seg_data.(feat);
                
                % TOOLBOX REQUIREMENT: Column Vectors
                if size(traj, 2) > size(traj, 1), traj = traj'; end
                
                if ~isfield(DataStore(snap_id).(safe_sens), feat)
                    DataStore(snap_id).(safe_sens).(feat) = {};
                end
                DataStore(snap_id).(safe_sens).(feat){end+1} = traj;
            end
        end
    end
end

%% 4. CALCULATION PHASE (Mantendo tutti i valori 0)
fprintf('Calculating Metrics...\n');
SnapshotResults = containers.Map('KeyType', 'double', 'ValueType', 'any');

for snap_id = SNAPSHOTS_TO_ANALYZE
    if snap_id > length(DataStore) || isempty(DataStore(snap_id)), continue; end
    
    % Prepare table for this snapshot
    results = table({}, {}, [], [], [], [], ...
        'VariableNames', {'Sensor', 'Feature', 'Monotonicity', 'Trendability', 'Prognosability', 'FinalScore'});
    
    fs = fieldnames(DataStore(snap_id));
    for i = 1:length(fs)
        sens_safe = fs{i};
        if strcmp(sens_safe, 'Index'), continue; end
        
        feats = fieldnames(DataStore(snap_id).(sens_safe));
        for j = 1:length(feats)
            feat_name = feats{j};
            ensembleData = DataStore(snap_id).(sens_safe).(feat_name);
            
            % Inizializziamo a 0 (Il dato di default se mancano condizioni)
            m = 0; t = 0; p = 0;
            
            if ~isempty(ensembleData)
                try
                    % Monotonicità (Sempre calcolabile se c'è almeno 1 traiettoria)
                    m_vec = monotonicity(ensembleData); 
                    m = mean(m_vec, 'omitnan');
                    
                    % Trend e Progn (Solo se ci sono almeno 2 motori)
                    if length(ensembleData) >= 2
                        t_val = trendability(ensembleData);
                        t = mean(t_val, 'all', 'omitnan');
                        p_val = prognosability(ensembleData);
                        p = mean(p_val, 'all', 'omitnan');
                    end
                catch ME
                    fprintf('Error calc metrics for Snap %d, %s, %s: %s\n', snap_id, sens_safe, feat_name, ME.message);
                    % Se il toolbox va in errore (es. segnale piatto), i valori restano 0
                end
            end
            
            % Gestione NaN e calcolo Score
            m(isnan(m)) = 0; t(isnan(t)) = 0; p(isnan(p)) = 0;
            score = m + t + p;
            
            % AGGIUNGIAMO SEMPRE LA RIGA (anche se score è 0)
            % Usiamo una cell array per l'inserimento compatibile con i tipi definiti
            new_row = {string(sens_safe), string(feat_name), m, t, p, score};
            results = [results; new_row];
        end
    end
    
    % Ordiniamo: i migliori in alto, gli zeri in fondo
    results = sortrows(results, 'FinalScore', 'descend');
    SnapshotResults(snap_id) = results;
    fprintf('Snapshot %d: Analizzate %d feature (inclusi zeri).\n', snap_id, height(results));
end

%% 5. PLOTTING PHASE (Visualizzazione Top 10)
fprintf('\n--- Generating Plots ---\n');
res_keys = keys(SnapshotResults);

for k = 1:length(res_keys)
    snap_id = res_keys{k};
    tbl = SnapshotResults(snap_id);
    
    % Se la tabella è vuota (zero righe totali), salta
    if height(tbl) == 0, continue; end
    
    % Estrazione del vincitore (anche se lo score è 0, prendiamo il primo)
    best_sens = char(tbl.Sensor(1));
    best_feat = char(tbl.Feature(1));
    best_score = tbl.FinalScore(1);
    
    fprintf('Snapshot %d Winner: %s (%s) - Score: %.2f\n', snap_id, best_sens, best_feat, best_score);
    
    figure('Name', sprintf('Analysis Snap %d', snap_id), 'Color', 'w');
    tlo = tiledlayout(2,1, 'Padding', 'compact');
    
    % 1. Bar Chart (Top 10 migliori)
    nexttile;
    n_show = min(10, height(tbl));
    b = bar(tbl.FinalScore(1:n_show), 'FaceColor', [0.2 0.6 0.5]);
    grid on;
    xticks(1:n_show);
    % Pulizia nomi per il grafico
    clean_labels = strrep(tbl.Sensor(1:n_show) + " " + tbl.Feature(1:n_show), "_", " ");
    xticklabels(clean_labels);
    xtickangle(30);
    title(sprintf('Top 10 PHM Scores - Snapshot %d', snap_id));
    ylabel('Score (M+T+P)');
    
    % 2. Traiettorie (Del primo in classifica)
    nexttile;
    if isfield(DataStore(snap_id), best_sens)
        trajs = DataStore(snap_id).(best_sens).(best_feat);
        hold on;
        for tr = 1:length(trajs)
            plot(trajs{tr}, 'LineWidth', 1.5);
        end
        hold off;
        title(sprintf('Traiettorie: %s (%s)', strrep(best_sens, '_', ' '), best_feat));
        xlabel('Windows'); ylabel('Feature Value'); grid on;
    end
end

%% --- HELPER FUNCTIONS ---
function df_out = get_step_points(df, column_name)
    vals = df.(column_name);
    prev_vals = [NaN; vals(1:end-1)];
    mask = (vals ~= prev_vals) & ~isnan(prev_vals); 
    df_out = df(mask, :);
end

function res = moving_features(signal, stop, N, step)
    stop = sort(stop);
    L = size(signal, 1);
    i = 1; 
    stop_ptr = 1;
    group_id = 1;
    
    % Initialize struct
    res = struct('rms', [], 'mean', [], 'std', [], 'kurtosis', [], 'skewness', [], 'shape_factor', []);
    
    while i + N - 1 <= L
        if stop_ptr <= length(stop) && signal(i,1) >= stop(stop_ptr)
            stop_ptr = stop_ptr + 1;
            group_id = group_id + 1;
            % Grow struct array for next group
            res(group_id) = struct('rms', [], 'mean', [], 'std', [], 'kurtosis', [], 'skewness', [], 'shape_factor', []);
            i = i + 1; 
            continue; 
        end
        
        window = signal(i : i+N-1, 2);
        
        % Grow feature vectors (default row vectors in struct)
        res(group_id).mean(end+1) = mean(window);
        res(group_id).std(end+1)  = std(window, 1);
        res(group_id).rms(end+1)  = rms(window);
        res(group_id).kurtosis(end+1) = kurtosis(window, 1) - 3; 
        res(group_id).skewness(end+1) = skewness(window, 1);
        
        ma = mean(abs(window));
        if ma ~= 0, sf = rms(window)/ma; else, sf = 0; end
        res(group_id).shape_factor(end+1) = sf;
        
        i = i + step;
    end
end