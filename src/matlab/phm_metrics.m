clear; clc; close all;

% --- CONFIGURATION ---
dataPath = 'data/PHM2025_training_data/training_data.csv';
%data_path = ''
ESNS = [101, 102, 103, 104];
SNAPSHOTS_TO_PLOT = [1, 2, 3, 4, 5, 6, 7, 8]; % Which snapshots (flights) to visualize?
FEATURES_NAMES = {'mean', 'std', 'rms', 'kurtosis', 'skewness', 'shape_factor'};
SENSORS = {'Sensed_Altitude', 'Sensed_Mach', 'Sensed_Pamb', 'Sensed_Pt2', ...
           'Sensed_TAT', 'Sensed_WFuel', 'Sensed_VAFN', 'Sensed_VBV', ...
           'Sensed_Fan_Speed', 'Sensed_Core_Speed', 'Sensed_T25', 'Sensed_T3', ...
           'Sensed_Ps3', 'Sensed_T45', 'Sensed_P25', 'Sensed_T5'};

window_size = 10;
step_size = 3;

% --- 1. LOAD DATA ---
fprintf('Loading data...\n');
T_raw = readtable(dataPath);

% --- 1b. FILTRAGGIO GLOBALE ---
fprintf('Filtering sensors with moving average...\n');
% Creiamo una copia per non sovrascrivere i dati originali se servono
T_filtered = T_raw; 

for i = 1:length(ESNS)
    current_esn = ESNS(i);
    % Identifica le righe di questo motore
    idx = T_filtered.ESN == current_esn;
    
    % Applica il filtro solo alle colonne dei sensori per questo ESN
    for s = 1:length(SENSORS)
        sens_name = SENSORS{s};
        % movmean con finestra 15 (puoi regolarlo)
        T_filtered{idx, sens_name} = movmean(T_filtered{idx, sens_name}, 1);
    end
end

% --- 2. MAIN LOOP: PER ESN ---
for esn_idx = 1:length(ESNS)
    current_esn = ESNS(esn_idx);
    fprintf('\nProcessing ESN: %d\n', current_esn);
    
    % Prepare Data for this ESN
    df = T_filtered(T_filtered.ESN == current_esn, :);
    df = sortrows(df, 'Cycles_Since_New');
    df.Index = (1:height(df))';
    
    % Get Split Points (Snapshots boundaries)
    stops_table = get_step_points(df, 'Cumulative_WWs');
    if isempty(stops_table)
        stop_indices = [];
    else
        stop_indices = stops_table.Index;
    end
    
    % Storage for this ESN: Map<SnapshotID, Table_of_Rankings>
    % We will store ranking tables for every snapshot here
    SnapshotRankings = containers.Map('KeyType', 'double', 'ValueType', 'any');
    
    % --- 3. PER SENSOR CALCULATION ---
    for s_idx = 1:length(SENSORS)
        sens_name = SENSORS{s_idx};
        
        % Prepare signal matrix [Index, Value]
        signal = [df.Index, df.(sens_name)];
        
        % Run Moving Features (returns struct array, one element per snapshot)
        % segments(1) = Snapshot 1, segments(2) = Snapshot 2, etc.
        segments = moving_features(signal, stop_indices, window_size, step_size);
        
        % Iterate through the specific snapshots we care about
        for snap_id = SNAPSHOTS_TO_PLOT
            if snap_id > length(segments)
                continue; % Skip if this ESN doesn't have enough snapshots
            end
            
            seg_data = segments(snap_id);
            
            % Check if segment has enough data
            if length(seg_data.mean) < 5 
                continue; 
            end
            
            % --- CALCULATE MONOTONICITY FOR ALL FEATURES ---
            for f_name = FEATURES_NAMES
                feat = f_name{1};
                traj = seg_data.(feat);
                
                % Monotonicity = Abs(Spearman Correlation with Time)
                time_vec = (1:length(traj))';
                mono = abs(corr(time_vec, traj(:), 'Type', 'Spearman'));
                
                if isnan(mono), mono = 0; end
                
                % Store result in a temporary list for this snapshot
                % We need to aggregate these into a table later
                if ~isKey(SnapshotRankings, snap_id)
                     % Initialize empty table for this snapshot if new
                     SnapshotRankings(snap_id) = table({}, {}, [], 'VariableNames', {'Sensor', 'Feature', 'Monotonicity'});
                end
                
                current_tbl = SnapshotRankings(snap_id);
                new_row = table({sens_name}, {feat}, mono, 'VariableNames', {'Sensor', 'Feature', 'Monotonicity'});
                SnapshotRankings(snap_id) = [current_tbl; new_row];
            end
        end
    end
    
    % --- 4. PLOT TOP SENSORS FOR EACH SNAPSHOT (For this ESN) ---
    if isempty(SnapshotRankings.keys)
        fprintf('  No valid snapshots found for ESN %d\n', current_esn);
        continue;
    end
    
    % Create a figure for this ESN
    fig = figure('Name', sprintf('ESN %d Analysis', current_esn), 'Color', 'w');
    fig.WindowState = 'maximized';
    t = tiledlayout(fig, length(SNAPSHOTS_TO_PLOT), 2, 'TileSpacing', 'compact');
    title(t, sprintf('Top Features by Snapshot for ESN %d', current_esn));
    
    keys_list = SnapshotRankings.keys;
    
    for k = 1:length(keys_list)
        snap_id = keys_list{k};
        tbl = SnapshotRankings(snap_id);
        
        % Sort by Monotonicity
        tbl = sortrows(tbl, 'Monotonicity', 'descend');
        top_5 = tbl(1:min(5, height(tbl)), :);
        
        % 1. Bar Chart of Scores
        nexttile;
        bar(top_5.Monotonicity);
        labels = strcat(top_5.Sensor, " (", top_5.Feature, ")");
        xticklabels(labels);
        xtickangle(25);
        ylim([0 1]);
        title(sprintf('Snapshot %d: Top 5 Monotonicity', snap_id));
        grid on;
        
        % 2. Plot the trajectory of the #1 Best Feature
        nexttile;
        best_sens = top_5.Sensor{1};
        best_feat = top_5.Feature{1};
        
        % Re-extract the specific trajectory to plot it
        % (Inefficient to re-calc, but keeps code cleaner than storing everything)
        raw_sig = [df.Index, df.(best_sens)];
        segs = moving_features(raw_sig, stop_indices, window_size, step_size);
        best_traj = segs(snap_id).(best_feat);
        
        plot(best_traj, 'LineWidth', 1.5, 'Color', 'r');
        title(sprintf('Best Feature Trajectory: %s - %s', best_sens, best_feat));
        xlabel('Window Step');
        grid on;
    end
    drawnow;
end

% --- HELPER FUNCTIONS ---

function df_out = get_step_points(df, column_name)
    vals = df.(column_name);
    prev_vals = [NaN; vals(1:end-1)];
    mask = (vals ~= prev_vals) & ~isnan(prev_vals); % different
    df_out = df(mask, :);
end

function res = moving_features(signal, stop, N, step)
    % signal: [Index, Value]
    % stop: vector of indices
    stop = sort(stop);
    L = size(signal, 1);
    i = 1; 
    stop_ptr = 1;
    
    % Initialize struct array logic
    group_id = 1;
    res = struct('rms', [], 'mean', [], 'std', [], 'kurtosis', [], 'skewness', [], 'shape_factor', []);
    
    while i + N - 1 <= L
        % Check if we passed a stop point
        if stop_ptr <= length(stop) && signal(i,1) >= stop(stop_ptr)
            stop_ptr = stop_ptr + 1;
            group_id = group_id + 1;
            res(group_id) = struct('rms', [], 'mean', [], 'std', [], 'kurtosis', [], 'skewness', [], 'shape_factor', []);
            continue; 
        end
        
        window = signal(i : i+N-1, 2);
        
        % Calc features
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