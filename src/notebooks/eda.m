% ==========================================
% Octave Script for PHM Data Analysis
% ==========================================

% 1. Load Data
fprintf('Loading data...\n');

filename = "../../Data/PHM2025_training_data/training_data.csv";

% Check if file exists
if exist(filename, 'file') ~= 2
    error('File not found: %s. Please check the path.', filename);
end

% A. Read Headers (First line only)
fid = fopen(filename, 'r');
header_line = fgetl(fid);
fclose(fid);
% Split header string by comma
headers = strsplit(header_line, ',');
% Remove any surrounding quotes or whitespace from headers
headers = strtrim(strrep(headers, '"', ''));

% B. Read Numeric Data
% dlmread reads the numeric matrix, skipping the first row (headers)
% offset R=1, C=0
train_data = dlmread(filename, ',', 1, 0);

fprintf('Data loaded successfully. Rows: %d, Cols: %d\n', size(train_data));

% 2. Calculate and Plot Correlation Matrix
fprintf('Calculating and plotting correlation matrix...\n');

% Find sensor columns (using strncmp for "Sensed_" prefix)
% strncmp checks the first 7 characters
sensor_mask = cellfun(@(x) strncmp(x, 'Sensed_', 7), headers);
sensor_cols_indices = find(sensor_mask);
sensor_data = train_data(:, sensor_cols_indices);
sensor_headers = headers(sensor_cols_indices);

% Calculate correlation matrix
% Use corrcoef (core function) instead of corr (statistics pkg)
corr_matrix = corrcoef(sensor_data);

% Plot heatmap
figure('Name', 'Correlation Matrix');
% imagesc draws the matrix as an image
imagesc(corr_matrix);
colormap('cool');
colorbar;
title('Sensor Correlation Matrix');

% Set axis labels
% We adjust font size for readability
set(gca, 'XTick', 1:length(sensor_headers));
set(gca, 'YTick', 1:length(sensor_headers));
set(gca, 'XTickLabel', sensor_headers, 'XTickLabelRotation', 90, 'FontSize', 8);
set(gca, 'YTickLabel', sensor_headers, 'FontSize', 8);
axis('square');
fprintf('Correlation matrix plotted.\n');

% 3. Identify Maintenance Events
fprintf('Identifying maintenance events...\n');

% Helper function to safely find column index
find_col = @(name) find(strcmp(name, headers));

esn_col = find_col('ESN');
wws_col = find_col('Cumulative_WWs');
hpc_col = find_col('Cumulative_HPC_SVs');
hpt_col = find_col('Cumulative_HPT_SVs');
cycles_col = find_col('Cycles_Since_New');

if isempty(esn_col) || isempty(cycles_col)
    error('Critical columns (ESN or Cycles) missing from CSV.');
end

% Find points where cumulative counters change
% diff() calculates difference between adjacent elements.
wws_points_indices = find(diff(train_data(:, wws_col)) > 0);
hpc_points_indices = find(diff(train_data(:, hpc_col)) > 0);
hpt_points_indices = find(diff(train_data(:, hpt_col)) > 0);

% Note: diff returns index i for (i+1) - i.
% We usually want the index *after* the change (the event), so we add +1
wws_points = train_data(wws_points_indices + 1, :);
hpc_points = train_data(hpc_points_indices + 1, :);
hpt_points = train_data(hpt_points_indices + 1, :);

fprintf('Maintenance events identified.\n');


% 4. Plot Sensor Data with Events
fprintf('Generating sensor plots...\n');

% Get the list of unique Engine Serial Numbers (ESNs)
unique_esns = unique(train_data(:, esn_col));

% Select a default sensor to plot
default_sensor_name = 'Sensed_T25';
sensor_to_plot_col = find_col(default_sensor_name);

if isempty(sensor_to_plot_col)
    warning('Default sensor %s not found. Using first sensor available.', default_sensor_name);
    sensor_to_plot_col = sensor_cols_indices(1);
    default_sensor_name = headers{sensor_to_plot_col};
end

% SAFETY LIMIT: Plot only the first 3 engines to prevent crashing Octave
% Change '3' to 'length(unique_esns)' to plot ALL engines.
max_plots = 3;

fprintf('Plotting first %d ESNs...\n', max_plots);

for i = 1:min(length(unique_esns), max_plots)
    esn = unique_esns(i);

    % Create a new figure for each ESN
    figure('Name', sprintf('ESN %d - %s', esn, default_sensor_name));
    hold on;

    % Extract data for the current ESN
    esn_indices = find(train_data(:, esn_col) == esn);
    engine_data = train_data(esn_indices, :);

    % Plot the sensor data
    plot(engine_data(:, cycles_col), engine_data(:, sensor_to_plot_col), 'k-', 'LineWidth', 1.5);

    % --- Add vertical lines for maintenance events ---
    ylimits = get(gca, 'YLim');

    % Helper for plotting lines
    plot_event = @(event_rows, color, label, align) ...
        arrayfun(@(idx) ...
            [line([event_rows(idx, cycles_col) event_rows(idx, cycles_col)], ylimits, 'Color', color, 'LineStyle', '--', 'LineWidth', 1), ...
             text(event_rows(idx, cycles_col), ylimits(2), label, 'Color', color, 'HorizontalAlignment', align, 'VerticalAlignment', 'top')], ...
             1:size(event_rows, 1), 'UniformOutput', false);

    % Filter events for this specific ESN
    esn_wws = wws_points(wws_points(:, esn_col) == esn, :);
    esn_hpc = hpc_points(hpc_points(:, esn_col) == esn, :);
    esn_hpt = hpt_points(hpt_points(:, esn_col) == esn, :);

    % Draw lines if events exist
    if ~isempty(esn_wws)
        for k=1:size(esn_wws,1)
           c = esn_wws(k, cycles_col);
           line([c c], ylimits, 'Color', 'r', 'LineStyle', '--');
           text(c, ylimits(2), 'WWs', 'Color', 'r', 'VerticalAlignment', 'top');
        end
    end

    if ~isempty(esn_hpc)
        for k=1:size(esn_hpc,1)
           c = esn_hpc(k, cycles_col);
           line([c c], ylimits, 'Color', 'b', 'LineStyle', '--');
           text(c, ylimits(2)*0.95, 'HPC', 'Color', 'b', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');
        end
    end

    if ~isempty(esn_hpt)
         for k=1:size(esn_hpt,1)
           c = esn_hpt(k, cycles_col);
           line([c c], ylimits, 'Color', 'g', 'LineStyle', '--');
           text(c, ylimits(1), 'HPT', 'Color', 'g', 'VerticalAlignment', 'bottom');
        end
    end

    % Finalize plot
    title(sprintf('ESN %d - Sensor: %s', esn, default_sensor_name));
    xlabel('Cycles Since New');
    ylabel(strrep(default_sensor_name, '_', ' '));
    grid on;
    hold off;
end

fprintf('Done. (Plotted %d of %d engines)\n', min(length(unique_esns), max_plots), length(unique_esns));
