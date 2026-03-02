function res = moving_features(signal, stop, N, step, separate)
    % MOVING_FEATURES_WITH_STOP Calculates sliding window features with stop points.
    %
    % Inputs:
    %   signal : Nx2 matrix where col 1 is the index/time (int), col 2 is value (float)
    %   stop   : Vector of stop indices (ints)
    %   N      : Window length
    %   step   : (Optional) Step size. Defaults to N.
    %
    % Output:
    %   res    : A struct array where res(k) corresponds to the k-th group.
    
    if isempty(step)
        step = N;
    end

    % Helper to initialize a new group structure
    function s = new_group()
        s = struct('rms', [], 'mean', [], 'std', [], ...
                   'kurtosis', [], 'skewness', [], 'shape_factor', []);
    end

    % Initialize Result Struct Array
    % Python uses group_id 0, 1... we use indices 1, 2...
    current_group_idx = 1; 
    res = repmat(struct('rms', [], 'mean', [], 'std', [], 'kurtosis', [], 'skewness', [], 'shape_factor', []), 1, 1);
    res(current_group_idx) = new_group();

    L = size(signal, 1);
    stop_ptr = 1; % MATLAB indices start at 1
    
    % Ensure stop list is sorted
    stop = sort(stop);

    i = 1; % Signal row pointer
    
    while i + N - 1 <= L
        % Check if current signal index matches or passes the next stop point
        % signal(i, 1) accesses the 'index' column (equivalent to signal[i][0])
        if stop_ptr <= length(stop) && signal(i, 1) >= stop(stop_ptr)
            stop_ptr = stop_ptr + 1;
            if separate
                current_group_idx = current_group_idx + 1;
                res(current_group_idx) = new_group();
            end 
            % Continue to re-evaluate (matches Python logic)
            continue; 
        end

        % Extract values for the window (Column 2 contains values)
        % Indices: i to i + N - 1
        window = signal(i : i + N - 1, 2);

        if isempty(window)
            i = i + step;
            continue;
        end

        % --- Calculations ---
        % Note: Using specialized flags to match Python's numpy/scipy behavior
        
        m = mean(window);
        
        % Python np.std is population (N), MATLAB std default is sample (N-1).
        % We use flag 1 to force population std to match Python.
        s = std(window, 1); 
        
        r = rms(window);

        % Append results to the current group arrays
        res(current_group_idx).mean(end+1) = m;
        res(current_group_idx).std(end+1)  = s;
        res(current_group_idx).rms(end+1)  = r;

        % Kurtosis: Python (fisher=True) is Excess Kurtosis. 
        % MATLAB returns raw kurtosis (Normal=3). Subtract 3 to match.
        % We use flag 1 for bias correction to match scipy default behavior if needed,
        % usually raw uncorrected is safer for comparison, but here is standard:
        k = kurtosis(window, 1) - 3; 
        res(current_group_idx).kurtosis(end+1) = k;

        % Skewness: Python scipy.stats.skew is biased. 
        % MATLAB skewness default is unbiased. Use flag 1 to match Python.
        sk = skewness(window, 1);
        res(current_group_idx).skewness(end+1) = sk;

        % Shape Factor
        mean_abs = mean(abs(window));
        if mean_abs ~= 0
            sf = r / mean_abs;
        else
            sf = 0;
        end
        res(current_group_idx).shape_factor(end+1) = sf;

        i = i + step;
    end
end