function df_out = get_step_points(df, column_name, up)
    % DF_GET_STEP_POINTS Returns rows corresponding to a 'step'
    % in the opposite direction of expectation.
    %
    % Inputs:
    %   df          : MATLAB Table
    %   column_name : String/Char of the column to check
    %   up          : Boolean (default false). 
    %                 If true, checks for drops (current < prev).
    %                 If false, checks for rises (current > prev).
    
    if nargin < 3
        up = false;
    end

    % Extract the specific column vector
    % df.(name) allows dynamic column access (like df[name] in Python)
    vals = df.(column_name);
    
    % 1. Create a shifted vector (equivalent to df.shift(1))
    % We prepend NaN so the indices align (Row 2 is compared to Row 1)
    prev_vals = [NaN; vals(1:end-1)];
    
    % 2. Create the Boolean Mask
    if up
        % Python: df[column] < df[column].shift(1)
        mask = vals < prev_vals;
    else
        % Python: df[column] > df[column].shift(1)
        mask = vals > prev_vals;
    end
    
    % 3. Apply Filter
    df_out = find(mask);
end