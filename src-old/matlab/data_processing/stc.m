function t = stc(t)
    % 1. Define variables to clean
    % Note: Use "t" (lowercase) everywhere inside the function
    varsToClean = ["Altitude","Mach","Pamb","Pt2","TAT","WFuel","VAFN","VBV", ...
                   "Fan_Speed","Core_Speed","T25","T3","Ps3","T45","P25","T5"];

    % 2. FORCE NUMERIC CONVERSION (Must be first!)
    % This handles cases where CSV import turned numbers into text
    for i = 1:length(varsToClean)
        varName = varsToClean(i);
        
        if ismember(varName, t.Properties.VariableNames)
            if ~isnumeric(t.(varName))
                % Convert strings/cells to double; non-numeric text becomes NaN
                t.(varName) = str2double(string(t.(varName)));
            end
        else
            warning("Variable '%s' not found in table.", varName);
        end
    end

    % 3. CLEANING OPERATIONS
    % Now that we are sure they are numeric, we can process them
    t = fillmissing(t, "movmean", 10, "DataVariables", varsToClean);
    t = filloutliers(t, "nearest", "movmedian", 2, "DataVariables", varsToClean);
    % t = smoothdata(t, "movmean", "SmoothingFactor", 0.25, "DataVariables", varsToClean);
    % tolgo lo smoothing perchè potrebbe non essere corretto farlo qua
    
    % 4. CATEGORICAL CONVERSION
    % Convert IDs to categorical for better grouping performance
    groupVars = ["ww_cycle", "hpc_cycle", "hpt_cycle", "esn", "snap"];
    for g = 1:length(groupVars)
        if ismember(groupVars(g), t.Properties.VariableNames)
            t.(groupVars(g)) = categorical(t.(groupVars(g)));
        end
    end
end