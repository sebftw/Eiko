% =========================================================================
% Eiko Activation Script
% =========================================================================
% Locates the 'eiko' directory relative to this script's location and adds
% it to the MATLAB search path. This ensures Eiko can be used regardless of
% the current working directory.

% Determine the directory containing this script.
scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);

% Construct the path to the 'eiko' directory.
eikoPath = fullfile(scriptDir, 'eiko');

% Check if the 'eiko' directory actually exists before adding it.
if ~isfolder(eikoPath)
    error('Eiko:DirectoryNotFound', ...
          'The "eiko" directory was not found at\n   %s.', eikoPath);
end

% Add the 'eiko' folder to the top of MATLAB's search path.
% This allows MATLAB to find and run the functions/classes inside it.
addpath(eikoPath);
