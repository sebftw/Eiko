% =========================================================================
% Eiko Initialization Script
% =========================================================================
% Purpose: Automatically locates the 'eiko' package directory relative to 
%          this script's location and adds it to the MATLAB search path.
%          This ensures dependencies are resolved regardless of the current
%          working directory.

% Determine the absolute directory path of this running script.
%    - mfilename('fullpath') returns the full path and filename of this script.
%    - fileparts splits it into [path, name, extension]. 
%    - We only need the path (scriptDir), so '~' is used to ignore the rest.
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));

% Construct the full platform-independent path to the 'eiko' subdirectory.
%    - fullfile ensures correct folder slashes ('/' or '\') based on the OS.
eikoPath = fullfile(scriptDir, 'eiko');

% Dynamically add the 'eiko' folder to the top of MATLAB's search path.
%    - This allows MATLAB to find and execute any functions/classes inside it.
addpath(eikoPath);

% 4. Print a status message to the Command Window confirming initialization.
disp('[*] Eiko environment active. Ready to compute!');
