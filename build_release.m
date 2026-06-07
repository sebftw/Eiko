function build_release()
	setup('release');
	return;
    % Get the directory where this script resides
    scriptDir = fileparts(mfilename('fullpath'));
    
    % Use that directory to find Eiko.prj
    prjFile = fullfile(scriptDir, 'Eiko.prj');
    
    % Ensure the releases directory exists
    outputDir = fullfile(scriptDir, 'releases');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    outputFile = fullfile(outputDir, 'Eiko.mltbx');
    
    % Package
    matlab.addons.toolbox.packageToolbox(prjFile, outputFile);
    
    disp(['Success! Toolbox generated at: ', outputFile]);
end
