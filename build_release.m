function build_release()
    % Ensure the binaries are compiled first
    disp('Compiling release binaries...');
    setup('release');
    
    % Package the toolbox using the predefined configuration
    disp('Packaging Eiko.mltbx...');
    
    % MATLAB's built-in packaging function
    prjFile = fullfile(pwd, 'Eiko.prj');
    outputFile = fullfile(pwd, 'releases', 'Eiko.mltbx');
    
    matlab.addons.toolbox.packageToolbox(prjFile, outputFile);
    
    disp(['Success! Toolbox generated at: ', outputFile]);
end
