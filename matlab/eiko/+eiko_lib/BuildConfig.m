classdef BuildConfig < handle
    %BuildConfig A Singleton class to manage project directory structures.
    
    properties (SetAccess = private)
        ProjectRoot   char
        SourceDir     char
        SourceFile    char
        IncludeDir    char
        EikoDir       char
        OutDir        char
        OutFile       char
    end
    
    methods (Access = private)
        function obj = BuildConfig()
            % Private Constructor: Prevents direct instantiation via 'BuildConfig()'
            
            % Get the directory of this class file

            % Get the full path of the this file to ensure relative paths
            % work regardless of the user's current working directory.
            classPath = mfilename('fullpath');
            [currentDir, ~, ~] = fileparts(classPath);
            
            % Define absolute paths for source code
            obj.SourceDir = fullfile(currentDir, 'src');
            if exist(obj.SourceDir, 'dir') ~= 7
                % Source is either in the matching directory or 3 levels up
                obj.SourceDir = fullfile(currentDir, '..', '..', '..', 'src');
            end
            obj.SourceFile = fullfile(obj.SourceDir, 'bindings', 'mex_bindings.cu');
            obj.IncludeDir = obj.SourceDir;
            
            % Define absolute paths for the compiled output
            obj.EikoDir    = fullfile(currentDir, '..');                     % fullfile(obj.ProjectRoot, 'matlab', 'eiko');
            obj.OutDir     = currentDir;                                     % fullfile(obj.EikoDir, '+eiko_lib');
            obj.OutFile    = fullfile(obj.OutDir, 'mex_bindings');
            
            addpath(obj.EikoDir);
        end
    end
    
    methods (Static)
        function obj = getInstance()
            % Static method to fetch or create the single instance
            persistent localObj
            if isempty(localObj) || ~isvalid(localObj)
                localObj = eiko_lib.BuildConfig();
            end
            obj = localObj;
        end
    end
end
