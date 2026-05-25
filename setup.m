function setup()
    % SETUP Compiles and installs Eiko CUDA MEX bindings for MATLAB.

    
    %% Input/Output Path Setup.
    
    % Get the full path of the current script.
    currentScriptPath = mfilename('fullpath');
    
    % Extract the directory from the full path.
    [currentDir, ~, ~] = fileparts(currentScriptPath);
    
    % Define absolute paths for source.
    sourceDir = fullfile(currentDir, 'src');
        sourceFile = fullfile(sourceDir, 'bindings', 'mex_bindings.cu');
    includeDir = sourceDir;

    % Define absolute paths for output.
    eikoDir = fullfile(currentDir, 'matlab', 'eiko');
        outDir = fullfile(eikoDir, '+eiko_lib');
            outFile = fullfile(outDir, 'mex_bindings');
    

    if exist(outFile, 'file') == 3  % 3 for MEX file.
        addpath(eikoDir);
        % disp('You are now ready to use Eiko! :)');
        % disp('Run "help eiko" to read the documentation.');
        return;
    end
    
    %% CUDA setup.
    
    user_ver = 0;     % User-installed CUDA version.
    builtin_ver = 0;  % MATLAB-shipped CUDA version.
    
    % Get built-in CUDA toolkit version (recent MATLAB versions ship with CUDA).
    tf = parallel.gpu.enableCUDAForwardCompatibility(true);  % Allow newer hardware.
    try
        gpu = gpuDevice();
        builtin_ver = gpu.ToolkitVersion;
    catch
        % If no GPU is detected, default to 0.
    end
    parallel.gpu.enableCUDAForwardCompatibility(tf);

    user_nvcc = '';  % Path to user-installed CUDA.
    
    % Check Environment Variables first.
    env_vars = {'CUDA_PATH', 'CUDA_HOME'};
    for i = 1:length(env_vars)
        p = getenv(env_vars{i});
        if ~isempty(p)
            nvcc_ext = ''; if ispc, nvcc_ext = '.exe'; end
            test_path = fullfile(p, 'bin', ['nvcc' nvcc_ext]);
            if exist(test_path, 'file')
                user_nvcc = test_path;
                break;
            end
        end
    end
    
    % Fallback to checking the system PATH.
    if isempty(user_nvcc)
        if ispc
            [st, out] = system('where nvcc');
        else
            [st, out] = system('which nvcc');
        end
        
        if st == 0 && ~isempty(out)
            paths = strsplit(strtrim(out), '\n');
            user_nvcc = paths{1}; % Take the first valid path found
        end
    end
    
    % Extract the version number from the found nvcc.
    if ~isempty(user_nvcc) && exist(user_nvcc, 'file')
        [st, out] = system(['"' user_nvcc '" --version']);
        if st == 0
            % Matches standard nvcc output, e.g., "release 12.1, V12.1.105"
            tok = regexp(out, 'release (\d+\.\d+)', 'tokens');
            if ~isempty(tok)
                user_ver = str2double(tok{1}{1});
            end
        end
    end
    
    % Decide whether to use the user's CUDA toolkit (use the latest version).
    use_user_cuda = (user_ver > builtin_ver);

    % Inform the user.
    disp('[Eiko] Compiling MEX extension for MATLAB... (This may take a minute)');
    if use_user_cuda
        fprintf('Detected a user-installed CUDA (v%.1f) that is newer than MATLAB''s built-in CUDA (v%.1f).\n', user_ver, builtin_ver);
    else
        fprintf('Using MATLAB''s built-in CUDA (v%.1f).\n', builtin_ver);
    end
    
    %% CUDA Compilation flags.
    libraries = '-lut';  % To allow detection of CTRL+C.
    base_flags = [ ...
        '-std=c++20 ', ...         % Use C++20 standard.
        '-DMATLAB_MEX_FILE ', ...  % Assert that we are making a MEX file.
        '--use_fast_math ' ...     % Fast approximations of division and sqrt.
    ];
    
    if ispc
        % Windows / MSVC Configuration.
        os_flags = [ ...
            '-allow-unsupported-compiler ', ...                % Fixes the VS 2022 version mismatch.
            '-Xcompiler "/Zc:preprocessor" ', ...              % MSVC standard preprocessor.
            '-Xcompiler "/std:c++20" ', ...                    % MSVC C++20 standard.
            '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH ', ... % Bypass STL checks.
            '-DNOMINMAX ', ...                                 % Prevent Windows.h from overwriting min/max macros.
        ];
    else
        % Linux / GCC Configuration.
        os_flags = []; % No os-specific flags are needed as of yet.
    end
    
    % Construct the NVCC argument string for mexcuda.
    % $NVCCFLAGS preserves the default MATLAB CUDA flags.
    nvcc_arg = ['NVCCFLAGS=$NVCCFLAGS ', base_flags, os_flags];
    nvcc_arg_specific = [nvcc_arg ' -arch=native '];
    
    if use_user_cuda
        % Safely generate the CCCL paths just for this attempt
        [cuda_bin_dir, ~, ~] = fileparts(user_nvcc);
        [cuda_root, ~, ~] = fileparts(cuda_bin_dir);
        
        cccl_include = fullfile(cuda_root, 'include', 'cccl');
        cccl_flags = '';
        if exist(cccl_include, 'dir')
            thrust_include = fullfile(cccl_include, 'thrust');
            libcudacxx_include = fullfile(cccl_include, 'libcudacxx');
            cub_include = fullfile(cccl_include, 'cub');
            
            cccl_flags = sprintf('-I "%s" -I "%s" -I "%s" -I "%s" ', ...
                cccl_include, thrust_include, libcudacxx_include, cub_include);
        end
        nvcc_arg_user = [nvcc_arg_specific, cccl_flags, os_flags, ' -arch=native'];
    end
    
    %% Run the compiler.
    success = false;
    attempt = 1;
    if use_user_cuda
        fprintf('Attempt %d: Compiling with user-installed CUDA (-arch=native)...\n', attempt);
        
        % Setup environment variables.
        MW_NVCC_PATH = getenv('MW_NVCC_PATH');
        MW_ALLOW_ANY_CUDA = getenv('MW_ALLOW_ANY_CUDA');
        setenv('MW_NVCC_PATH', user_nvcc);
        setenv('MW_ALLOW_ANY_CUDA', '1');
        try
            % Attempt 1: Try to compile using the user-installed CUDA library.
            mexcuda('-R2018a', nvcc_arg_user, ['-I', includeDir], '-outdir', outDir, sourceFile, '-lut');
            success = true;
        catch ME
            % Discard any errors and just try again further below.
            fprintf('Compilation failed due to: %s\n', ME.message);
        end
        % Restore environment variables.
        setenv('MW_NVCC_PATH', MW_NVCC_PATH);
        setenv('MW_ALLOW_ANY_CUDA', MW_ALLOW_ANY_CUDA);
        attempt = attempt + 1;
    end
    
    if ~success
        try
            % Attempt number 2: Try to compile for the currently connected GPU.
            fprintf('Attempt %d: Compiling with MATLAB''s built-in CUDA (-arch=native)...\n', attempt+1);
            
            % This may fail if the GPU is more recent than the CUDA installation.
            mexcuda('-R2018a', nvcc_arg_specific, ['-I', includeDir], '-outdir', outDir, sourceFile, libraries);
        catch ME
            fprintf('Compilation failed due to: %s\n', ME.message);
            try
                % Attempt number 3: Try to compile for a non-specific GPU.
                fprintf('Attempt %d: Compiling with MATLAB''s built-in CUDA (default arch)...\n', attempt+2);
                mexcuda('-R2018a', nvcc_arg, ['-I', includeDir], '-outdir', outDir, sourceFile, libraries);
            catch ME
                fprintf('Compilation failed due to: %s\n', ME.message);
                disp('Compilation failed. :(');
                rethrow(ME);
            end
        end
    end

    % We either successfully compiled the code now, or an error was thrown.
    disp(['MEX file saved to: ', outDir]);
    disp('Congratulations, you are now ready to use Eiko! :)');
    disp('Run "help eiko" to read the documentation.');
    addpath(eikoDir);  % Add MEX file to path.
end
