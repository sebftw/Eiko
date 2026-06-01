function setup(build_type)
    % SETUP Compiles and installs Eiko CUDA MEX bindings for MATLAB.
    %
    % Usage:
    %   setup()          - Compiles for the local GPU (-arch=native) using C++17.
    %   setup('release') - Compiles a fat binary (-arch=all-major) and
    %                      statically links libstdc++ on Linux to ensure compatibility.

    if nargin < 1
        build_type = 'native';
    end
    
    is_release = strcmpi(build_type, 'release');

    %% Input/Output Path Setup
    
    % Get the full path of the current script to ensure relative paths work
    % regardless of the user's current working directory.
    currentScriptPath = mfilename('fullpath');
    [currentDir, ~, ~] = fileparts(currentScriptPath);
    
    % Define absolute paths for source code.
    sourceDir = fullfile(currentDir, 'src');
        sourceFile = fullfile(sourceDir, 'bindings', 'mex_bindings.cu');
    includeDir = sourceDir;

    % Define absolute paths for the compiled output.
    eikoDir = fullfile(currentDir, 'matlab', 'eiko');
        outDir = fullfile(eikoDir, '+eiko_lib');
            outFile = fullfile(outDir, 'mex_bindings');
    
    % If the MEX file already exists, just add it to the path and exit.
    if not(is_release) && exist(outFile, 'file') == 3  % 3 indicates a MEX file.
        addpath(eikoDir);
        return;
    end
    
    %% CUDA Setup & Version Detection
    
    user_ver = 0;     % User-installed CUDA toolkit version.
    builtin_ver = 0;  % MATLAB's internal CUDA toolkit version.
    
    % Query MATLAB for its built-in CUDA version.
	try
		tf = parallel.gpu.enableCUDAForwardCompatibility(true); 
        gpu = gpuDevice();
        builtin_ver = gpu.ToolkitVersion;
		parallel.gpu.enableCUDAForwardCompatibility(tf);
	catch
        % Fails safely if no GPU is present on the system.
    end

    % Hunt for a user-installed CUDA toolkit.
    user_nvcc = '';
    
    % Strategy 1: Check standard Environment Variables.
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
    
    % Strategy 2: Fallback to querying the system PATH.
    if isempty(user_nvcc)
        if ispc
            [st, out] = system('where nvcc');
        else
            [st, out] = system('which nvcc');
        end
        if st == 0 && ~isempty(out)
            % 'where'/'which' might return multiple paths; take the first.
            paths = strsplit(strtrim(out), '\n');
            user_nvcc = paths{1}; 
        end
    end
	
	% Strategy 3: Fallback to common default installation paths.
    if isempty(user_nvcc)
        if ispc
            % Windows: Typically installs in versioned subfolders
            base_cuda_dir = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA';
            if exist(base_cuda_dir, 'dir')
                % Find all subfolders starting with 'v'
                cuda_versions = dir(fullfile(base_cuda_dir, 'v*'));
                if ~isempty(cuda_versions)
                    % Sort descending so the highest version string is checked first
                    [~, idx] = sort({cuda_versions.name}, 'descend');
                    cuda_versions = cuda_versions(idx);
                    
                    for i = 1:length(cuda_versions)
                        test_path = fullfile(base_cuda_dir, cuda_versions(i).name, 'bin', 'nvcc.exe');
                        if exist(test_path, 'file')
                            user_nvcc = test_path;
                            break;
                        end
                    end
                end
            end
        else
            % Linux: Typically symlinked to /usr/local/cuda or installed in /opt/cuda
            common_paths = {'/usr/local/cuda', '/opt/cuda'};
            for i = 1:length(common_paths)
                test_path = fullfile(common_paths{i}, 'bin', 'nvcc');
                if exist(test_path, 'file')
                    user_nvcc = test_path;
                    break;
                end
            end
        end
    end
    
    % Extract the specific version number from the found NVCC binary.
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
    
    % Prefer the user's CUDA toolkit if it is newer than MATLAB's.
    use_user_cuda = (user_ver > builtin_ver);
    
    disp('[Eiko] Compiling MEX extension for MATLAB... (This may take a minute)');
    if use_user_cuda
        fprintf('Detected user-installed CUDA (v%.1f).\n', user_ver);
    else
        fprintf('Using MATLAB''s built-in CUDA (v%.1f).\n', builtin_ver);
    end
    
    %% Compilation Flags & Architecture configuration
    
    if is_release
        % Fat binary formulation: embeds PTX and SASS for robust distribution.
        arch_flag = '-arch=all-major';
        disp('Release mode: Compiling fat binary (-arch=all-major).');
    else
        % Native formulation: compiles strictly for the current machine's GPU.
        arch_flag = '-arch=native';
    end

    % Universal flags applied to both Windows and Linux compilation paths.
    base_flags = [ ...
        '-std=c++17 ', ...         % C++17 ensures compatibility with Ubuntu 20.04's default GCC 9.4.
        '-DMATLAB_MEX_FILE ', ...  % Asserts to the headers that we are building a MEX file.
        '--use_fast_math ' ...     % Enables fast hardware approximations for trig, div, and sqrt.
    ];
    
    if is_release && ~ispc
        % CRITICAL LINUX FIX: Statically link the C++ standard library.
        % This prevents fatal "GLIBCXX_X.X.X not found" errors when a user tries 
        % to run the MEX file in a MATLAB version shipped with an older libstdc++.
        host_ldflags = {'LDFLAGS=$LDFLAGS -static-libstdc++'};
        disp('Release mode: Statically linking libstdc++ (Linux).');
    else
        host_ldflags = {};
    end
    
    if ispc
        % Windows-specific MSVC compiler overrides and safety bypasses.
        os_flags = [ ...
            '-allow-unsupported-compiler ', ...                % Bypasses strict Visual Studio version locks.
            '-Xcompiler "/Zc:preprocessor" ', ...              % Forces MSVC to use standard-compliant preprocessing.
            '-Xcompiler "/std:c++17" ', ...                    % Forces MSVC host compilation to C++17.
            '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH ', ... % Disables fatal errors on STL version mismatches.
            '-DNOMINMAX ' ...                                  % Prevents Windows.h from overwriting std::min/max.
        ];
    else
        os_flags = '';
    end
    
    % Host compiler flags strictly enforced on the `mex` linkage phase.
    host_cflags   = 'CFLAGS=$CFLAGS';
    host_cxxflags = 'CXXFLAGS=$CXXFLAGS -std=c++17';

    % Fallback configurations for the mexcuda pipeline.
    nvcc_arg_specific = ['NVCCFLAGS=', arch_flag, ' ', base_flags, os_flags];
    nvcc_arg_fallback = ['NVCCFLAGS=', base_flags, os_flags];
    
    %% Include Paths and Linker Directives
    
    % Gather all potential include directories required by MATLAB and CUDA.
    includes = { ...
        includeDir, ...
        fullfile(matlabroot, 'extern', 'include'), ...
        fullfile(matlabroot, 'toolbox', 'parallel', 'gpu', 'extern', 'include'), ...
        fullfile(matlabroot, 'toolbox', 'distcomp', 'gpu', 'extern', 'include') ... % Legacy distcomp fallback
    };
    
    % Silently filter out non-existent paths to prevent compiler warnings.
    includes = includes(cellfun(@(x) exist(x, 'dir') == 7, includes));
    includes_str = strjoin(cellfun(@(x) sprintf('-I"%s"', x), includes, 'UniformOutput', false), ' ');
    
    % Configure linker flags based on the operating system.
    if ispc
        cuda_lib_dir = fullfile(fileparts(fileparts(user_nvcc)), 'lib', 'x64');
        ml_lib_dir = fullfile(matlabroot, 'extern', 'lib', computer('arch'), 'microsoft');
        link_flags = {['-L' ml_lib_dir], ['-L' cuda_lib_dir], '-lmwgpu', '-lcudart'};
        fallback_libs = {'-lut'}; % libut allows MEX to detect MATLAB CTRL+C interrupts.
    else
        cuda_lib_dir = fullfile(fileparts(fileparts(user_nvcc)), 'lib64');
        link_flags = {['-L' cuda_lib_dir], '-lmwgpu', '-lcudart', '-ldl'}; % -ldl required for dlopen dynamics.
        fallback_libs = {'-lut', '-ldl'}; 
    end

    %% Execution Phase
    success = false;
    attempt = 1;
    
    if use_user_cuda
        fprintf('Attempt %d: Compiling with user-installed CUDA (bypassing mexcuda)...\n', attempt);
        
        % Step 1: Attempt to inject modern CCCL (Thrust, CUB, libcudacxx) if available.
        [cuda_bin_dir, ~, ~] = fileparts(user_nvcc);
        [cuda_root, ~, ~] = fileparts(cuda_bin_dir);
        
        cccl_include = fullfile(cuda_root, 'include', 'cccl');
        cccl_flags = '';
        if exist(cccl_include, 'dir')
            cccl_flags = sprintf('-I"%s" -I"%s" -I"%s" -I"%s" ', ...
                cccl_include, fullfile(cccl_include, 'thrust'), fullfile(cccl_include, 'libcudacxx'), fullfile(cccl_include, 'cub'));
        end
        
        % Step 2: Force NVCC to use MATLAB's specifically configured host compiler.
        % This is vital on Windows so NVCC can locate cl.exe without Developer Prompt environment vars.
        if ispc && ~isempty(getenv('IS_RELEASE_BUILD'))
			% We are on the GitHub Runner. Dynamically find the co-installed VS2019 (v142) compiler.
			msvc_base = 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC';
			
			% Search for the 14.29.x toolchain directory
			v142_dirs = dir(fullfile(msvc_base, '14.29.*'));
			if ~isempty(v142_dirs)
				% Grab the first match (usually only one exists)
				v142_version = v142_dirs(1).name;
				cc_bin_dir = fullfile(msvc_base, v142_version, 'bin', 'Hostx64', 'x64');
				ccbin_flag = sprintf('-ccbin "%s"', cc_bin_dir);
			else
				error('Could not locate the v142 (VS2019) build tools on the GitHub runner.');
			end
		else
			% We are on a local machine. Trust MATLAB's selected C++ compiler.
			cc_info = mex.getCompilerConfigurations('C++', 'Selected');
			if ~isempty(cc_info)
				cc_bin_dir = fileparts(cc_info(1).Details.CompilerExecutable);
				ccbin_flag = sprintf('-ccbin "%s"', cc_bin_dir);
			else
				ccbin_flag = '';
			end
		en
        
        % Step 3: Define output objects and Position Independent Code (PIC) flags.
        if ispc
            obj_ext = '.obj';
            pic_flag = '-Xcompiler "/MD"'; % Dynamic runtime linkage required by MSVC for MEX.
        else
            obj_ext = '.o';
            pic_flag = '-Xcompiler "-fPIC"'; % Required by GCC for shared library generation.
        end
        obj_file = fullfile(outDir, ['mex_bindings', obj_ext]);
        
        % Formulate the raw system NVCC command.
        nvcc_cmd = sprintf('"%s" %s -c "%s" -o "%s" %s %s %s %s %s %s', ...
            user_nvcc, ccbin_flag, sourceFile, obj_file, includes_str, cccl_flags, pic_flag, base_flags, os_flags, arch_flag);
        
        try
            % Execute device compilation (NVCC -> .o/.obj)
            [st, cmdout] = system(nvcc_cmd);
            if st == 0
                % Execute host linkage (MEX -> .mexw64/.mexa64)
                mex('-R2018a', host_cflags, host_cxxflags, host_ldflags{:}, obj_file, '-outdir', outDir, '-lut', link_flags{:});
                success = true;
            else
                fprintf('NVCC device compilation failed:\n%s\n', cmdout);
            end
        catch ME
            fprintf('Linkage failed due to: %s\n', ME.message);
        end
        
        % Clean up intermediate object file.
        if exist(obj_file, 'file')
            delete(obj_file);
        end
        attempt = attempt + 1;
    end
    
    if ~success
        try
            % Fallback 1: Try MATLAB's built-in CUDA toolkit with specific architecture.
            fprintf('Attempt %d: Compiling with MATLAB''s built-in CUDA (%s)...\n', attempt, arch_flag);
            mexcuda('-R2018a', host_cflags, host_cxxflags, host_ldflags{:}, nvcc_arg_specific, ['-I', includeDir], '-outdir', outDir, sourceFile, fallback_libs{:});
        catch ME
            fprintf('Compilation failed due to: %s\n', ME.message);
            try
                % Fallback 2: Try MATLAB's built-in CUDA toolkit with default architecture flags.
                fprintf('Attempt %d: Compiling with MATLAB''s built-in CUDA (default arch)...\n', attempt+1);
                mexcuda('-R2018a', host_cflags, host_cxxflags, host_ldflags{:}, nvcc_arg_fallback, ['-I', includeDir], '-outdir', outDir, sourceFile, fallback_libs{:});
            catch ME2
                fprintf('Compilation failed due to: %s\n', ME2.message);
                disp('Compilation failed. :(');
                rethrow(ME2);
            end
        end
    end

    if success
        disp(['MEX file saved to: ', outDir]);
        disp('Congratulations, you are now ready to use Eiko! :)');
        disp('Run "help eiko" to read the documentation.');
        addpath(eikoDir);
    end
end
