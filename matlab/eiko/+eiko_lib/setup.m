function ver_out = setup(build_type)
    % SETUP Compiles and installs Eiko CUDA MEX bindings for MATLAB.
    %
    % Usage:
    %   setup()          - Compiles for the local GPU (-arch=native).
    %   setup('native')  - Compiles for the local GPU (-arch=native).
    %   setup('release') - Compiles a fat binary (-arch=all-major).
    %   setup('version') - Returns the current version of Eiko.
    
    if nargin < 1
        build_type = 'native';
    else
        build_type = validatestring(lower(build_type), {'native', 'release', 'version'});
    end
    
    EIKO_VERSION = '0.8.5';
    
    if strcmpi(build_type, 'version')
        ver_out = EIKO_VERSION;
        return;
    end
    
    is_release = strcmpi(build_type, 'release');
    config = eiko_lib.BuildConfig.getInstance();
    
    % Download MEX if it already exists (skip for release mode)
    if ~is_release && eiko_lib.bootstrap(EIKO_VERSION, config.OutFile)
        return;
    end
    
    %% 1. CUDA Setup & Version Detection
    logMessage('Compiling MEX extension for MATLAB... (This may take a minute)');
    
    [best_nvcc, nvcc_ver] = findBestNVCC();
    
    if ~isempty(best_nvcc)
        if contains(best_nvcc, matlabroot)
            logMessage('Detected MATLAB-shipped NVCC (v%.1f).', nvcc_ver);
        else
            logMessage('Detected user-installed CUDA (v%.1f).', nvcc_ver);
        end
    else
        logMessage('No standalone NVCC found. Relying strictly on mexcuda fallbacks.');
    end
    
    %% 2. Compilation Flags & Architecture Configuration
    arch_flag = '-arch=native';
    if is_release
        arch_flag = '-arch=all-major';
        logMessage('Release mode: Compiling fat binary (-arch=all-major).');
    end
    
    base_flags = '-std=c++17 -DMATLAB_MEX_FILE --use_fast_math ';
    
    % OS-Specific Flags
    [os_flags, host_cflags, host_cxxflags, host_ldflags, pic_flag, obj_ext, ccbin_flag] = getOSFlags(best_nvcc);
    
    nvcc_arg_specific = ['NVCCFLAGS=', arch_flag, ' ', base_flags, os_flags];
    nvcc_arg_fallback = ['NVCCFLAGS=', base_flags, os_flags];
    
    %% 3. Include Paths and Linker Directives
    includes = { config.IncludeDir, ...
                 fullfile(matlabroot, 'extern', 'include'), ...
                 fullfile(matlabroot, 'toolbox', 'parallel', 'gpu', 'extern', 'include'), ...
                 fullfile(matlabroot, 'toolbox', 'distcomp', 'gpu', 'extern', 'include') };
    
    % Filter existent paths
    includes = includes(cellfun(@(x) exist(x, 'dir') == 7, includes));
    includes_str = strjoin(cellfun(@(x) sprintf('-I"%s"', x), includes, 'UniformOutput', false), ' ');
    
    [link_flags, fallback_libs] = getLinkerFlags(best_nvcc);

    %% 4. Execution Phase
    success = false;
    attempt = 1;
    
    if ~isempty(best_nvcc)
        logMessage('Attempt %d: Compiling with NVCC...', attempt);
        
        obj_file = fullfile(config.OutDir, ['mex_bindings', obj_ext]);
        cccl_flags = getCCCLFlags(best_nvcc);
        [cuda_root, ~, ~] = fileparts(fileparts(best_nvcc));
        
        nvcc_cmd = sprintf('"%s" %s -c "%s" -o "%s" %s %s %s %s %s %s -I"%s"', ...
            best_nvcc, ccbin_flag, config.SourceFile, obj_file, includes_str, cccl_flags, pic_flag, base_flags, os_flags, arch_flag, fullfile(cuda_root, 'include'));
        
        if ispc
            vcvars_cmd = getMSVCEnvironment();
            if ~isempty(vcvars_cmd)
                logMessage('Activating MSVC environment.');
                nvcc_cmd = sprintf('%s && %s', vcvars_cmd, nvcc_cmd);
            end
        end

        try
            % Compile with NVCC
            [st, cmdout] = system(nvcc_cmd);
            if st ~= 0
                error('NVCC device compilation failed!\n%s', cmdout);
            end
            logMessage('NVCC compilation successful. Linking MEX...');
        
            % Link with MEX
            try
                mex('-R2018a', host_cflags, host_cxxflags, host_ldflags{:}, obj_file, ...
                    '-outdir', config.OutDir, '-lut', link_flags{:});
            catch mex_ME
                % MATLAB occasionally throws a spurious "... is not a MEX file" error 
                % even when it succeeds. We catch it here silently and rely on the 
                % file existence check below as the ultimate source of truth.
            end
            
            if (exist(fullfile(config.OutDir, ['mex_bindings.', mexext]), 'file') == 3) || (exist('eiko_lib.mex_bindings', 'file') == 3)
                success = true;
                logMessage('MEX compilation successful.');
            else
                % If the file is missing, figure out why and throw the appropriate error
                if exist('mex_ME', 'var')
                    error('MEX compilation failed: %s', mex_ME.message);
                else
                    error('MEX run finished but output file is missing.');
                end
            end
        
        catch ME
            % Handle genuine, unrecoverable errors (like NVCC failing)
            warning('Compilation attempt %d aborted: %s', attempt, ME.message);
        end
        
        if exist(obj_file, 'file'), delete(obj_file); end
        attempt = attempt + 1;
    end
    
    % Fallbacks using mexcuda
    if ~success
        try
            logMessage('Attempt %d: Compiling with MATLAB''s built-in CUDA (%s)...', attempt, arch_flag);
            mexcuda('-R2018a', host_cflags, host_cxxflags, host_ldflags{:}, nvcc_arg_specific, sprintf('-I"%s"', config.IncludeDir), '-outdir', config.OutDir, config.SourceFile, fallback_libs{:});
            success = true;
        catch ME
            logMessage('Compilation failed: %s', ME.message);
            try
                logMessage('Attempt %d: Compiling with built-in CUDA (default arch)...', attempt+1);
                mexcuda('-R2018a', host_cflags, host_cxxflags, host_ldflags{:}, nvcc_arg_fallback, sprintf('-I"%s"', config.IncludeDir), '-outdir', config.OutDir, config.SourceFile, fallback_libs{:});
                success = true;
            catch ME2
                logMessage('Compilation failed.');
                rethrow(ME2);
            end
        end
    end
    
    if success
        logMessage('MEX file saved to: %s\n', config.OutDir);
        disp('Congratulations, you are now ready to use Eiko! :)');
        disp('Run "help eiko" to read the documentation.');
        addpath(config.EikoDir);
    end
end

%% --- Local Subroutines ---

function logMessage(msg, varargin)
    fprintf(['[Eiko] ' msg '\n'], varargin{:});
end

function [best_nvcc, max_ver] = findBestNVCC()
    candidates = {};
    ext = ''; if ispc, ext = '.exe'; end
    
    % Strategy 1: Built-in MATLAB Shipped NVCC
    arch = computer('arch');
    matlab_paths = {
        fullfile(matlabroot, 'sys', 'cuda', arch, 'cuda', 'bin', ['nvcc' ext]), ...
        fullfile(matlabroot, 'toolbox', 'parallel', 'gpu', 'extern', 'bin', ['nvcc' ext]), ...
        fullfile(matlabroot, 'bin', arch, ['nvcc' ext])
    };
    for i = 1:length(matlab_paths)
        if exist(matlab_paths{i}, 'file')
            candidates{end+1} = matlab_paths{i}; %#ok<AGROW>
        end
    end
    
    % Strategy 2: Environment Variables
    env_vars = {'CUDA_PATH', 'CUDA_HOME'};
    for i = 1:length(env_vars)
        p = getenv(env_vars{i});
        if ~isempty(p)
            test_path = fullfile(p, 'bin', ['nvcc' ext]);
            if exist(test_path, 'file')
                candidates{end+1} = test_path; %#ok<AGROW>
            end
        end
    end
    
    % Strategy 3: System Path
    cmd = 'which nvcc'; if ispc, cmd = 'where nvcc'; end
    [st, out] = system(cmd);
    if st == 0 && ~isempty(out)
        paths = splitlines(strtrim(out));
        for i = 1:length(paths)
            if exist(paths{i}, 'file')
                candidates{end+1} = paths{i}; %#ok<AGROW>
            end
        end
    end
    
    % Strategy 4: Common Default Paths
    if ispc
        base = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA';
        if exist(base, 'dir')
            cuda_versions = dir(fullfile(base, 'v*'));
            for i = 1:length(cuda_versions)
                test_path = fullfile(base, cuda_versions(i).name, 'bin', 'nvcc.exe');
                if exist(test_path, 'file')
                    candidates{end+1} = test_path; %#ok<AGROW>
                end
            end
        end
    else
        bases = {'/usr/local', '/opt'};
        for b = 1:length(bases)
            if exist(bases{b}, 'dir')
                cuda_folders = dir(fullfile(bases{b}, 'cuda*'));
                for i = 1:length(cuda_folders)
                    test_path = fullfile(bases{b}, cuda_folders(i).name, 'bin', 'nvcc');
                    if exist(test_path, 'file')
                        candidates{end+1} = test_path; %#ok<AGROW>
                    end
                end
            end
        end
    end
    
    candidates = unique(candidates);
    best_nvcc = '';
    max_ver = 0;
    
    % Evaluate all candidates; best match wins.
    for i = 1:length(candidates)
        current_path = candidates{i};
        [st, out] = system(['"' current_path '" --version']);
        if st == 0
            tok = regexp(out, 'release (\d+\.\d+)', 'tokens');
            if ~isempty(tok)
                current_ver = str2double(tok{1}{1});
                if current_ver > max_ver
                    max_ver = current_ver;
                    best_nvcc = current_path;
                end
            end
        end
    end
end

function [os_flags, host_cflags, host_cxxflags, host_ldflags, pic_flag, obj_ext, ccbin_flag] = getOSFlags(~)
    host_cflags = 'CFLAGS=$CFLAGS';
    if ispc
        os_flags = '-allow-unsupported-compiler -Xcompiler "/Zc:preprocessor" -Xcompiler "/std:c++17" -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -DNOMINMAX ';
        host_cxxflags = 'COMPFLAGS="$COMPFLAGS /std:c++17"';
        host_ldflags = {};
        pic_flag = '-Xcompiler "/MD"';
        obj_ext = '.obj';
        
        msvc_root = getenv('VCToolsInstallDir');
        ccbin_flag = '';
        if ~isempty(msvc_root)
            cc_bin_dir = fullfile(msvc_root, 'bin', 'Hostx64', 'x64');
            if exist(fullfile(cc_bin_dir, 'cl.exe'), 'file')
                ccbin_flag = sprintf('-ccbin "%s"', cc_bin_dir);
            end
        end
    else
        os_flags = '';
        host_cxxflags = 'CXXFLAGS=$CXXFLAGS -std=c++17 ';
        host_ldflags = {'LDFLAGS=$LDFLAGS -static-libstdc++ -static-libgcc -Wl,--exclude-libs,libstdc++ -Wl,-s'};
        obj_ext = '.o';
        
        if exist('/usr/bin/gcc-10', 'file') && exist('/usr/bin/g++-10', 'file')
            host_ldflags = [{'GCC=/usr/bin/gcc-10'}, {'G++=/usr/bin/g++-10'}, host_ldflags];
            pic_flag = '-ccbin "/usr/bin/g++-10" -Xcompiler "-fPIC -static-libstdc++ -static-libgcc"'; 
            ccbin_flag = '';
        else
            pic_flag = '-Xcompiler "-fPIC -static-libstdc++ -static-libgcc"'; 
            
            cc_info = mex.getCompilerConfigurations('C++', 'Selected');
            if ~isempty(cc_info)
                ccbin_flag = sprintf('-ccbin "%s"', fileparts(cc_info(1).Details.CompilerExecutable));
            else
                ccbin_flag = '';
            end
        end
    end
end

function vcvars_cmd = getMSVCEnvironment()
    vcvars_cmd = '';
    cc_info = mex.getCompilerConfigurations('C++', 'Selected');
    
    if ~isempty(cc_info)
        details = cc_info(1).Details;

        % Safe check for both older (struct) and newer (object) MATLAB versions
        has_shell = (isstruct(details) && isfield(details, 'CommandLineShell')) || ...
                    (isobject(details) && isprop(details, 'CommandLineShell'));

        if has_shell && ~isempty(details.CommandLineShell)
            base = details.CommandLineShell;
            
            has_args = (isstruct(details) && isfield(details, 'CommandLineShellArg')) || ...
                       (isobject(details) && isprop(details, 'CommandLineShellArg'));
            if has_args && ~isempty(details.CommandLineShellArg)
                vcvars_cmd = sprintf('"%s" %s', base, details.CommandLineShellArg);
            else
                if startsWith(base, '"')
                    vcvars_cmd = base;
                else
                    vcvars_cmd = sprintf('"%s"', base);
                end
            end
        end
    end
    
    % Fallback to VCToolsInstallDir if the above failed.
    if isempty(vcvars_cmd)
        msvc_root = getenv('VCToolsInstallDir');
        if ~isempty(msvc_root)
            idx = strfind(msvc_root, fullfile('VC', 'Tools', 'MSVC'));
            if ~isempty(idx)
                vcvars_path = fullfile(msvc_root(1:idx-1), 'VC', 'Auxiliary', 'Build', 'vcvars64.bat');
                if exist(vcvars_path, 'file'), vcvars_cmd = sprintf('"%s"', vcvars_path); end
            end
        end
    end
end

function [link_flags, fallback_libs] = getLinkerFlags(best_nvcc)
    if ispc
        cuda_lib_dir = fullfile(fileparts(fileparts(best_nvcc)), 'lib', 'x64');
        ml_lib_dir = fullfile(matlabroot, 'extern', 'lib', computer('arch'), 'microsoft');
        link_flags = {['-L' ml_lib_dir], ['-L' cuda_lib_dir], '-lcudart_static'};
        
        if exist(fullfile(ml_lib_dir, 'gpu.lib'), 'file'), link_flags{end+1} = '-lgpu'; end
        if exist(fullfile(ml_lib_dir, 'mwgpu.lib'), 'file'), link_flags{end+1} = '-lmwgpu'; end
        if exist(fullfile(ml_lib_dir, 'gpumexbinder.lib'), 'file'), link_flags{end+1} = '-lgpumexbinder'; end
        fallback_libs = {'-lut'};
    else
        cuda_lib_dir = fullfile(fileparts(fileparts(best_nvcc)), 'lib64');
        ml_lib_dir = fullfile(matlabroot, 'bin', computer('arch'));
        link_flags = {['-L' cuda_lib_dir], '-lcudart_static', '-ldl', '-lrt'}; 
        
        if exist(fullfile(ml_lib_dir, 'libgpu.so'), 'file'), link_flags{end+1} = '-lgpu'; end
        if exist(fullfile(ml_lib_dir, 'libmwgpu.so'), 'file'), link_flags{end+1} = '-lmwgpu'; end
        if exist(fullfile(ml_lib_dir, 'libgpumexbinder.so'), 'file'), link_flags{end+1} = '-lgpumexbinder'; end
        fallback_libs = {'-lut', '-ldl', '-lrt'};
    end
end

function cccl_flags = getCCCLFlags(best_nvcc)
    cccl_flags = '';
    cccl_include = getenv('CCCL_ROOT');
    if isempty(cccl_include) || ~exist(cccl_include, 'dir')
        [cuda_root, ~, ~] = fileparts(fileparts(best_nvcc));
        cccl_include = fullfile(cuda_root, 'include', 'cccl');
    end
    if exist(cccl_include, 'dir')
        cccl_flags = sprintf('-I"%s" -I"%s" -I"%s" -I"%s" ', ...
            cccl_include, fullfile(cccl_include, 'thrust'), fullfile(cccl_include, 'libcudacxx'), fullfile(cccl_include, 'cub'));
    end
end
