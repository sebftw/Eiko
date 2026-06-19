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
    
    % Check if MEX already exists (skip for release mode)
    if ~is_release && exist(config.OutFile, 'file') 
        % && eiko_lib.bootstrap(EIKO_VERSION, config.OutFile)
        return;
    end

    % Determine the output file name
    if is_release
        ml_release = version('-release'); % e.g., '2021b'
        target_name = sprintf('mex_bindings_R%s', ml_release);
    else
        target_name = 'mex_bindings';
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
    [os_flags, host_cflags, host_cxxflags, host_ldflags, pic_flag, obj_ext, ccbin_flag] = getOSFlags(best_nvcc, is_release);
    
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
        c_mexapi_obj = fullfile(config.OutDir, ['c_mexapi_version', obj_ext]);
        cccl_flags = getCCCLFlags(best_nvcc);
        [cuda_root, ~, ~] = fileparts(fileparts(best_nvcc));
        
        nvcc_cmd = sprintf('"%s" %s -c "%s" -o "%s" %s %s %s %s %s %s -I"%s"', ...
            best_nvcc, ccbin_flag, config.SourceFile, obj_file, includes_str, cccl_flags, pic_flag, base_flags, os_flags, arch_flag, fullfile(cuda_root, 'include'));
        
        vcvars_cmd = '';
        if ispc
            vcvars_cmd = getMSVCEnvironment();
        end

        if ~isempty(vcvars_cmd)
            nvcc_cmd = sprintf('%s && %s', vcvars_cmd, nvcc_cmd);
        end

        try
            % 1. Compile Device Code with NVCC
            [st, cmdout] = system(nvcc_cmd);
            if st ~= 0
                error('NVCC device compilation failed!\n%s', cmdout);
            end
            logMessage('NVCC compilation successful. Linking via direct system call...');
        
            % 2. Direct Linker Invocation
            out_file = fullfile(config.OutDir, [target_name, '.', mexext]);
            link_err_msg = [];
            
            try
                % Compile the mandatory MEX API version object
                compileMexApi(c_mexapi_obj, ispc, vcvars_cmd);
                
                % Build the system linker command
                link_cmd = buildDirectLinkCommand(obj_file, c_mexapi_obj, out_file, best_nvcc, ispc, matlabroot, computer('arch'), is_release);
                
                if ~isempty(vcvars_cmd)
                    link_cmd = sprintf('%s && %s', vcvars_cmd, link_cmd);
                end
                
                [st_link, cmdout_link] = system(link_cmd);
                if st_link ~= 0
                    error('Direct linker call failed:\n%s\nCommand was:\n%s', cmdout_link, link_cmd);
                end
            catch link_ME
                link_err_msg = link_ME.message;
            end
            
            % 3. Verify Output
           if exist(out_file, 'file') || (exist(['eiko_lib.', target_name], 'file') == 3)
                success = true;
                logMessage('Direct system linkage successful.');
            else
                if not(isempty(link_err_msg))
                    error('Direct linking failed: %s', link_err_msg);
                else
                    error('Linker ran but output file is missing.');
                end
            end
        
        catch ME
            warning('Compilation attempt %d aborted: %s', attempt, ME.message);
        end
        
        % Cleanup object files and linker byproducts
        if exist(obj_file, 'file'), delete(obj_file); end
        if exist(c_mexapi_obj, 'file'), delete(c_mexapi_obj); end
        if ispc
            % MSVC link.exe generates .lib and .exp files when building a DLL/MEX
            out_base = fullfile(config.OutDir, 'mex_bindings');
            if exist([out_base, '.lib'], 'file'), delete([out_base, '.lib']); end
            if exist([out_base, '.exp'], 'file'), delete([out_base, '.exp']); end
        end
        attempt = attempt + 1;
    end
    
    % Fallbacks using mexcuda
    if ~success
        try
            logMessage('Attempt %d: Compiling with MATLAB''s built-in CUDA (%s)...', attempt, arch_flag);
            mexcuda('-R2018a', host_cflags, host_cxxflags, host_ldflags{:}, nvcc_arg_specific, ...
                sprintf('-I"%s"', config.IncludeDir), '-outdir', config.OutDir, ...
                '-output', target_name, config.SourceFile, fallback_libs{:});
            success = true;
        catch ME
            logMessage('Compilation failed: %s', ME.message);
            try
                logMessage('Attempt %d: Compiling with built-in CUDA (default arch)...', attempt+1);
                mexcuda('-R2018a', host_cflags, host_cxxflags, host_ldflags{:}, nvcc_arg_fallback, ...
                    sprintf('-I"%s"', config.IncludeDir), '-outdir', config.OutDir, ...
                    '-output', target_name, config.SourceFile, fallback_libs{:});
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

function [os_flags, host_cflags, host_cxxflags, host_ldflags, pic_flag, obj_ext, ccbin_flag] = getOSFlags(~, is_release)
    host_cflags = 'CFLAGS=$CFLAGS';
    if ispc
        os_flags = '-allow-unsupported-compiler -Xcompiler "/Zc:preprocessor" -Xcompiler "/std:c++17" -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -DNOMINMAX ';
        host_cxxflags = 'COMPFLAGS="$COMPFLAGS /std:c++17"';
        host_ldflags = {};
        pic_flag = '-Xcompiler "/MD"';
        obj_ext = '.obj';
        
        ccbin_flag = '';
        
        % Try MATLAB's officially selected C++ compiler first
        [cc_info, is_valid] = getValidCpp17Compiler();
        if is_valid && ~isempty(cc_info(1).Details.CompilerExecutable)
            cl_dir = fileparts(cc_info(1).Details.CompilerExecutable);
            if ~isempty(cl_dir)
                ccbin_flag = sprintf('-ccbin "%s"', cl_dir);
            end
        end
        
        % Fall back to Environment Variables if still empty
        if isempty(ccbin_flag)
            msvc_root = getenv('VCToolsInstallDir');
            if ~isempty(msvc_root)
                cc_bin_dir = fullfile(msvc_root, 'bin', 'Hostx64', 'x64');
                if exist(fullfile(cc_bin_dir, 'cl.exe'), 'file')
                    ccbin_flag = sprintf('-ccbin "%s"', cc_bin_dir);
                end
            end
        end
    else
        os_flags = '';
        host_cxxflags = 'CXXFLAGS=$CXXFLAGS -std=c++17 ';
        
        % Conditionally strip symbols for the mexcuda fallback
        strip_ld = '';
        if is_release, strip_ld = ' -Wl,-s'; end
        host_ldflags = {['LDFLAGS=$LDFLAGS -static-libstdc++ -static-libgcc -Wl,--exclude-libs,libstdc++' strip_ld]};
        
        obj_ext = '.o';
        
        % Try MATLAB's officially selected C++ compiler first
        [cc_info, is_valid] = getValidCpp17Compiler();
        if is_valid
            cxx_path = cc_info(1).Details.CompilerExecutable;
            cc_path = strrep(cxx_path, 'g++', 'gcc'); % Best guess for the C equivalent
            
            host_ldflags = [{sprintf('GCC="%s"', cc_path)}, {sprintf('G++="%s"', cxx_path)}, host_ldflags];
            pic_flag = '-Xcompiler "-fPIC -static-libstdc++ -static-libgcc"'; 
            ccbin_flag = sprintf('-ccbin "%s"', fileparts(cxx_path));
            
        % Fallback to hardcoded gcc-10 (Safe for R2021b/R2022a on Ubuntu)
        elseif exist('/usr/bin/gcc-10', 'file') && exist('/usr/bin/g++-10', 'file')
            host_ldflags = [{'GCC=/usr/bin/gcc-10'}, {'G++=/usr/bin/g++-10'}, host_ldflags];
            pic_flag = '-ccbin "/usr/bin/g++-10" -Xcompiler "-fPIC -static-libstdc++ -static-libgcc"'; 
            ccbin_flag = '';
            
        % Ultimate Fallback (System default)
        else
            pic_flag = '-Xcompiler "-fPIC -static-libstdc++ -static-libgcc"'; 
            ccbin_flag = '';
        end
    end
end

function [cc_info, is_valid] = getValidCpp17Compiler()
    % Retrieves a MATLAB C++ compiler that supports C++17.
    % First checks the 'Selected' compiler. If invalid, scans 'Installed' compilers.
    
    cc_info = [];
    is_valid = false;
    
    % 1. Check the currently selected compiler
    selected = mex.getCompilerConfigurations('C++', 'Selected');
    if ~isempty(selected) && isCompilerValid(selected(1))
        cc_info = selected(1);
        is_valid = true;
        return;
    end
    
    % 2. If selected is invalid or missing, search all installed compilers
    installed = mex.getCompilerConfigurations('C++', 'Installed');
    for i = 1:length(installed)
        if isCompilerValid(installed(i))
            cc_info = installed(i);
            is_valid = true;
            % fprintf('[Eiko] Selected compiler too old/missing. Automatically routing to installed %s (v%s).\n', cc_info.Name, cc_info.Version);
            return;
        end
    end
end

function valid = isCompilerValid(comp_obj)
    % Evaluates if a given MATLAB compiler object supports C++17
    valid = false;
    tok = regexp(comp_obj.Version, '^(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        major_ver = str2double(tok{1});
        if ispc && (major_ver >= 15)
            valid = true; % MSVC 15.0 (VS2017) or newer
        elseif ~ispc && (major_ver >= 7)
            valid = true; % GCC 7 or newer
        end
    end
end

function vcvars_cmd = getMSVCEnvironment()
    vcvars_cmd = '';
    [cc_info, is_valid] = getValidCpp17Compiler();
    
    if is_valid
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

function compileMexApi(c_mexapi_obj, is_pc, vcvars_cmd)
    % Compiles MATLAB's required API version object file
    c_mexapi_src = fullfile(matlabroot, 'extern', 'version', 'c_mexapi_version.c');
    ml_inc_dir = fullfile(matlabroot, 'extern', 'include');
    
    if is_pc
        % Added /I"%s" to point to the MATLAB extern/include directory
        api_cmd = sprintf('cl.exe /c /nologo /MD /O2 /I"%s" /Fo"%s" "%s"', ml_inc_dir, c_mexapi_obj, c_mexapi_src);
        if ~isempty(vcvars_cmd), api_cmd = sprintf('%s && %s', vcvars_cmd, api_cmd); end
    else
        api_cmd = sprintf('gcc -c -fPIC -I"%s" "%s" -o "%s"', ml_inc_dir, c_mexapi_src, c_mexapi_obj);
    end
    
    [st, cmdout] = system(api_cmd);
    if st ~= 0
        error('Failed to compile c_mexapi_version.c:\n%s', cmdout);
    end
end

function link_cmd = buildDirectLinkCommand(obj_file, c_mexapi_obj, out_file, best_nvcc, is_pc, ml_root, ml_arch, is_release)
    % Constructs the raw CLI link string, bypassing the MATLAB mex function
    if is_pc
        ml_lib_dir = fullfile(ml_root, 'extern', 'lib', ml_arch, 'microsoft');
        cuda_lib_dir = fullfile(fileparts(fileparts(best_nvcc)), 'lib', 'x64');
        
        libs = sprintf('"%s\\libmx.lib" "%s\\libmex.lib" "%s\\libmat.lib" "%s\\libMatlabDataArray.lib" "%s\\libut.lib" "%s\\cudart_static.lib" ', ...
            ml_lib_dir, ml_lib_dir, ml_lib_dir, ml_lib_dir, ml_lib_dir, cuda_lib_dir);
            
        if exist(fullfile(ml_lib_dir, 'gpu.lib'), 'file'), libs = [libs, sprintf('"%s\\gpu.lib" ', ml_lib_dir)]; end
        if exist(fullfile(ml_lib_dir, 'mwgpu.lib'), 'file'), libs = [libs, sprintf('"%s\\mwgpu.lib" ', ml_lib_dir)]; end
        if exist(fullfile(ml_lib_dir, 'gpumexbinder.lib'), 'file'), libs = [libs, sprintf('"%s\\gpumexbinder.lib" ', ml_lib_dir)]; end
        
        % Optimize and strip unreferenced symbols for release mode
        strip_flag = '';
        if is_release, strip_flag = '/RELEASE /OPT:REF /OPT:ICF '; end
        
        % Ask MATLAB for the correct linker executable
        link_exec = 'link.exe';
        [cc_info, is_valid] = getValidCpp17Compiler();
        if is_valid && ~isempty(cc_info(1).Details.CompilerExecutable)
            cl_path = cc_info(1).Details.CompilerExecutable;
            link_path = fullfile(fileparts(cl_path), 'link.exe');
            if exist(link_path, 'file')
                link_exec = sprintf('"%s"', link_path);
            end
        end
        
        % MSVC uses /EXPORT:mexFunction to expose the correct entrypoint to MATLAB
        link_cmd = sprintf('%s /DLL /nologo %s/OUT:"%s" /EXPORT:mexFunction "%s" "%s" %s', link_exec, strip_flag, out_file, obj_file, c_mexapi_obj, libs);
    else
        ml_bin = fullfile(ml_root, 'bin', ml_arch);
        ml_ext_bin = fullfile(ml_root, 'extern', 'bin', ml_arch);
        ml_ext_lib = fullfile(ml_root, 'extern', 'lib', ml_arch);
        cuda_lib_dir = fullfile(fileparts(fileparts(best_nvcc)), 'lib64');
        
        % The map file specifies exactly what symbols to export
        map_file = fullfile(ml_ext_lib, 'c_exportsmexfileversion.map');
        
        % Extracted directly from the Makefile template to fix glibc & rpath issues
        ldflags = sprintf('-pthread -shared -O -Wl,--no-undefined -Wl,--version-script,"%s" -Wl,--as-needed ', map_file);
        ldflags = [ldflags, sprintf('-Wl,-rpath-link,"%s" -L"%s" ', ml_bin, ml_bin)];
        ldflags = [ldflags, sprintf('-Wl,-rpath-link,"%s" -L"%s" ', ml_ext_bin, ml_ext_bin)];
        
        libs = '-lMatlabDataArray -lmx -lmex -lmat -lm -lc -lut ';
        if exist(fullfile(ml_bin, 'libmwgpu.so'), 'file'), libs = [libs, '-lmwgpu ']; end
        if exist(fullfile(ml_bin, 'libgpu.so'), 'file'), libs = [libs, '-lgpu ']; end
        if exist(fullfile(ml_bin, 'libgpumexbinder.so'), 'file'), libs = [libs, '-lgpumexbinder ']; end
        
        cuda_libs = sprintf('-L"%s" -lcudart_static -ldl -lrt ', cuda_lib_dir);
        
        % Static glibc linking
        static_libs = '-static-libstdc++ -static-libgcc ';
        
        % Strip symbols for release mode
        strip_flag = '';
        if is_release, strip_flag = '-s '; end
        
        % Ask MATLAB for the correct linker executable
        cxx_exec = 'g++';
        [cc_info, is_valid] = getValidCpp17Compiler();
        if is_valid
            cxx_exec = sprintf('"%s"', cc_info(1).Details.CompilerExecutable);
        end
        
        % Use the selected compiler explicitly to link the final binary
        link_cmd = sprintf('%s %s"%s" "%s" %s %s %s %s -o "%s"', cxx_exec, strip_flag, obj_file, c_mexapi_obj, ldflags, libs, cuda_libs, static_libs, out_file);
    end
end
