function success = bootstrap(package_version, full_mex_path)
    % BOOTSTRAP Checks for existing binaries, matches the current environment
    % against a local registry, downloads the appropriate zip archive,
    % verifies its SHA-256 hash, and extracts the precompiled MEX binaries.
    %
    % Arguments:
    %   package_version - String (e.g., '1.0.0')
    %   full_mex_path   - Full path to the target MEX file (e.g., '/path/to/bin/mex_bindings')
    %                     Extension is optional.
    %
    % Returns: true if binary is ready/downloaded, false if compilation fallback is needed.


    
    success = false;
    
    % Strip the extension if the user provided one, so we have the raw base name
    [target_dir, target_impl, ~] = fileparts(full_mex_path);
    if ~exist(target_dir, 'dir'), mkdir(target_dir); end

    %% Early Exit: Check if a valid MEX binary already exists
    % Code 3 specifically verifies that a valid MEX file exists on disk
    clean_target_path = fullfile(target_dir, target_impl);
    if exist(clean_target_path, 'file') == 3
        success = true;
        return;
    end

    fprintf('[Eiko] Downloading precompiled CUDA kernels... (This may take a minute)\n');

    %% 1. Load Local Registry
    script_dir = fileparts(mfilename('fullpath'));
    registry_path = fullfile(script_dir, 'registry.json');
    
    if ~exist(registry_path, 'file')
        fprintf('[Eiko] Local registry.json not found. Falling back to local compilation.\n');
        return;
    end

    try
        fid = fopen(registry_path, 'r', 'encoding', 'UTF-8');
        raw_json = fread(fid, '*char')';
        fclose(fid);
        registry = jsondecode(raw_json);
    catch ME
        fprintf('[Eiko] Failed to parse local registry (%s). Falling back to compilation.\n', ME.message);
        return;
    end

    %% 2. OS-Based Target Selection
    if ispc
        os_name = 'windows';
    else
        os_name = 'linux';
    end

    % Handle dynamic field parsing safely
    sanitized_version = ['x', replace(package_version, '.', '_')];
    if isfield(registry.versions, package_version)
        builds = registry.versions.(package_version);
    elseif isfield(registry.versions, sanitized_version)
        builds = registry.versions.(sanitized_version);
    else
        fprintf('[Eiko] No precompiled builds tracked for package version %s.\n', package_version);
        return;
    end

    matched_build = [];

    if strcmp(os_name, 'windows')
        % Windows: Single unified build
        for i = 1:numel(builds)
            if strcmp(builds(i).os, 'windows')
                matched_build = builds(i);
                break;
            end
        end
    else
        % Linux: Determine Ubuntu version to map to CUDA target
        target_cuda = '12.8'; % Default to 22.04+ baseline
        
        try
            % Check OS release natively (avoids missing lsb_release package issues)
            [status, cmdout] = system('cat /etc/os-release | grep "VERSION_ID"');
            if status == 0 && contains(cmdout, '20.04')
                target_cuda = '12.4';
            end
        catch
            % Silently fall back to default 12.8 target on error
        end
        
        % Search for the exact Linux + Target CUDA combination
        for i = 1:numel(builds)
            if strcmp(builds(i).os, 'linux') && strcmp(builds(i).cuda, target_cuda)
                matched_build = builds(i);
                break;
            end
        end
        
        % Safety fallback: If target CUDA build is missing, grab the first available Linux build
        if isempty(matched_build)
            for i = 1:numel(builds)
                if strcmp(builds(i).os, 'linux')
                    matched_build = builds(i);
                    break;
                end
            end
        end
    end

    if isempty(matched_build)
        fprintf('[Eiko] No precompiled binaries match OS: %s. Falling back to compilation.\n', os_name);
        return;
    end

    %% 3. Multiprocessing & Context-Safe Atomic Download
    zip_url = matched_build.url;
    tmp_cache_dir = fullfile(tempdir, 'eiko_cache');
    if ~exist(tmp_cache_dir, 'dir'), mkdir(tmp_cache_dir); end

    zip_path = fullfile(tmp_cache_dir, matched_build.filename);
    
    unique_id = char(java.util.UUID.randomUUID().toString());
    tmp_suffix = sprintf('.%s.tmp', unique_id(1:8));
    tmp_zip_path = [zip_path, tmp_suffix];

    % Scope the temporary extraction path for safety cleanup via onCleanup
    extract_tmp = fullfile(tmp_cache_dir, ['extract_', unique_id(1:8)]);
    cleanup_obj = onCleanup(@() cleanup_directory(extract_tmp, tmp_zip_path));

    try
        if ~exist(zip_path, 'file')
            import matlab.net.http.*
            
            % Enforce max size constraints cleanly using the RequestMessage API
            req = RequestMessage(RequestMethod.HEAD);
            resp = req.send(zip_url);
            if resp.StatusCode ~= StatusCode.OK
                error('Eiko:DownloadError', 'Remote server returned status code: %s', resp.StatusCode);
            end
            
            % Download the file using properly constructed WebOptions
            options = WebOptions('Timeout', 15, 'HeaderFields', {'User-Agent', 'eiko-bootstrap-matlab'});
            websave(tmp_zip_path, zip_url, options);
            
            % Verify SHA-256 Signature safely chunked
            if isfield(matched_build, 'sha256') && ~isempty(matched_build.sha256)
                if ~strcmpi(calc_sha256(tmp_zip_path), matched_build.sha256)
                    error('Eiko:HashMismatch', 'Hash mismatch verified! File may be corrupted.');
                end
            end
            
            % Transition archive atomically
            movefile(tmp_zip_path, zip_path, 'f');
        end

        %% Atomic Extraction Phase
        unzip(zip_path, extract_tmp);
        
        extracted_files = dir(fullfile(extract_tmp, '**', [target_impl, '.', mex_ext]));
        if isempty(extracted_files)
            error('Eiko:BinaryMissing', 'Target binary artifact was not found inside the zip payload.');
        end
        
        % Place binary files into destination
        for k = 1:numel(extracted_files)
            src_file = fullfile(extracted_files(k).folder, extracted_files(k).name);
            final_dest = fullfile(target_dir, extracted_files(k).name);
            tmp_dest = [final_dest, tmp_suffix];
            
            copyfile(src_file, tmp_dest, 'f');
            movefile(tmp_dest, final_dest, 'f'); 
        end
        
        success = true;

    catch ME
        fprintf('[Eiko] Download or deployment pipeline failed: %s. Falling back to local compilation.\n', ME.message);
        success = false;
    end
end

%% Helper Function for Scoped Cleanup
function cleanup_directory(dir_path, file_path)
    if exist(dir_path, 'dir'), rmdir(dir_path, 's'); end
    if exist(file_path, 'file'), delete(file_path); end
end

%% Optimized Helper Function for Verification (Chunked Stream)
function hex_hash = calc_sha256(file_path)
    md = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(file_path, 'r');
    
    % Stream file in chunks of 1MB to keep memory consumption low
    chunk_size = 1024 * 1024; 
    while ~feof(fid)
        data = fread(fid, chunk_size, '*uint8');
        if ~isempty(data)
            md.update(data);
        end
    end
    fclose(fid);
    
    hash = typecast(md.digest(), 'uint8');
    hex_hash = sprintf('%02x', hash);
end
