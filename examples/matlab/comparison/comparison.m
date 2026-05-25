% =========================================================================
% MATLAB Eikonal Solver Performance Comparison
% =========================================================================
% Compares performance (FPS) and accuracy of different methods for solving 
% the 2D Eikonal equation.
%
% Libraries tested:
%   1. eiko (Eiko library)
%   2. bwdist (Built-in Image Processing Toolbox)
%   3. msfm2d (Accurate Fast Marching by Dirk-Jan Kroon)
%   4. perform_fast_marching (Toolbox Fast Marching by Gabriel Peyré)
%   5. fsm_2d (Custom pure MATLAB Fast Sweeping Method)
% =========================================================================

function comparison()
    % --- Configuration ---
    BATCH_SIZE = 128;
    GRID_SIZE = 256; % Using 256 instead of 512 so pure MATLAB FSM completes
    NUM_RUNS = 20;
    NUM_RUNS_CPU = 1;  % Batching already means it runs a lot of times on CPU.

    device = gpuDevice;
    fprintf('GPU Device: %s\n', device.Name);
    fprintf('-----------------------------------------------------------------\n');
    fprintf('2D Benchmarking with Batch: %d, Grid: %d^2, Runs: %d (%d for CPU)\n', BATCH_SIZE, GRID_SIZE, NUM_RUNS, NUM_RUNS_CPU);
    fprintf('-----------------------------------------------------------------\n');
    
    % --- Data Generation ---
    % 1. Slowness field f(x) = 1 (Uniform speed)
    f_grid = ones(GRID_SIZE, GRID_SIZE);
    
    % 2. Source initialization (Center of grid)
    src_idx = floor(GRID_SIZE / 2) + 1; % +1 for 1-based indexing
    
    % Masks for bwdist and FMM
    bw_mask = zeros(GRID_SIZE, GRID_SIZE, 'single');
    bw_mask(src_idx, src_idx) = true;
    
    % Source array for Peyre's FMM: [x; y]
    start_points = [src_idx; src_idx];

    % Initial distance field 'u' for eiko
    u_init = inf(GRID_SIZE, GRID_SIZE);
    u_init(src_idx, src_idx) = 0;
    
    % Data storage for results
    fps_results = struct();
    outputs = struct();
    error_log = {};

    % --- BENCHMARK 1: eiko ---
    u_init_gpu = single(gpuArray(u_init));
    f_grid_gpu = single(gpuArray(f_grid));
    try
        [fps, out] = run_benchmark('eiko', @() run_eiko(u_init_gpu, f_grid_gpu), NUM_RUNS, BATCH_SIZE, true);
        fps_results.eiko = fps;
        outputs.eiko = out;
    catch ME
        error_log{end+1} = {'eiko', ME.message};
    end

    % --- BENCHMARK 2: bwdist (Built-in) ---
    % Note: bwdist only works for uniform speed (f=1).
    try
        bw_mask_gpu = gpuArray(bw_mask);
        [fps, out] = run_benchmark('bwdist (f=1 only)', @() run_bwdist(bw_mask_gpu, BATCH_SIZE), NUM_RUNS_CPU, BATCH_SIZE, true);
        fps_results.bwdist = fps;
        outputs.bwdist = out;
    catch ME
        error_log{end+1} = {'bwdist', ME.message};
    end

    fprintf('-----------------------------------------------------------------\n');

    % --- BENCHMARK 3: msfm (Dirk-Jan Kroon) ---
    if true
        % Does not seem to work on Windows.
        if exist('msfm', 'file') == 3 || exist('msfm', 'file') == 2
            try
                speed_map = 1./f_grid;
                [fps, out] = run_benchmark('msfm', @() run_msfm2d(speed_map, src_idx, src_idx, BATCH_SIZE), NUM_RUNS_CPU, BATCH_SIZE);
                fps_results.msfm = fps;
                outputs.msfm = out;
            catch ME
                error_log{end+1} = {'msfm', ME.message};
            end
        else
            fprintf('%22s | %31s\n', 'msfm2d', 'SKIPPED (Not found on path)');
        end
    end
    
    % --- BENCHMARK 4: Toolbox Fast Marching (Gabriel Peyre) ---
    if true
        if exist('perform_fast_marching', 'file') == 2 || exist('perform_fast_marching', 'file') == 3
            try
                [fps, out] = run_benchmark('Toolbox FMM', @() run_peyre(f_grid, start_points, BATCH_SIZE), NUM_RUNS_CPU, BATCH_SIZE);
                fps_results.ToolboxFMM = fps;
                outputs.ToolboxFMM = out * GRID_SIZE;  % Its output must be scaled up.
            catch ME
                error_log{end+1} = {'Toolbox FMM', ME.message};
            end
        else
            fprintf('%22s | %31s\n', 'Toolbox FMM', 'SKIPPED (Not found on path)');
        end
    end

    fprintf('-----------------------------------------------------------------\n');
    
    % --- Error Logging ---
    if ~isempty(error_log)
        fprintf('\n=================================================================\n');
        fprintf('ERROR LOG\n');
        fprintf('=================================================================\n');
        for i = 1:length(error_log)
            fprintf('[%s]\n  %s\n\n', error_log{i}{1}, error_log{i}{2});
        end
        fprintf('=================================================================\n');
    end

    % --- Analytical Ground Truth ---
    [X, Y] = meshgrid(1:GRID_SIZE, 1:GRID_SIZE);
    truth = sqrt((X - src_idx).^2 + (Y - src_idx).^2);

    % --- Plotting ---
    plot_fps(fps_results);
    plot_errors(outputs, truth);
end

% ==========================================
% Benchmark Execution Engine
% ==========================================
function [fps, first_output] = run_benchmark(name, target_func, runs, batch_size, is_gpu)
    if nargin < 5 || isempty(is_gpu)
        is_gpu = false;
    end

    % Warmup
    for i = 1:2
        out = target_func();
    end
    if iscell(out)
        first_output = out{1}; % Capture one output for accuracy plotting
    else
        first_output = out(:, :, 1);
    end
    
    if is_gpu
        wait(gpuDevice());
    end

    % Timing
    total_time = 0;
    for i = 1:runs
        if is_gpu
            total_time = total_time + gputimeit(target_func, 1);
        else
            total_time = total_time + timeit(target_func, 1);
        end
    end
    
    avg_time_ms = (total_time / runs) * 1000;
    fps = (batch_size * 1000) / avg_time_ms;
    
    fprintf('%22s | %8.3f ms/batch | %10.1f frames/s\n', name, avg_time_ms, fps);
end

% ==========================================
% Solver Wrappers
% ==========================================
function out = run_bwdist(bw_mask, batch_size)
    out = cell(batch_size, 1);
    for b = 1:batch_size
        out{b} = bwdist(bw_mask, 'euclidean');
    end
end

function out = run_msfm2d(f_grid, src_x, src_y, batch_size)
    out = cell(batch_size, 1);
    for b = 1:batch_size
        % msfm uses True/False or continuous maps. It requires MEX compilation.
        out{b} = msfm(f_grid, [src_y; src_x], false, false);  % second order = false, multi stencil = false.
    end
end

function out = run_peyre(f_grid, start_points, batch_size)
    out = cell(batch_size, 1);
    for b = 1:batch_size
        options = struct();
        [out{b}, ~] = perform_fast_marching(f_grid, start_points, options);
    end
end

function out = run_eiko(u_init, f_grid)
    out = eiko(u_init, f_grid);
end

% ==========================================
% Plotting Functions
% ==========================================
function plot_fps(fps_results)
    fields = fieldnames(fps_results);
    if isempty(fields); return; end
    
    vals = zeros(length(fields), 1);
    for i = 1:length(fields)
        vals(i) = fps_results.(fields{i});
    end
    
    % Sort for clean presentation
    [vals, sort_idx] = sort(vals, 'ascend');
    fields = fields(sort_idx);
    
    fig = figure('Name', 'FPS Comparison', 'Position', [100, 100, 800, 400], 'Color', 'w');
    
    % Modern barh styling
    b = barh(vals, 'FaceColor', [0.1608, 0.5020, 0.7255], 'EdgeColor', 'none', 'BarWidth', 0.6);
    
    ax = gca;
    ax.YTick = 1:length(fields);
    ax.YTickLabel = strrep(fields, '_', ' ');
    ax.FontName = 'Arial';
    ax.FontSize = 11;
    ax.XColor = [0.5, 0.5, 0.5];
    
    % Despine
    box off;
    % ax.XRuler.Axle.LineStyle = 'none';
    ax.YColor = [0.17, 0.24, 0.31];
    
    % Background grid
    grid on;
    ax.XGrid = 'on';
    ax.YGrid = 'off';
    ax.GridColor = [0.74, 0.76, 0.78];
    ax.GridAlpha = 0.5;
    ax.GridLineStyle = '--';
    
    xlabel('Processing Rate (Frames per Second)', 'FontWeight', 'bold');
    title('Eikonal Solver Speed Comparison (MATLAB)', 'FontWeight', 'bold', 'FontSize', 14);
    
    % Inline data labels
    for i = 1:length(vals)
        text(vals(i) + (max(vals)*0.01), i, sprintf('%.0f', round(vals(i))), ...
             'VerticalAlignment', 'middle', 'FontWeight', 'bold', ...
             'Color', [0.17, 0.24, 0.31], 'FontSize', 10, 'Clipping', 'off');
    end
    drawnow;

    exportgraphics(fig, 'fps_comparison.png', 'Resolution', 300);  % 300 DPI for high quality.
end

function plot_errors(outputs, truth)
    fields = fieldnames(outputs);
    if isempty(fields); return; end
    
    num_plots = length(fields);
    cols = min(3, num_plots);
    rows = ceil(num_plots / cols);
    
    fig = figure('Name', 'Absolute Error Comparison', 'Position', [150, 150, 400*cols, 350*rows], 'Color', 'w');
    
    % Using 'magma' equivalent in MATLAB. R2018b+ supports python colormaps.
    % If older, parula is standard.
    
    for i = 1:num_plots
        subplot(rows, cols, i);
        solver_name = fields{i};
        
        out_arr = gather(outputs.(solver_name));
        err = abs(out_arr - truth);
        
        imagesc(err);
        axis image; axis off;
        
        % Try to set a modern colormap
        try; colormap(gca, 'magma'); catch; colormap(gca, 'parula'); end
        
        cb = colorbar;
        cb.Label.String = 'Error (Distance)';
        
        title(strrep(solver_name, '_', ' '), 'FontWeight', 'bold');
    end

    exportgraphics(fig, 'error_comparison.png', 'Resolution', 300);  % 300 DPI for high quality.
end