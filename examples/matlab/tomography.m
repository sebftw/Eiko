% =========================================================================
% EIKONAL AUTOGRAD TEST: Traveltime Tomography Inversion
% =========================================================================
clear all; clc; close all;
parallel.gpu.enableCUDAForwardCompatibility(true);

%% 1. Setup Grid and True Slowness Model
N = 64; 
dx = single(1.0);
msfm = 0; % Use MSFM for sharper gradients

% Create true slowness (f) natively on the GPU
true_f = gpuArray(ones(N, N, 'single'));
true_f(25:40, 25:40) = 2.0;

%% 2. Setup Sources (Batched 2D)
% Size: [H, W, Batch]
u_init = gpuArray(inf(N, N, 4, 'single'));
src_coords = [
    5, 5;    % Top-Left
    5, 60;   % Top-Right
    60, 5;   % Bottom-Left
    60, 60   % Bottom-Right
];

for b = 1:4
    u_init(src_coords(b,1), src_coords(b,2), b) = 0;
end

%% 3. Generate "Measured" Data (Target Traveltimes)
disp('Generating target traveltime data...');
T_measured = eiko(u_init, true_f, dx, msfm=msfm);

%% 4. Optimization Setup
% Initial guess: A completely homogeneous field of 1.0
f_guess = gpuArray(ones(N, N, 'single'));

% Initialize the unconstrained latent variable (m = ln(f))
m_guess = log(f_guess); 
m_dl = dlarray(m_guess, 'SS'); 

learning_rate = 0.04; 
num_iters = 100;

% Preallocate history arrays on the GPU so we don't transfer data during the loop
loss_history = gpuArray(nan(num_iters, 1, 'single'));
f_history = gpuArray(zeros(N, N, num_iters, 'single')); 

% --- ADAM INITIALIZATION ---
averageGrad = [];
averageSqGrad = [];

%% 5. Pure Computation Loop (No Graphics, No D2H Transfers)
disp('Starting ADAM inversion (Pure GPU Computation)...');
opt_timer = tic; % START TIMER

for iter = 1:num_iters
    
    % Pass m_dl to the loss function, get grad w.r.t m_dl
    [loss, grad_m] = dlfeval(@eikonal_inversion_loss, m_dl, u_init, T_measured, dx, msfm);
    
    % Track loss (extractdata strips dlarray wrapper, but data stays on GPU)
    loss_history(iter) = extractdata(loss);
    
    % Update the latent variable m_dl
    [m_dl, averageGrad, averageSqGrad] = adamupdate(m_dl, grad_m, ...
        averageGrad, averageSqGrad, iter, learning_rate);
    
    % Extract and store the physical data matrix natively on the GPU
    f_history(:,:,iter) = extractdata(exp(m_dl)); 
    
end

% Force the CPU to wait for the GPU's asynchronous queue to clear before stopping the clock
wait(gpuDevice); 
elapsed_time = toc(opt_timer);

fprintf('Math finished in %.4f seconds (%.1f iterations/sec).\n', elapsed_time, num_iters/elapsed_time);

%% 6. Visualization Loop
disp('Rendering visualization...');

% Bring histories back to CPU RAM once, in a single block transfer, for plotting
loss_history_cpu = gather(loss_history);
f_history_cpu = gather(f_history);
true_f_cpu = gather(true_f);

% Setup Figure with clean white background and slightly wider aspect
fig = figure('Position', [100, 100, 1350, 1080/2], 'Color', 'w');

mseplot = false;
tl = tiledlayout(1, 2+mseplot, "TileSpacing","compact","Padding","loose");

% Initial Title
sgtitle(sprintf('Traveltime Tomography Inversion\nFinished in %.2f seconds (%d iterations).', elapsed_time, num_iters), 'FontSize', 16, 'FontWeight', 'bold');

for iter = 1:num_iters
    
    current_f = f_history_cpu(:,:,iter);
    
    if iter == 1
        % Axes 1: True Model
        ax1 = nexttile();
        plt1 = imagesc(1540./true_f_cpu, 1540./[2, 1]); 
        colormap(ax1, flipud(parula)); 
        % cb1 = colorbar;
        % cb1.Label.String = 'Sound Speed [m/s]';
        title('True Sound Speed', 'FontSize', 14); 
        axis image off; 
        
        % Axes 2: Recovered Model
        ax2 = nexttile();
        plt2 = imagesc(1540./current_f, 1540./[2, 1]); 
        colormap(ax2, flipud(parula)); 
        cb2 = colorbar;
        cb2.Label.String = 'Sound Speed [m/s]';
        cb2.Layout.Tile = 'East';
        ttl = title(sprintf('Recovered Sound Speed (Iter %d)', iter), 'FontSize', 14); 
        axis image off;

        linkaxes([ax1, ax2]);

        % set(gcf, 'Color', 'w');       % Changes the outer background to white
        % set(gca, 'Color', 'w');       % Changes the actual plot area background to white
        % sijkoet(gca, 'XColor', 'k', 'YColor', 'k'); % Changes the axes lines and text to black
        
        if mseplot
            % Axes 3: Loss Curve (Log Scale)
            ax3 = nexttile();
            plt3 = plot(1:num_iters, loss_history_cpu, 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410]);
            xlim([1, num_iters]);
            title('Optimization Loss', 'FontSize', 14); 
            xlabel('Iteration', 'FontSize', 12); 
            ylabel('MSE Loss (Log Scale)', 'FontSize', 12);
            grid on; box on;
            set(gca, 'FontSize', 11, 'TickDir', 'out');
        end
    else
        % Update plot data efficiently
        ttl.String = sprintf('Recovered Sound Speed (Iter %d)', iter);
        plt2.CData = 1540./current_f;
        if mseplot
            plt3.YData = loss_history_cpu;
        end
    end
    drawnow; % Animate as fast as MATLAB can render
    pause(0.05);
end

disp('Visualization complete!');

% =========================================================================
% LOSS FUNCTION FOR DLFEVAL
% =========================================================================
function [loss, grad_m] = eikonal_inversion_loss(m_dl, u_init, T_measured, dx, msfm)
    
    % Enforce positivity by mapping latent to physical
    f_dl = exp(m_dl); 
    
    % 1. Forward Pass
    T_pred = eiko(u_init, f_dl, dx, msfm=msfm);
    
    % 2. Compute Mean Squared Error Loss
    mse_diff = T_pred - T_measured;
    loss = 0.5 * sum(mse_diff.^2, 'all') / numel(T_measured);
    
    % 3. Trigger the Custom Backward Pass
    grad_m = dlgradient(loss, m_dl);
end