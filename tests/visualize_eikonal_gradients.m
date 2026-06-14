% =========================================================================
% visualize_eikonal_gradients.m
% Visualizes and compares AD (dlarray), Analytical, and Numerical gradients 
% for a 1D differentiable Eikonal solver.
% =========================================================================

clear; clc; close all;

%% 1. Domain and Parameter Setup
N = 20;                     % Number of grid points (increased for better visuals)
dx_val = single(0.1);       % Grid spacing
f_val = single(1.5);        % Constant slowness
epsilon = single(1e-3);     % Perturbation for numerical gradient

% Initialize standard numeric arrays [N, 1] layout
u_init_num = inf(N, 1, 'single');
u_init_num(1) = 0.0;        % Point source at the left boundary

f_num = repmat(f_val, N, 1);

%% 2. Automatic Differentiation (dlarray) Gradients
disp('Computing AD Gradients...');
u_init_dl = dlarray(u_init_num);
f_dl = dlarray(f_num);
dx_dl = dlarray(dx_val);

% Evaluate through the AD loss function
[~, grad_u_ad, grad_f_ad, grad_dx_ad] = dlfeval(@eikoLoss, u_init_dl, f_dl, dx_dl);

% Extract back to double for plotting
grad_u_ad = double(extractdata(grad_u_ad));
grad_f_ad = double(extractdata(grad_f_ad));
grad_dx_ad = double(extractdata(grad_dx_ad));

%% 3. Analytical (Closed-Form) Gradients
disp('Computing Analytical Gradients...');
% Analytical u_init: Only the source node shifts the final travel time directly.
grad_u_ana = zeros(N, 1);
grad_u_ana(1) = 1.0;

% Analytical f: Every node EXCEPT the source contributes dx to the final time.
grad_f_ana = repmat(double(dx_val), N, 1);
grad_f_ana(1) = 0.0;

% Analytical dx: Traverses (N-1) intervals, each adding f_val.
grad_dx_ana = double(f_val) * (N - 1);


%% 4. Numerical (Finite Difference) Gradients
disp('Computing Numerical Gradients...');
grad_u_num = zeros(N, 1);
for i = 1:N
    % Only perturb the finite source node. Inf nodes don't affect the gradient.
    if isinf(u_init_num(i))
        continue; 
    end
    u_plus = u_init_num; u_plus(i) = u_plus(i) + epsilon;
    u_minus = u_init_num; u_minus(i) = u_minus(i) - epsilon;
    
    loss_plus = computeNumLoss(u_plus, f_num, dx_val);
    loss_minus = computeNumLoss(u_minus, f_num, dx_val);
    grad_u_num(i) = double((loss_plus - loss_minus) / (2 * epsilon));
end

grad_f_num = zeros(N, 1);
for i = 1:N
    f_plus = f_num; f_plus(i) = f_plus(i) + epsilon;
    f_minus = f_num; f_minus(i) = f_minus(i) - epsilon;
    
    loss_plus = computeNumLoss(u_init_num, f_plus, dx_val);
    loss_minus = computeNumLoss(u_init_num, f_minus, dx_val);
    grad_f_num(i) = double((loss_plus - loss_minus) / (2 * epsilon));
end

dx_plus = dx_val + epsilon;
dx_minus = dx_val - epsilon;
loss_plus = computeNumLoss(u_init_num, f_num, dx_plus);
loss_minus = computeNumLoss(u_init_num, f_num, dx_minus);
grad_dx_num = double((loss_plus - loss_minus) / (2 * epsilon));


%% 5. Visualization
disp('Plotting Results...');
fig = figure('Name', 'Eikonal Gradient Validation', 'Position', [100, 100, 900, 800]);
t = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Validation of Eikonal Solver Gradients (\nabla L = \nabla u_N)', 'FontSize', 16, 'FontWeight', 'bold');

% --- Tile 1: Gradient w.r.t u_init ---
nexttile;
hold on; grid on;
% Using stem for u_init because it's a sparse impulse response
stem(1:N, grad_u_ana, 'o', 'MarkerSize', 10, 'LineWidth', 1.5, 'DisplayName', 'Analytical');
stem(1:N, grad_u_ad, 'x', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'AD (dlarray)');
stem(1:N, grad_u_num, '.', 'MarkerSize', 15, 'DisplayName', 'Numerical (FD)');
title('Gradient w.r.t Initial Conditions (u_{init})');
xlabel('Node Index'); ylabel('\partial L / \partial u_{init}');
legend('Location', 'northeast');
xlim([0, N+1]);

% --- Tile 2: Gradient w.r.t f ---
nexttile;
hold on; grid on;
plot(1:N, grad_f_ana, '-o', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Analytical');
plot(1:N, grad_f_ad, '--x', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'AD (dlarray)');
plot(1:N, grad_f_num, ':s', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Numerical (FD)');
title('Gradient w.r.t Slowness Field (f)');
xlabel('Node Index'); ylabel('\partial L / \partial f');
legend('Location', 'northeast');
xlim([0, N+1]); ylim([-0.05, double(dx_val) + 0.05]);

% --- Tile 3: Gradient w.r.t dx ---
nexttile;
% dx is a scalar, so we use a categorical bar chart
c = categorical({'Analytical', 'AD (dlarray)', 'Numerical (FD)'});
c = reordercats(c, {'Analytical', 'AD (dlarray)', 'Numerical (FD)'}); % Keep order
bar_data = [grad_dx_ana, grad_dx_ad, grad_dx_num];

b = bar(c, bar_data, 'FaceColor', 'flat');
b.CData(1,:) = [0 0.4470 0.7410];       % MATLAB default blue
b.CData(2,:) = [0.8500 0.3250 0.0980];  % MATLAB default orange
b.CData(3,:) = [0.9290 0.6940 0.1250];  % MATLAB default yellow
title('Gradient w.r.t Grid Spacing (dx)');
ylabel('\partial L / \partial dx');
grid on;

% Add text values on top of bars for exact comparison
for i = 1:numel(bar_data)
    text(i, bar_data(i), sprintf('%.4f', bar_data(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 12, 'FontWeight', 'bold');
end
ylim([0, max(bar_data)*1.2]); % Add headroom for text


% =========================================================================
% Local Helper Functions
% =========================================================================

function [loss, grad_u, grad_f, grad_dx] = eikoLoss(u_in, f_in, dx_in)
    % Forward pass tracked by AD
    msfm = false;
    u_out = eiko(u_in, f_in, dx_in, 'msfm', msfm);
    
    % The loss is the travel time at the furthest boundary
    loss = u_out(end, 1);
    
    % Compute exact gradients using automatic differentiation
    [grad_u, grad_f, grad_dx] = dlgradient(loss, u_in, f_in, dx_in);
end

function loss = computeNumLoss(u_in, f_in, dx_in)
    % Pure functional forward pass (No dlarray tracking)
    msfm = false;
    u_out = eiko(u_in, f_in, dx_in, 'msfm', msfm);
    loss = u_out(end, 1);
end