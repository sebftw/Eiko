%% 
% Validates the 'eiko' solver against a closed-form analytical solution
% for a point source in a constant speed-of-sound medium.

%% 1. Setup the Domain and Medium Properties
N = 101;            % Number of grid points (NxN grid)
dx = 0.001;         % Grid spacing in meters (e.g., 1 mm)
c = 1540;           % Speed of sound in m/s (uniform medium)
f = ones(N, N) / c; % Slowness map (1/c) everywhere
msfm = true;

% Create spatial coordinate vectors centered at 0
x_coords = ((1:N) - ceil(N/2)) * dx;
y_coords = ((1:N) - ceil(N/2)) * dx;
[X, Y] = meshgrid(x_coords, y_coords);

%% 2. Calculate the Analytical Solution
% The closed-form time-of-flight is distance / speed
R = sqrt(X.^2 + Y.^2);
u_analytical = R / c;

%% 3. Compute Numerical Solution using 'eiko'
% Initialize u_init with infinity at unknown points
u_init = inf(N, N);

% Set the point source at the center of the grid to time = 0
center_idx = ceil(N/2);
u_init(center_idx, center_idx) = 0;

% Call the eiko solver. We enable MSFM to reduce diagonal bias.
u_numerical = eiko(u_init, f, dx, 'msfm', msfm);

%% 4. Calculate Error and Check Tolerance
% Compare numerical and analytical fields
error_map = abs(u_numerical - u_analytical);
max_error = max(error_map(:));

% Define an acceptable tolerance. 
% Numerical errors in Eikonal solvers typically scale with dx. 
% We allow a maximum error roughly equivalent to traveling 1.5 grid cells.
tolerance = 1.5 * (dx / c); 

%% 5. Visualization
figure('Name', 'EIKO Solver Validation', 'Position', [100, 100, 1200, 400]);

% Plot Numerical
subplot(1, 3, 1);
imagesc(x_coords*1e3, y_coords*1e3, u_numerical);
axis image; colorbar;
title('Numerical (eiko)');
xlabel('x (mm)'); ylabel('y (mm)');

% Plot Analytical
subplot(1, 3, 2);
imagesc(x_coords*1e3, y_coords*1e3, u_analytical);
axis image; colorbar;
title('Analytical');
xlabel('x (mm)'); ylabel('y (mm)');

% Plot Error
subplot(1, 3, 3);
imagesc(x_coords*1e3, y_coords*1e3, error_map);
axis image; colorbar; colormap(gca, 'hot');
title(sprintf('Abs Error (Max: %1.2e s)', max_error));
xlabel('x (mm)'); ylabel('y (mm)');

drawnow; % Force the plot to render before potentially throwing an error

%% 6. Assertion
if max_error > tolerance
    error('EIKO:TestFailed', ...
        'Validation failed! Maximum error (%e s) exceeds the tolerance (%e s).', ...
        max_error, tolerance);
else
    fprintf('EIKO validation passed! Maximum error is %e s (Tolerance: %e s).\n', ...
        max_error, tolerance);
end