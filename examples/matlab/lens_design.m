% =========================================================================
% ACOUSTIC LENS GENERATOR
% Transforms a plane wave into a diverging spherical wave.
% Configuration: Plano-convex ellipse (Lens is "faster" than the medium)
% =========================================================================
clear; clc; close all;

%% 1. Lens and Medium Parameters
c1 = 3000; % Speed of sound in the lens (m/s) (e.g., metal or hard plastic)
c2 = 1500; % Speed of sound in the surrounding medium (m/s) (e.g., water)
f  = 0.031; % Virtual focal length (m)
d  = 0.04; % Maximum thickness of the lens (m)
D  = 0.08; % Aperture diameter of the lens (m)

%% 2. k-Wave Grid Parameters
dx = 0.25e-3; % Grid resolution in x (250 microns)
dy = 0.25e-3; % Grid resolution in y (250 microns)

% Total size of the computational domain
x_domain = 0.1; % 15 cm long
y_domain = 0.12; % 12 cm wide

% Calculate number of grid points (Nx by Ny)
Nx = round(x_domain / dx);
Ny = round(y_domain / dy);

% Create k-Wave style coordinate matrices [Nx, Ny]
% We shift x so the lens flat face sits at x = 0.02 m inside the domain
x_vec = (0:Nx-1)*dx - 0.02; 
y_vec = (0:Ny-1)*dy - y_domain/2; 
[X, Y] = ndgrid(x_vec, y_vec);

%% 3. Mathematical Surface Evaluation
n = c2 / c1;           
L = d*(1 - n) + f;     
A = 1 - n^2;
B = 2 * (f - n * L);
C0 = f^2 - L^2;        

h_grid = zeros(Nx, Ny);
valid_y = abs(Y) <= D/2;

C_valid = Y(valid_y).^2 + C0;
Delta = B^2 - 4*A.*C_valid;

h_grid(valid_y) = (-B + sqrt(Delta)) / (2*A);

%% 4. Discretization (Creating the Mask)
lens_mask = (abs(Y) <= D/2) & (X >= 0) & (X <= h_grid);

sound_speed_map = c2 * ones(Nx, Ny);
sound_speed_map(lens_mask) = c1;

%% 5. Visualization & Plane Wave Animation
source_x = -10; % mm
if false
figure('Name', 'k-Wave Discretized Medium', 'Color', 'w', 'Position', [150 150 800 600]);
imagesc(x_vec*1000, y_vec*1000, sound_speed_map'); 
axis image tight;
set(gca, 'YDir', 'normal'); 
colormap(parula);
c = colorbar;
ylabel(c, 'Speed of Sound (m/s)', 'FontSize', 12, 'Rotation', 270, 'VerticalAlignment', 'bottom');
title(sprintf('Discretized Acoustic Lens\nResolution: %d \\mum, Grid: %d x %d', dx*1e6, Nx, Ny));
xlabel('Axial Position x (mm)');
ylabel('Transverse Position y (mm)');

hold on;
plot([source_x, source_x], [-D/2*1000, D/2*1000], 'r--', 'LineWidth', 2);
text(source_x - 4, 0, 'Source', 'FontSize', 16, 'Color', 'r', 'Rotation', 90, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% --- RENAMED VARIABLE HERE TO PREVENT OVERWRITING 'f' ---
valid_y_idx = abs(y_vec) <= D/2;
source_idx_x = find(x_vec <= source_x/1000, 1, 'last') - 1;

slowness = single(1./sound_speed_map); 
u_init = slowness * 0 + inf;
u_init(find(x_vec <= source_x/1000, 1, 'last')-1, valid_y_idx) = 0;

% Block reverse/wrap-around propagation
slowness(1:source_idx_x-1, valid_y_idx) = inf;
slowness(1:find(lens_mask(:, ceil(end/2)), 1, "first"), not(valid_y_idx)) = inf;

v_init = slowness * 0;
tw = tukeywin(sum(abs(y_vec) <= D/2)+2);
v_init(find(x_vec <= source_x/1000, 1, 'last')-1, abs(y_vec) <= D/2) = tw(2:end-1);
v_init(find(x_vec <= source_x/1000, 1, 'last'), abs(y_vec) <= D/2) = tw(2:end-1);

[u, v] = eiko(u_init, slowness, dx, 'v_init', v_init);



aperture_mask = zeros(size(slowness));
aperture_mask(1:source_idx_x-1, valid_y_idx) = 1;
figure(8);
animate_eikonal(u / dx * 1500, 'Overlay', aperture_mask, 'Outline', lens_mask, 'Style', 'real', 'Title', 'Lens Design - Plane Wave into Spherical');

%% 6. Virtual Focus Emission Synthesized by the Source Aperture (Phased Array)
% Here we use the physical array at x = source_x, but apply a time-delay 
% profile to synthesize a diverging wave originating from the virtual focus.

% We use a homogeneous medium (c2) to show the ideal synthesized wave 
% mimicking the lens output (acting as a phased array replacement).
% If you want to see what happens when the synthesized wave hits the physical
% lens, change this to: slowness_virt = single(1./sound_speed_map);
slowness_virt = (1/c2) * ones(Nx, Ny, 'single');

u_init_virt = slowness_virt * 0 + inf;

% Calculate distance from the virtual focus (-f, 0) to each point on the aperture
dist_to_vf = sqrt((source_x/1000 - (-f)).^2 + y_vec(valid_y_idx).^2);

% Convert distance to time delays (u = arrival time in Eikonal solver)
% We subtract the minimum distance so the center of the aperture fires at t = 0
time_delays = (dist_to_vf - min(dist_to_vf)) / c2;

% Apply the delay profile to the source aperture
u_init_virt(source_idx_x, valid_y_idx) = time_delays;

% Block backward propagation
slowness_virt(1:source_idx_x-1, valid_y_idx) = inf;

% Apply amplitude window (apodization)
v_init_virt = slowness_virt * 0;
v_init_virt(source_idx_x, valid_y_idx) = tw(2:end-1);

% Solve and animate
[u_virt, v_virt] = eiko(u_init_virt, slowness_virt, dx, 'v_init', v_init_virt);

aperture_mask = zeros(size(slowness_virt));
aperture_mask(1:source_idx_x-1, valid_y_idx) = 1;


figure(9);
animate_eikonal(u_virt / dx * 1500, 'Overlay', aperture_mask, 'Style', 'real', 'Title', 'Aperture Synthesizing a Virtual Focus');

%% 7. Plot the Lens Geometry & Ray Tracing 
% (Cleaned up duplications from your original script)
figure('Name', 'Acoustic Lens Ray Tracing', 'Color', 'w', 'Position', [100 100 900 500]);
hold on; grid on; axis equal;

y = linspace(-D/2, D/2, 500);
C = y.^2 + C0;
Delta = B^2 - 4*A.*C;
h = (-B + sqrt(Delta)) / (2*A); 

% Create the lens patch (flat face at x=0, curved face at x=h)
lens_x = [0, h, 0];
lens_y = [y(1), y, y(end)];
patch(lens_x, lens_y, [0.8 0.9 1.0], 'EdgeColor', 'b', 'LineWidth', 1.5, 'FaceAlpha', 0.5);

% Plot virtual source
plot(-f, 0, 'p', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
text(-f, -0.005, ' Virtual Source', 'Color', 'r', 'FontWeight', 'bold');

% Ray Tracing
num_rays = 11;
y_rays = linspace(-D/2 * 0.9, D/2 * 0.9, num_rays);
ray_length = 0.06;

for i = 1:num_rays
    yi = y_rays(i);
    Ci = yi^2 + C0;
    hi = (-B + sqrt(B^2 - 4*A*Ci)) / (2*A);
    
    % Incident ray
    plot([-f, hi], [yi, yi], 'b-', 'LineWidth', 1.5); 
    
    % Normal and Snell's Law
    dh_dy = -2*yi / (2*A*hi + B);
    phi_n = atan2(-dh_dy, 1);
    theta_i = 0 - phi_n;
    theta_t = asin((c2 / c1) * sin(theta_i));
    phi_t = phi_n + theta_t;
    
    % Transmitted ray
    x_end = hi + ray_length * cos(phi_t);
    y_end = yi + ray_length * sin(phi_t);
    plot([hi, x_end], [yi, y_end], 'r-', 'LineWidth', 1.5);
    
    % Back-traced dashed line
    plot([-f, hi], [0, yi], 'r--', 'LineWidth', 1);
end

title(sprintf('Acoustic Lens: Plano-Convex Ellipse\n(c_{lens} = %d m/s, c_{medium} = %d m/s)', c1, c2));
xlabel('Propagation Axis x (m)');
ylabel('Transverse Axis y (m)');
xlim([-f - 0.02, d + ray_length + 0.01]);
ylim([-D/2 - 0.02, D/2 + 0.02]);
legend('Acoustic Lens', 'Virtual Source', 'Incident/Internal Rays', ...
       'Transmitted Rays', 'Back-traced Rays', 'Location', 'best');