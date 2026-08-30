% =========================================================================
% ACOUSTIC LENS GENERATOR & EIKONAL SOLVER
% 3D Steered Cylindrical Wave Emission
%
% Simulates a cylindrical acoustic lens focusing a plane wave while 
% simultaneously applying elevation steering via phased delays.
% =========================================================================
clear; clc; close all;

%% 1. Setup 3D Domain and Lens Parameters
c1 = 3000.0; % Speed of sound in the lens (m/s)
c2 = 1500.0; % Speed of sound in the surrounding medium (m/s)
f  = 0.031;  % Virtual focal length (m)
d  = 0.04;   % Maximum thickness of the lens (m)
D  = 0.08;   % Aperture diameter of the lens (m)

% Global solver settings
msfm = true; 
elevation_steering_angle_deg = 15.0;
theta_target = deg2rad(elevation_steering_angle_deg);

% Isotropic 0.5mm grid starting from 0
dx = 0.5e-3; 
x_domain = 0.1;
y_domain = 0.12;
z_domain = 0.12;

Nx = round(x_domain / dx);
Ny = round(y_domain / dx);
Nz = round(z_domain / dx);

x_vec = (0:Nx-1)*dx; % Starts at 0 instead of -0.02
y_vec = (0:Ny-1)*dx - y_domain / 2;
z_vec = (0:Nz-1)*dx - z_domain / 2;

% ndgrid: X = Depth (propagation), Y = Lateral (lens curve), Z = Elevation (steering)
[X, Y, Z] = ndgrid(x_vec, y_vec, z_vec);

%% 2. Mathematical Surface Evaluation (Cylindrical Extrusion)
n_ratio = c2 / c1;            
L = d * (1 - n_ratio) + f;     
A = 1 - n_ratio^2;
B = 2 * (f - n_ratio * L);
C0 = f^2 - L^2;        

h_grid = zeros(Nx, Ny, Nz, 'single');
valid_Y = abs(Y) <= D / 2;

C_valid = Y(valid_Y).^2 + C0;
Delta = B^2 - 4 * A * C_valid;
h_grid(valid_Y) = (-B + sqrt(Delta)) / (2 * A);

% Place the lens near the beginning (e.g., starting at x = 2 mm)
lens_start_x = 0.002; 
lens_mask = valid_Y & (X >= lens_start_x) & (X <= (lens_start_x + h_grid));

%% 3. Discretize Map & Transducer Aperture
% The lens slowness is scaled relative to the background medium (c=1)
f_map = ones(Nx, Ny, Nz, 'single');
f_map(lens_mask) = c2 / c1;

source_x = 0.0;
source_idx_x = find(x_vec >= lens_start_x, 1, 'first');

valid_y_mask = abs(y_vec) <= D / 2;
valid_z_mask = abs(z_vec) <= D / 2;

y_in = y_vec(valid_y_mask);
z_in = z_vec(valid_z_mask);
valid_aperture = valid_y_mask(:) * valid_z_mask(:)'; % 2D logical mask [Ny x Nz]

%% 4. Apply Delays and Apodization to 3D Initial Conditions
% Create 2D apodization window (v_init) using the outer product of two Tukey windows
tukey_fn = @(N, r) (N == 1) * 1 + (N > 1) * (...
    (abs(linspace(-1, 1, N)') <= (1 - r)) .* 1 + ...
    ((abs(linspace(-1, 1, N)') > (1 - r)) & (abs(linspace(-1, 1, N)') <= 1)) .* ...
    (0.5 * (1 + cos(pi * (abs(linspace(-1, 1, N)') - (1 - r)) / r))) ...
); % Inline Tukey window matching scipy/signal toolbox

tw_y = tukey_fn(length(y_in), 0.15);
tw_z = tukey_fn(length(z_in), 0.15);
apod_2d = tw_y * tw_z'; % 2D weights for the transducer matrix

% Calculate target phase delays to steer the flat elevation axis (Z)
% Account for Snell's law refraction inside the lens
theta_lens = asin((c1 / c2) * sin(theta_target));

% Convert physical arrival times into grid-pixel units for the Eikonal solver
z_delays_phys = z_in * sin(theta_lens) / c1;
z_delays_pixels = z_delays_phys * (c2 / dx);
z_delays_pixels = z_delays_pixels - min(z_delays_pixels); % Shift so first element is 0
z_delays_2d = repmat(reshape(z_delays_pixels, 1, []), length(y_in), 1);

% Apply to initial conditions
u_init = inf(Nx, Ny, Nz, 'single');
v_init = zeros(Nx, Ny, Nz, 'single');

u_init(source_idx_x, valid_y_mask, valid_z_mask) = reshape(z_delays_2d, 1, sum(valid_y_mask), sum(valid_z_mask));
v_init(source_idx_x, valid_y_mask, valid_z_mask) = reshape(apod_2d, 1, sum(valid_y_mask), sum(valid_z_mask));
v_init(source_idx_x + 1, valid_y_mask, valid_z_mask) = reshape(apod_2d, 1, sum(valid_y_mask), sum(valid_z_mask));

% Baffle everything before the source
f_map(1:source_idx_x-1, :, :) = inf;

%% 5. 3D Steered Cylindrical Wave in Lens Medium
fprintf('Simulating 3D plane wave passing through cylindrical lens...\n');

[u_lensed, v_lensed] = eiko(u_init, f_map, 1, 'v_init', v_init, 'msfm', msfm);

% Animate using the default system animate_eikonal
animate_eikonal(u_lensed, v_lensed, ...
    'RenderMode', 'slice', ...
    'Outline', lens_mask, ...
    'Style', 'real', ...
    'Title', sprintf('Lensed Wave (%d° Steered)', elevation_steering_angle_deg));