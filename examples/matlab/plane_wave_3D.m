%% Example: 3D Volumetric Plane-Wave Steering and Aberration Correction
% Simulates a 2D matrix transducer array emitting a 3D steered plane wave.
% Demonstrates propagation in a homogeneous medium, an aberrated medium, 
% and corrects the 3D wavefront using time-reversal.

%% 1. Setup 3D Domain and Transducer Aperture
canvasHeight = 1080/4;
canvasWidth  = 1920/4;
canvasDepth  = canvasWidth; % Make the volume a square block in X-Z

% Global solver settings
msfm = false; % MSFM can be computationally heavy in 3D, disabled by default
steering_angle_deg = 20; 
theta = deg2rad(steering_angle_deg);

% Define the 2D aperture at the top of the canvas (y = 1)
aperture_y = 1;
aperture_x = floor(canvasWidth*0.2) : ceil(canvasWidth*0.8);
aperture_z = floor(canvasDepth*0.2) : ceil(canvasDepth*0.8);

% Create 2D apodization window (v_init) using the outer product of two Tukey windows
tw_x = tukeywin(numel(aperture_x));
tw_z = tukeywin(numel(aperture_z));
apod_2d = tw_z * tw_x'; % 2D weights for the matrix array

% Calculate 1D firing delays for steering in the X direction
delays_x = aperture_x * sin(theta);
delays_x = delays_x - min(delays_x); % Shift so first element fires at t=0

% Apply delays and apodization to the 3D Initial Condition Matrices
u_init = inf(canvasHeight, canvasWidth, canvasDepth, 'single');
v_init = zeros(canvasHeight, canvasWidth, canvasDepth, 'single');

for k = 1:numel(aperture_z)
    z_idx = aperture_z(k);
    u_init(aperture_y, aperture_x, z_idx) = delays_x;
    v_init(aperture_y, aperture_x, z_idx) = apod_2d(k, :);
end

%% 2. 3D Plane Wave in a Homogeneous Medium
fprintf('Simulating 3D plane wave in a homogeneous medium...\n');
f_homo = ones(canvasHeight, canvasWidth, canvasDepth, 'single'); % c=1

[u_homo, v_homo] = eiko(u_init, f_homo, 'v_init', v_init, 'msfm', msfm);

animate_eikonal(u_homo, v_homo, 'RenderMode', 'slice', 'Style', 'real', ...
    'Title', sprintf('Plane Wave (%d° Steered)', steering_angle_deg));

%% 3. 3D Plane Wave in an Inhomogeneous Medium (Aberrated)
fprintf('Simulating 3D plane wave in an aberrated medium...\n');

% Create a 3D spherical anomaly in the center of the volume
[X, Y, Z] = meshgrid(1:canvasWidth, 1:canvasHeight, 1:canvasDepth);
center_x = canvasWidth / 2;
center_y = canvasHeight * 0.35;
center_z = canvasDepth / 2;
radius = canvasWidth * 0.12;

sphere_mask = ((X - center_x).^2 + (Y - center_y).^2 + (Z - center_z).^2) <= radius^2;

% Modify the slowness map (e.g., 20% faster sound speed inside the sphere)
f_inhomo = ones(canvasHeight, canvasWidth, canvasDepth, 'single');
f_inhomo(sphere_mask) = 0.8; 

% Run forward solver with the exact same initial uncorrected delays
[u_uncorrected, v_uncorrected] = eiko(u_init, f_inhomo, 'v_init', v_init, 'msfm', msfm);

animate_eikonal(u_uncorrected, v_uncorrected, 'RenderMode', 'slice', 'Outline', sphere_mask, 'Style', 'real', ...
    'Title', sprintf('Aberrated Plane Wave (%d° Steered)', steering_angle_deg));

%% 4. 3D Aberration Correction via Time-Reversal
fprintf('Performing 3D aberration correction (time-reversal)...\n');

% 4a. Propagate a wave from the bottom face (target) back to the transducer
u_init_bw = inf(canvasHeight, canvasWidth, canvasDepth, 'single');
for z_idx = 1:canvasDepth
    u_init_bw(end, :, z_idx) = -(1:canvasWidth) * sin(theta);
end

u_bw = eiko(flipud(u_init_bw), flipud(f_inhomo), 'msfm', msfm);
u_bw = flipud(u_bw);

% 4b. Extract the highly complex 2D arrival times at the matrix array
arrival_times = u_bw(aperture_y, aperture_x, aperture_z);

% Use the maximum value to invert the wavefront shape
delays_corrected = max(arrival_times(:)) - arrival_times;

% 4c. Re-initialize the forward propagation with the 2D corrected delays
u_init_corrected = inf(canvasHeight, canvasWidth, canvasDepth, 'single');
for k = 1:numel(aperture_z)
    z_idx = aperture_z(k);
    u_init_corrected(aperture_y, aperture_x, z_idx) = delays_corrected(1, :, k);
end

% 4d. Run the corrected forward simulation
[u_corrected, v_corrected] = eiko(u_init_corrected, f_inhomo, 'v_init', v_init, 'msfm', msfm);

animate_eikonal(u_corrected, v_corrected, 'RenderMode', 'slice', 'Outline', sphere_mask, 'Style', 'real', ...
    'Title', sprintf('Aberration Corrected Plane Wave (%d° Steered)', steering_angle_deg));