%% Example: Plane-Wave Steering and Aberration Correction
% This script demonstrates how to use the 'eiko' solver to simulate a 
% steered plane wave from a transducer aperture. It then showcases how to 
% perform aberration correction in an inhomogeneous medium using time-reversal.

%% 1. Setup Canvas and Transducer Aperture
canvasHeight = 1080/4;
canvasWidth  = 1920/4;

% Define the transducer aperture at the top of the canvas (y = 1)
aperture_y = 1;
% Create a 400-pixel wide transducer centered horizontally
aperture_x = floor(canvasWidth*0.2) : ceil(canvasWidth*0.8); 

% Calculate apodization window (v_init) for the source using a Tukey window
v_init = zeros(canvasHeight, canvasWidth, 'single');
tw = tukeywin(numel(aperture_x) + 2);
v_init(aperture_y, aperture_x) = tw(2:end-1);

% Plane-wave steering angle parameters
steering_angle_deg = 15; % Try changing this between -40 and 40!
theta = deg2rad(steering_angle_deg);

% Global solver settings
msfm = 1;

%% 2. Uncorrected Plane Wave in a Homogeneous Medium
fprintf('Simulating uncorrected plane wave in a homogeneous medium...\n');
f_homo = ones(canvasHeight, canvasWidth, 'single'); % Homogeneous medium (c=1)

% Calculate the ideal firing delays for the steered plane wave
delays_ideal = aperture_x * sin(theta);
delays_ideal = delays_ideal - min(delays_ideal); % Shift so first element fires at t=0

% Apply delays to the initial condition matrix
u_init = inf(canvasHeight, canvasWidth, 'single');
u_init(aperture_y, aperture_x) = delays_ideal;

% Run forward solver
[u_homo, v_homo] = eiko(u_init, f_homo, 'v_init', v_init, 'msfm', msfm);

animate_eikonal(u_homo, v_homo, 'Style', 'real', ...
    'Title', sprintf('Plane Wave (%d° Steered)', steering_angle_deg));

%% 3. Uncorrected Plane Wave in an Inhomogeneous Medium
fprintf('Simulating uncorrected plane wave in an inhomogeneous medium...\n');
% Simulate an inhomogeneous medium (e.g., tissue with fat/muscle layers)
islands = generate_island(canvasWidth, canvasHeight, 0.2);
outline = islands > 0;

% Modify the slowness map (faster sound speed inside the anomalies)
f_inhomo = ones(canvasHeight, canvasWidth, 'single') - (islands * 0.2);

% Run forward solver with the SAME ideal delays (no correction yet)
[u_uncorrected, v_uncorrected] = eiko(u_init, f_inhomo, 'v_init', v_init, 'msfm', msfm);

animate_eikonal(u_uncorrected, v_uncorrected, 'Outline', outline, 'Style', 'real', ...
    'Title', sprintf('Aberrated Plane Wave (%d° Steered)', steering_angle_deg));

%% 4. Aberration Correction via Time-Reversal
fprintf('Performing aberration correction (time-reversal)...\n');
% To correct for the distorted medium, we simulate a plane wave originating 
% from the bottom of the canvas (deep tissue) and traveling back to the transducer.

source_x = 1:canvasWidth;
u_init_bw = inf(canvasHeight, canvasWidth, 'single');
u_init_bw(end, source_x) = -source_x * sin(theta);

% We use flipud to propagate the wave from the bottom (end) back to the top (1)
u_bw = eiko(flipud(u_init_bw), flipud(f_inhomo), 'msfm', msfm, 'gated', gated_x);
u_bw = flipud(u_bw);

% Extract arrival times at the transducer
arrival_times = u_bw(aperture_y, aperture_x);

% Use the maximum value to invert the wavefront shape
delays_corrected = max(arrival_times(:)) - arrival_times;

% Re-initialize the forward propagation with the corrected delays
u_init_corrected = inf(canvasHeight, canvasWidth, 'single');
u_init_corrected(aperture_y, aperture_x) = delays_corrected;

% Run the corrected forward simulation
[u_corrected, v_corrected] = eiko(u_init_corrected, f_inhomo, 'v_init', v_init, 'msfm', msfm);

animate_eikonal(u_corrected, v_corrected, 'Outline', outline, 'Style', 'real', ...
    'Title', sprintf('Aberration Corrected Plane Wave (%d° Steered)', steering_angle_deg));