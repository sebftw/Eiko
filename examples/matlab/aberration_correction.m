%% 1. Setup Canvas and Uniform Medium
canvasHeight = ceil(1080/4);
canvasWidth  = ceil(1920/4) + 1;
f = ones(canvasHeight, canvasWidth, 'single'); 

% Simulate inhomogeneous medium (islands)
iheight = ceil(canvasHeight * 0.5);
islands = generate_island(canvasWidth, iheight, 0.25);
% Pad the islands to sit within the canvas
islands = [zeros(10, size(islands, 2)); islands; zeros(canvasHeight - iheight - 10, size(islands, 2))];
f = f + islands * 0.3;

%% 2. Transducer Aperture & Initial Setup
steering_angle_deg = 0; 
theta = deg2rad(steering_angle_deg);

aperture_y = 1;
aperture_x = floor(canvasWidth*0.2) : ceil(canvasWidth*0.8); 

% Calculate the Initial (Uncorrected) Firing Delays & Amplitudes
delays_init = aperture_x * sin(theta);
delays_init = delays_init - min(delays_init);

v_init = zeros(canvasHeight, canvasWidth, 'single');
tw = tukeywin(numel(aperture_x) + 2);  % Apodization
v_init(aperture_y, aperture_x) = tw(2:end-1);

%% 3. Dynamic Target Depth
% Find all rows that contain part of the island aberration
island_rows = find(any(islands > 0, 2));

if isempty(island_rows)
    % Fallback if no island was generated
    last_island_row = 0;
    target_y = canvasHeight - 20; 
    fprintf('No island detected. Evaluating wavefront at row %d.\n', target_y);
else
    last_island_row = max(island_rows);
    target_y = ceil(canvasHeight / 2);
    fprintf('Island ends at row %d. Evaluating wavefront at row %d.\n', last_island_row, target_y);
end

%% 4. Focused Aberration Correction (Time Reversal / Backward Pass)
focus_x = round(canvasWidth / 2);
focus_y = target_y; 
fprintf('Tracing virtual source from (x=%d, y=%d)...\n', focus_x, focus_y);

% F-number geometry for the Amplitude Template
aperture_width = numel(aperture_x);
focal_distance = focus_y - aperture_y;
dy = 21; 

% Initialize the Virtual Source Time-of-Flight (Analytical Sphere)
u_virtual = inf(canvasHeight, canvasWidth, 'single');
[X, Y] = meshgrid(1:canvasWidth, 1:canvasHeight);
dist_from_focus = sqrt((X - focus_x).^2 + (Y - focus_y).^2);

% Determine a radius that safely covers the template lines
template_width = max(1, round(aperture_width * (dy / focal_distance)));
R = ceil(sqrt(dy^2 + (template_width/2)^2)) + 2; 

% Assume the medium is locally homogeneous near the focus 
init_mask = dist_from_focus <= R;
local_slowness = f(focus_y, focus_x);
u_virtual(init_mask) = dist_from_focus(init_mask) * local_slowness;

% Initialize Amplitude (v_backwards) with F-number matching
v_backwards = zeros(canvasHeight, canvasWidth, 'single');
for current_dy = 1:dy
    % Project the width of the beam at this distance
    current_width = max(1, round(aperture_width * (current_dy / focal_distance)));
    virt_x = focus_x - floor(current_width/2) : focus_x + ceil(current_width/2) - 1;
    
    % Safety check to ensure we stay within bounds
    valid_idx = virt_x >= 1 & virt_x <= canvasWidth;
    virt_x = virt_x(valid_idx);
    
    % Create and apply apodization window
    tw_virt = tukeywin(current_width + 2);
    tw_virt = tw_virt(2:end-1);
    tw_virt = tw_virt(valid_idx)';
    
    v_backwards(focus_y + current_dy, virt_x) = tw_virt;
    v_backwards(focus_y - current_dy, virt_x) = tw_virt;
end

% Clear TOF outside the initialized analytical bounds
u_virtual(focus_y + dy + 1:end, :) = inf;
u_virtual(1:focus_y - dy - 1, :) = inf;

% Propagate Backwards (Upwards) to the Transducer
[u_backwards, v_backwards] = eiko(u_virtual, f, 'v_init', v_backwards, 'msfm', true); 

% Extract Arrival Times and compute Time-Reversed Firing Delays
arrival_times = u_backwards(aperture_y, aperture_x);
delays_corrected = max(arrival_times) - arrival_times;

%% 5. Forward Pass (Aberration Corrected)
% Inject the corrected delays into the forward model
u_init_forward = inf(canvasHeight, canvasWidth, 'single');
u_init_forward(aperture_y, aperture_x) = delays_corrected;

% Run the forward simulation through the aberrating medium
[u_focused, v_focused] = eiko(u_init_forward, f, 'v_init', v_init, 'msfm', true);

%% 6. Stitch Post-Focal Expansion
% The Eikonal solver computes first-arrival times. To visualize the expanding 
% wave after the focus, we stitch the flipped backward-pass onto the forward field.
t_focal = u_focused(focus_y, focus_x);

u_after = u_focused;
u_after(focus_y:end, :) = u_backwards(focus_y:end, :) + t_focal;

% Stitch and mirror the amplitudes around the focal plane
v_after = max(v_backwards, flipud_at_y(v_backwards, focus_y));

% Apply horizontal smoothing filter to the stitched amplitude
h_kernel = ones(1, 11);
v_after_smoothed = conv2(gather(v_after), h_kernel, "same") ./ conv2(gather(v_after*0+1), h_kernel, "same");

%% 7. Visualization
figure('Position', [100, 100, 1200, 400]);

subplot(1,3,1);
imagesc(u_backwards); colormap parula; hold on;
plot(focus_x, focus_y, 'r*', 'MarkerSize', 10);
title('Backward Pass (Virtual Source)'); axis image;

subplot(1,3,2);
imagesc(u_focused); colormap parula; hold on;
plot(focus_x, focus_y, 'r*', 'MarkerSize', 10);
contour(islands > 0, [0 1], 'Color', 'k', 'LineWidth', 2);
title('Forward Pass (First Arrival)'); axis image;

subplot(1,3,3);
imagesc(u_after); colormap parula; hold on;
plot(focus_x, focus_y, 'r*', 'MarkerSize', 10);
contour(islands > 0, [0 1], 'Color', 'k', 'LineWidth', 2);
title('Full Pass (Aberration Corrected)'); axis image;

% Animate using the smoothed amplitudes
animate_eikonal(u_after, v_after_smoothed, 'Outline', islands > 0, 'Style', 'real', 'Title', 'Aberration Correction');

%% --- Helper Functions ---

function flippedImg = flipud_at_y(img, y_center, fillValue)
    if nargin < 3
        fillValue = 0; 
    end
    [rows, cols, ~] = size(img);
    
    % Affine transformation to flip around a specific Y-coordinate
    T = [ 1,  0, 0; ... 
          0, -1, 0; ...  
          0,  2*y_center, 1]; 
      
    tform = affine2d(T);
    Rout = imref2d([rows, cols]);
    flippedImg = imwarp(img, tform, 'OutputView', Rout, 'FillValues', fillValue);
end