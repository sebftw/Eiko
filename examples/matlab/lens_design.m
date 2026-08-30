% =========================================================================
% ACOUSTIC LENS GENERATOR & EIKONAL SOLVER
% Transforms a plane wave into a diverging spherical wave.
% =========================================================================
clear; clc; close all;

% =========================================================================
% CONFIGURATION
% =========================================================================
% Toggle apodization window along the lateral axis (true = Tukey, false = Rectangular / all ones)
apod_tukey_lateral = true;

% =========================================================================
% 1. Lens and Medium Parameters
% =========================================================================
c1 = 3000; % Speed of sound in the lens (m/s)
c2 = 1500; % Speed of sound in the surrounding medium (m/s)
f  = 0.031; % Virtual focal length (m)
d  = 0.04; % Maximum thickness of the lens (m)
D  = 0.08; % Aperture diameter of the lens (m)

% =========================================================================
% 2. Grid Parameters
% =========================================================================
dx = 0.25e-3; 
dy = 0.25e-3; 
x_domain = 0.1;
y_domain = 0.1+dy;

Nx = round(x_domain / dx);
Ny = round(y_domain / dy);

x_vec = (0:Nx-1)*dx - 0.02; 
y_vec = (0:Ny-1)*dy - y_domain/2; 
[X, Y] = ndgrid(x_vec, y_vec);

x_mm = x_vec * 1e3;
y_mm = y_vec * 1e3;

% =========================================================================
% 3. Mathematical Surface Evaluation
% =========================================================================
n = c2 / c1;           
L = d*(1 - n) + f;     
A = 1 - n^2;
B = 2 * (f - n * L);
C0 = f^2 - L^2;        

h_grid = zeros(Nx, Ny);
valid_y_idx = abs(y_vec) <= D/2;

C_valid = y_vec(valid_y_idx).^2 + C0;
Delta = B^2 - 4*A.*C_valid;
h_grid(:, valid_y_idx) = repmat((-B + sqrt(Delta)) / (2*A), Nx, 1);

% =========================================================================
% 4. Discretization 
% =========================================================================
lens_mask = (abs(Y) <= D/2) & (X >= 0) & (X <= h_grid);
sound_speed_map = c2 * ones(Nx, Ny, 'single');
sound_speed_map(lens_mask) = c1;

% =========================================================================
% 5. Compute Plane Wave (u)
% =========================================================================
source_x = 0.0; % mm
source_idx_x = find(x_vec <= source_x/1000, 1, 'last');
start_x_idx = find(x_vec >= 0.0, 1, 'first'); 

slowness = single(1 ./ sound_speed_map);
u_init = slowness * 0 + inf;
u_init(source_idx_x, valid_y_idx) = 0.0;

% Apply lateral baffles
slowness(1:source_idx_x-1, valid_y_idx) = inf;
slowness(1:start_x_idx-1, ~valid_y_idx) = inf;

y_in = y_vec(valid_y_idx);

if apod_tukey_lateral
    tw_lens = tukey_window(y_in / (D / 2.0), 0.15);
else
    tw_lens = ones(1, length(y_in));
end

h_in = (-B + sqrt(B^2 - 4*A*(y_in.^2 + C0))) / (2*A);
sin_theta_lens = y_in ./ sqrt((h_in + f).^2 + y_in.^2);

v_init = slowness * 0;
v_init(source_idx_x, valid_y_idx) = tw_lens;
v_init(source_idx_x + 1, valid_y_idx) = tw_lens;

fprintf('Computing plane wave passing through lens...\n');
[u, v] = eiko(u_init, slowness, dx, 'v_init', v_init);

aperture_mask = zeros(Nx, Ny);
aperture_mask(1:source_idx_x-1, valid_y_idx) = 1.0;
aperture_mask(1:start_x_idx-1, ~valid_y_idx) = 1.0;

% =========================================================================
% 6. Compute Virtual Focus (u_virt)
% =========================================================================
slowness_virt = (1/c2) * ones(Nx, Ny, 'single');
u_init_virt = slowness_virt * 0 + inf;

x_src = x_vec(source_idx_x);
dist_to_vf = sqrt((x_src - (-f)).^2 + y_vec(valid_y_idx).^2);

u_init_virt(source_idx_x, valid_y_idx) = dist_to_vf / c2;
slowness_virt(1:source_idx_x-1, valid_y_idx) = inf;
slowness_virt(1:start_x_idx-1, ~valid_y_idx) = inf;

sin_theta_virt = y_in ./ sqrt((x_src + f).^2 + y_in.^2);
tw_virt = interp1(sin_theta_lens, tw_lens, sin_theta_virt, 'linear', 0);

v_init_virt = slowness_virt * 0;
v_init_virt(source_idx_x, valid_y_idx) = tw_virt;
v_init_virt(source_idx_x + 1, valid_y_idx) = tw_virt;

fprintf('Computing virtual focus synthesis...\n');
[u_virt, v_virt] = eiko(u_init_virt, slowness_virt, dx, 'v_init', v_init_virt);

% =========================================================================
% 7. Synchronize the Wavefronts
% =========================================================================
target_depth = 0.05;
target_depth_idx = find(x_vec >= target_depth, 1, 'first');
center_y_idx = find(y_vec >= 0, 1, 'first');

t_target_lens = u(target_depth_idx, center_y_idx);
t_target_virt = u_virt(target_depth_idx, center_y_idx);

u_shifted = u;
u_shifted(isfinite(u_shifted)) = u_shifted(isfinite(u_shifted)) - t_target_lens;

u_virt_shifted = u_virt;
u_virt_shifted(isfinite(u_virt_shifted)) = u_virt_shifted(isfinite(u_virt_shifted)) - t_target_virt;

u_scaled = u_shifted * c2 / dx;
u_virt_scaled = u_virt_shifted * c2 / dx;

max_v1 = max(v(isfinite(v))); if isempty(max_v1) || max_v1 == 0, max_v1 = 1.0; end
max_v2 = max(v_virt(isfinite(v_virt))); if isempty(max_v2) || max_v2 == 0, max_v2 = 1.0; end

v_norm = v / max_v1; v_norm(isnan(v_norm)) = 0;
v_virt_norm = v_virt / max_v2; v_virt_norm(isnan(v_virt_norm)) = 0;

% =========================================================================
% 8. Render Static Ray Tracing Plot
% =========================================================================
fprintf('Rendering Ray Tracing Plot...\n');
fig1 = figure('Name', 'Acoustic Lens Ray Tracing', 'Color', 'w', 'Position', [100 100 900 500]);
hold on; grid on; axis equal;
set(gca, 'TickDir', 'out');

y_pts = linspace(-D/2, D/2, 500);
C_pts = y_pts.^2 + C0;
Delta_pts = B^2 - 4*A.*C_pts;
h_trace = (-B + sqrt(Delta_pts)) / (2*A); 

lens_x = [0, h_trace, 0] * 1e3;
lens_y = [y_pts(1), y_pts, y_pts(end)] * 1e3;
patch(lens_x, lens_y, [0.85 0.92 1.0], 'EdgeColor', 'b', 'LineWidth', 1.2, 'FaceAlpha', 0.6);

source_pos_mm = x_vec(source_idx_x) * 1e3;
plot([source_pos_mm, source_pos_mm], [-D/2 * 1e3, D/2 * 1e3], 'g-', 'LineWidth', 3.0);
plot(-f * 1e3, 0, 'p', 'MarkerSize', 11, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');

num_rays = 5;
y_rays = linspace(-D/2 * 0.92, D/2 * 0.92, num_rays);
for i = 1:num_rays
    yi = y_rays(i);
    hi = (-B + sqrt(B^2 - 4*A*(yi^2 + C0))) / (2*A);
    
    phi_n = atan2(-(-2*yi / (2*A*hi + B)), 1);
    phi_t = phi_n + asin((c2 / c1) * sin(-phi_n));
    
    x_end = hi + 0.05 * cos(phi_t);
    y_end = yi + 0.05 * sin(phi_t);

    plot([source_pos_mm, hi * 1e3, x_end * 1e3], [yi * 1e3, yi * 1e3, y_end * 1e3], 'b-', 'LineWidth', 1.3);
    
    if abs(yi) > 1e-4
        plot([-f * 1e3, hi * 1e3], [0, yi * 1e3], 'r:', 'LineWidth', 1.0);
    end
end
title('Acoustic Lens & Virtual Source Ray Geometry', 'FontSize', 13);
xlabel('Depth - Z [mm]');
ylabel('Lateral - X [mm]');
xlim([(-f - 0.015) * 1e3, (d + 0.055) * 1e3]);
ylim([(-D/2 - 0.015) * 1e3, (D/2 + 0.015) * 1e3]);
drawnow;

% =========================================================================
% 9. Render Stacked Synchronized Animation
% =========================================================================
fprintf('Animating wavefronts...\n');

pulse_width = 80.0;
speed = 0.5;
freq = 6 * pi;

valid_u1 = u_virt_scaled(isfinite(u_virt_scaled));
valid_u2 = u_scaled(isfinite(u_scaled));
min_t = min(min(valid_u1), min(valid_u2));
max_t = max(max(valid_u1), max(valid_u2));

u_top_safe = u_virt_scaled; u_top_safe(isinf(u_top_safe)) = max_t + pulse_width * 2;
u_bot_safe = u_scaled;      u_bot_safe(isinf(u_bot_safe)) = max_t + pulse_width * 2;

time_steps = (min_t - pulse_width/2) : speed : (max_t + pulse_width);

fig2 = figure('Name', 'Lens Design Animation', 'Color', 'w', 'Position', [150 150 700 900]);

red_overlay = cat(3, ones(Ny, Nx), zeros(Ny, Nx), zeros(Ny, Nx));
alpha_mask = 0.2 * aperture_mask'; 

% --- Top Panel: Virtual Focus Reference ---
ax1 = subplot(2, 1, 1);
im_virt = imagesc(x_mm, y_mm, zeros(Ny, Nx));
axis image; colormap(ax1, gray); caxis(ax1, [-1 1]);
% Set neutral mid-gray background to match the 0-level acoustic baseline
set(ax1, 'Color', [0.5 0.5 0.5], 'YDir', 'normal', 'TickDir', 'out');
title('Spherical Wave', 'FontSize', 12);
ylabel('Lateral - X [mm]');
set(ax1, 'XTickLabel', []);
hold on;
h_overlay_top = imagesc(x_mm, y_mm, red_overlay);
set(h_overlay_top, 'AlphaData', alpha_mask);
contour(x_mm, y_mm, aperture_mask', [0.5 0.5], 'r-', 'LineWidth', 1.5);
plot([source_pos_mm, source_pos_mm], [-D/2*1e3, D/2*1e3], 'g-', 'LineWidth', 3.5);
% text(x_mm(1)-5, 0, 'Virtual Focus', 'Rotation', 90, 'FontWeight', 'bold', 'FontSize', 13, 'HorizontalAlignment', 'center');

% --- Bottom Panel: Physical Lens ---
ax2 = subplot(2, 1, 2);
im_lens = imagesc(x_mm, y_mm, zeros(Ny, Nx));
axis image; colormap(ax2, gray); caxis(ax2, [-1 1]);
% Set neutral mid-gray background to match the 0-level acoustic baseline
set(ax2, 'Color', [0.5 0.5 0.5], 'YDir', 'normal', 'TickDir', 'out');
title('Plane Wave into Spherical', 'FontSize', 12);
xlabel('Depth - Z [mm]'); ylabel('Lateral - X [mm]');
hold on;
h_overlay_bot = imagesc(x_mm, y_mm, red_overlay);
set(h_overlay_bot, 'AlphaData', alpha_mask);
contour(x_mm, y_mm, lens_mask', [0.5 0.5], 'c-', 'LineWidth', 1.5);
contour(x_mm, y_mm, aperture_mask', [0.5 0.5], 'r-', 'LineWidth', 1.5);
plot([source_pos_mm, source_pos_mm], [-D/2*1e3, D/2*1e3], 'g-', 'LineWidth', 3.5);
% text(x_mm(1)-5, 0, 'Physical Lens', 'Rotation', 90, 'FontWeight', 'bold', 'FontSize', 13, 'HorizontalAlignment', 'center');

sgtitle('Lens Design', 'FontSize', 16, 'FontWeight', 'bold');

for t = time_steps
    % Virtual Focus Update
    diff_top = u_top_safe - t;
    valid_top = abs(diff_top) <= pulse_width/2;
    frame_top = zeros(Nx, Ny, 'single');
    env_top_full = zeros(Nx, Ny, 'single');
    if any(valid_top(:))
        d_top = diff_top(valid_top);
        env_top = 0.5 * (1.0 + cos(2 * pi * d_top / pulse_width));
        env_top_full(valid_top) = v_virt_norm(valid_top) .* env_top;
        frame_top(valid_top) = env_top_full(valid_top) .* cos(freq * d_top / pulse_width);
    end
    
    % Physical Lens Update
    diff_bot = u_bot_safe - t;
    valid_bot = abs(diff_bot) <= pulse_width/2;
    frame_bot = zeros(Nx, Ny, 'single');
    env_bot_full = zeros(Nx, Ny, 'single');
    if any(valid_bot(:))
        d_bot = diff_bot(valid_bot);
        env_bot = 0.5 * (1.0 + cos(2 * pi * d_bot / pulse_width));
        env_bot_full(valid_bot) = v_norm(valid_bot) .* env_bot;
        frame_bot(valid_bot) = env_bot_full(valid_bot) .* cos(freq * d_bot / pulse_width);
    end
    
    im_virt.CData = frame_top';
    im_lens.CData = frame_bot';
    
    % Smooth alpha envelope fading into the neutral gray background
    set(im_virt, 'AlphaData', double(env_top_full'));
    set(im_lens, 'AlphaData', double(env_bot_full'));
    
    drawnow;
end

%% Helper Functions
function w = tukey_window(x, alpha)
    % Generates a Tukey window matching the Python scipy implementation
    if nargin < 2
        alpha = 0.15;
    end
    r = abs(x);
    w = zeros(size(r));
    flat_idx = r <= (1.0 - alpha);
    taper_idx = (r > (1.0 - alpha)) & (r <= 1.0);
    w(flat_idx) = 1.0;
    w(taper_idx) = 0.5 * (1.0 + cos(pi * (r(taper_idx) - (1.0 - alpha)) / alpha));
end