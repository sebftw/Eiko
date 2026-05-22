%% Generates eiko logo.
parallel.gpu.enableCUDAForwardCompatibility(true);

% --- Toggle Features Here ---
saveStaticLogo = true;  % Set to true to save 'eiko_logo.png'
saveVideo = false;       % Set to true to save 'eiko_animation.mp4'
% ----------------------------

% --- 1. Setup Canvas and Solver ---
canvasHeight = 200;
canvasWidth = 600;
supersample = 0.5;
msfm = true;
gating = true;

canvas = zeros([canvasHeight, canvasWidth]*max(supersample, 1), 'uint8');
canvas = insertText(canvas, [canvasWidth, canvasHeight]/2, 'Eiko', ...
                     'Font', 'Cascadia Code', ...
                     'FontSize', 200, ...
                     'TextColor', 'white', ...
                     'BoxOpacity', 0, "AnchorPoint", "Center");
canvas = imresize(single(rgb2gray(canvas)), 1/supersample, 'nearest');

% Slowness map: 1 in background, 3 inside text
f = (single(canvas)/255)*2 + 1;

if saveStaticLogo
    padHeight = 90;
    static_mask = [zeros(padHeight, size(canvas, 2), 'single'); canvas];
    
    % Insert the title text into the padded mask
    static_mask = insertText(uint8(static_mask), [size(static_mask,2)/2, padHeight/2+5], ...
                             'TIME OF FLIGHT CALCULATOR', ...
                             'Font', 'Cascadia Code', ...
                             'FontSize', 36, ...
                             'TextColor', 'white', ...
                             'BoxOpacity', 0, "AnchorPoint", "Center");
                         
    % Convert back to 2D grayscale array to use as an alpha map
    static_alpha = rgb2gray(static_mask);
    
    % Create a flat gray RGB image (128, 128, 128) matching the new size
    gray_img = repmat(uint8(128 * ones(size(static_alpha))), [1, 1, 3]);
    
    % Use the static_alpha (0 to 255) as the alpha channel for transparency
    imwrite(gray_img, 'eiko_logo.png', 'Alpha', static_alpha);
    disp('Static logo saved as eiko_logo.png');
end

u_init = inf(size(canvas), 'single');
u_init(ceil(end/2), 1) = 0; % Start from middle-left edge

u = eiko(u_init.', f.', 'msfm', msfm, 'gated', gating).';

% --- 2. Visualization Setup ---
pulse_width = 80;  
freq = 3 * pi;     

maxv = max(u(u < inf)); 
u_safe = single(u); 
u_safe(isinf(u_safe)) = maxv + pulse_width; 

% Push arrays to GPU *before* the loop to prevent PCIe bottlenecking
u_safe_gpu = gpuArray(u_safe);
text_base = gpuArray(imgaussfilt(single(f > 1), 0.5) * 0.95);
current_wave = zeros(size(u), 'single', 'gpuArray');
wave_history = zeros(size(u), 'single', 'gpuArray');

figure(2); clf;
set(gcf, 'Color', [1 1 1], 'Position', [100, 100, canvasWidth*1.5, canvasHeight*1.5], 'InvertHardcopy', 'off'); 

ax = gca;
plt = imagesc(zeros(size(u)), [0, 1.0]); 
colormap(ax, 1-gray(256));
axis image off;
title('TIME OF FLIGHT CALCULATOR', 'Color', [0.2 0.2 0.2], 'FontName', 'Cascadia Code', 'FontSize', 16, 'FontWeight', 'bold');

% --- 3. Optimized Animation Loop ---
trail_decay = 0.0;
time_steps = 0:0.5:(maxv + pulse_width);

% Precompute the exact normalization constants your original code used
temp_x = linspace(-pulse_width/2, pulse_width/2, 400);
temp_y = cos(freq * temp_x / pulse_width) .* hanning(400).';
global_my = min(temp_y);

if saveVideo
    frame_skip = 8;
    vidObj = VideoWriter('eiko_animation.avi', 'Uncompressed AVI');
    % vidObj = VideoWriter('eiko_animation.mp4', 'MPEG-4');
    
    vidObj.FrameRate = 60; % Adjust frame rate if the animation feels too fast/slow
    if isfield('Quality', vidObj)
        vidObj.Quality = 100;
    end
    open(vidObj);
end

for i = 1:numel(time_steps)
    t = time_steps(i);
    diff_t = u_safe_gpu - t;
    active_mask = abs(diff_t) <= (pulse_width / 2);
    
    raw_wave = zeros(size(u), 'single', 'gpuArray');
    
    if any(active_mask, 'all')
        d_val = diff_t(active_mask);
        
        % Analytical Gabor patch
        window = 0.5 * (1 + cos(2 * pi * d_val / pulse_width));
        carrier = cos(freq * d_val / pulse_width);
        packet = window .* carrier;
        
        % Apply your exact original global normalization
        packet = (packet + global_my) / (1 + global_my);
        
        raw_wave(active_mask) = packet;
    end
    
    wave_history = max(wave_history, raw_wave);
    current_wave = max(raw_wave, current_wave * trail_decay);
    visible_text = text_base .* wave_history;
    
    final_img = max(current_wave, visible_text);
    
    plt.CData = gather(final_img);
    if saveVideo && mod(i-1, frame_skip) == 0
        drawnow; % Standard drawnow ensures the figure updates fully before capture
        frame = getframe(gcf);
        writeVideo(vidObj, frame);
    else
        drawnow limitrate;
    end
end

if saveVideo
    close(vidObj);
    disp('Animation saved as eiko_animation.mp4');

    % The video can then be converted to webm using
    % !ffmpeg -i eiko_animation.avi -c:v libvpx-vp9 -crf 20 -b:v 0 -an eiko_animation.webm
    % or gif using
    % !ffmpeg -i eiko_animation.avi -filter_complex "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse" eiko_animation.gif
end

