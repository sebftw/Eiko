function interactive_eiko()
    % INTERACTIVE_EIKO Runs an interactive Eikonal solver visualization.
    % Hover inside the plot for a point source, outside for a plane wave.
    
    %% 1. Setup Domain and Coordinates
    N = 201;            % Number of grid points (NxN grid)
    dx = 0.0005;         % Grid spacing in meters (1 mm)
    msfm = true;        % Multi-stencil fast marching
    c_bg = 1540.0;      % Background speed of sound
    
    % Create spatial coordinate grids centered at 0
    offset = floor(N / 2) * dx;
    x_coords = ((0:N-1) - floor(N/2)) * dx;
    y_coords = ((0:N-1) - floor(N/2)) * dx;
    
    % MATLAB meshgrid: X varies along columns, Y varies along rows
    [X, Y] = meshgrid(x_coords, y_coords);
    
    %% 2. Define Slowness Field (Customizable)
    c_field = c_bg * ones(N, N);
    
    % Example: High-speed circular anomaly (e.g., lens or bone at 3000 m/s)
    radius = 0.015;
    mask = (X - 0.0).^2 + (Y - 0.01).^2 <= radius^2;
    c_field(mask) = 3000.0;

    % Example U shaped impenetrable region.
    c_field(get_U_mask()) = 0.0;
    
    % Convert speed map to slowness map (1/c)
    f = 1.0 ./ c_field;
    
    % Calculate worst-case max time to fix the colorbar
    max_dist = sqrt(2) * (N * dx);
    min_speed = min(c_field(c_field > 0));
    global_max_time_ms = (max_dist / min_speed) * 1000;
    
    %% 3. Interactive Visualization Setup
    fprintf('Initializing Interactive EIKO solver...\n');
    
    fig = figure('Name', 'Interactive Eikonal Solver', ...
                 'Color', 'w', ...
                 'Position', [100, 100, 700, 600], ...
                 'WindowButtonMotionFcn', @on_mouse_move, ...
                 'KeyPressFcn', @on_key_press);
    
    ax = axes('Parent', fig);
    
    % Run an initial solve at the center
    u_init = inf(N, N);
    center_idx = ceil(N / 2);
    u_init(center_idx, center_idx) = 0.0;
    
    u_initial = eiko(u_init, f, dx, 'msfm', msfm);
    
    % Plot the Time-of-Flight map
    im = imagesc(x_coords*1000, y_coords*1000, u_initial * 1000, 'Parent', ax);
    set(ax, 'YDir', 'normal'); 
    colormap(ax, 'parula');
    
    % Apply the fixed color limits
    caxis(ax, [0, global_max_time_ms]); 
    
    cbar = colorbar(ax);
    ylabel(cbar, 'Time of Flight (ms)');
    xlabel(ax, 'x (mm)'); 
    ylabel(ax, 'y (mm)');
    title(ax, {'Interactive Eikonal Solver', '(Inside plot = Point Source | Outside plot = Plane Wave)'});
    
    hold(ax, 'on');
    
    % Overlay the speed map as faint contours
    contour(ax, X*1000, Y*1000, c_field, [1540 3000], ...
            'LineColor', 'w', 'LineStyle', '--', 'LineWidth', 1.5);

    % We specify 5 contour levels spread across the expected time range.
    contour_levels = linspace(0, global_max_time_ms, 7); 
    [~, hc] = contour(ax, x_coords*1000, y_coords*1000, u_initial * 1000, contour_levels, ...
                      'LineColor', 'k', 'LineStyle', '-', 'LineWidth', 0.5);
        
    % Initialize text box with two lines
    info_text = text(ax, 0.03, 0.96, sprintf('Source: Point at (0.0, 0.0) mm\nLocal Speed: %.0f m/s', c_bg), ...
                     'Units', 'normalized', ...
                     'Color', 'w', ...
                     'BackgroundColor', 'k', ...
                     'Margin', 4, ...
                     'VerticalAlignment', 'top');
    
    %% 4. Interactive Mouse Callback
    function on_mouse_move(~, ~)
        cp = get(ax, 'CurrentPoint');
        update_wave_source(cp(1,1), cp(1,2));
    end

    %% 5. Unified Update Function (Shared by Mouse & Recorder)
    function update_wave_source(x_m, y_m)
        x_min = x_coords(1)*1000; x_max = x_coords(end)*1000;
        y_min = y_coords(1)*1000; y_max = y_coords(end)*1000;
        is_inside = (x_m >= x_min && x_m <= x_max && y_m >= y_min && y_m <= y_max);
        
        u_init_new = inf(N, N);
        
        if is_inside
            idx_x = max(1, min(N, round((x_m / 1000.0 + offset) / dx) + 1));
            idx_y = max(1, min(N, round((y_m / 1000.0 + offset) / dx) + 1));
            
            c_local = c_field(idx_y, idx_x);
            u_init_new(idx_y, idx_x) = 0.0;
            set(info_text, 'String', sprintf('Source: Point at (%.1f, %.1f) mm\nLocal Speed: %.0f m/s', x_m, y_m, c_local));
        else
            theta = atan2(y_m, x_m);
            d_x = -cos(theta); d_y = -sin(theta);
            T = (X .* d_x + Y .* d_y) ./ c_bg;
            T = T - min(T(:));
            if d_x > 0; u_init_new(:, 1)   = T(:, 1);   end 
            if d_x < 0; u_init_new(:, end) = T(:, end); end 
            if d_y > 0; u_init_new(1, :)   = T(1, :);   end 
            if d_y < 0; u_init_new(end, :) = T(end, :); end 
            set(info_text, 'String', sprintf('Source: Plane Wave from %.1f°\nLocal Speed: N/A', rad2deg(theta)));
        end
        
        u_numerical = eiko(u_init_new, f, dx, 'msfm', msfm);
        u_ms = gather(u_numerical * 1000);
        
        set(im, 'CData', u_ms);
        set(hc, 'ZData', u_ms);
        drawnow limitrate;
    end

    %% 6. Key Press Callback for Automatic Recording
    function on_key_press(~, event)
        if strcmp(event.Key, 'r')
            % Unbind mouse motion temporarily so user hand movements don't interfere
            set(fig, 'WindowButtonMotionFcn', []);
            set(info_text, 'BackgroundColor', [0.5, 0, 0]); % Turn label dark red to indicate recording
            
            video_filename = 'eikonal_smooth_path';
            fprintf('Recording smooth trajectory to %s...\n', video_filename);
            
            % Setup VideoWriter (MPEG-4, 30 frames per second)
            if true
                v = VideoWriter(video_filename, 'Uncompressed AVI');
            else
                v = VideoWriter(video_filename, 'MPEG-4');
                v.Quality = 95;
            end
            v.FrameRate = 60;
            open(v);
            
            % --- Trajectory Configuration ---
            num_frames = 60*3;                % 60 frames gives a smooth 60 FPS result over 3 seconds
            radius_mm = 20.0;               % Radius of the circular trajectory (in mm)
            center_x = 0.0;                 % Center of rotation orbit
            center_y = -5.0;                
            
            % Generate theta values from 0 to 2*pi
            % Note: explicitly matching start and end positions makes the loop seamless
            angles = linspace(0, 2*pi, num_frames + 1); 
            angles(end) = []; % Drop last element to prevent double frames at the loop boundary
            
            x_circle = center_x + radius_mm * -cos(angles);
            y_circle = center_y + radius_mm * sin(angles);
            
            % Frame rate configurations
            delay_time = 1 / 60; % 60 FPS playback speed
            
            for k = 1:num_frames
                % 1. Force the wave source position to the next path step
                update_wave_source(x_circle(k), y_circle(k));
                
                % 2. Let the graphics refresh fully
                drawnow; 
                
                % 3. Capture the plot frame and write to disk
                frame = getframe(fig);
                writeVideo(v, frame);
            end
            
            close(v);
            fprintf('Recording Complete!\n');
            
            % Restore interactive mouse controls
            set(info_text, 'BackgroundColor', 'k');
            set(fig, 'WindowButtonMotionFcn', @on_mouse_move);
        end
    end

    function mask = get_U_mask()
        % Center of the U's curved bottom (matching your original center)
        xc = 0.033;
        yc = -0.025; % From (Y + 0.025) == 0
        
        % Dimensions for the U-shape
        r_outer = 0.006;                % Outer radius of the curve
        thickness = 0.002;              % How thick the wall is
        r_inner = r_outer - thickness;  % Inner radius of the curve
        arm_length = 0.015;             % How tall the vertical arms are
        
        % 1. Bottom Arc: A thick ring, cut in half
        % Note: If your Y-axis increases downwards (standard image/matrix coordinates), 
        % you may need to flip the `<` and `>` signs for the Y axis to keep it upright.
        arc_mask = ((X - xc).^2 + (Y - yc).^2 <= r_outer^2) & ... % Outer boundary
                   ((X - xc).^2 + (Y - yc).^2 >= r_inner^2) & ... % Inner boundary
                   (Y <= yc);                                     % Bottom half only
        
        % 2. Left Arm: Vertical rectangle
        left_arm_mask = (X >= xc - r_outer) & (X <= xc - r_inner) & ...
                        (Y > yc) & (Y <= yc + arm_length);
        
        % 3. Right Arm: Vertical rectangle
        right_arm_mask = (X >= xc + r_inner) & (X <= xc + r_outer) & ...
                         (Y > yc) & (Y <= yc + arm_length);
        
        % Combine all parts into a single mask using logical OR (|)
        mask = arc_mask | left_arm_mask | right_arm_mask;
    end
end