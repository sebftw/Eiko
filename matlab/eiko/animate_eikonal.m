function plt = animate_eikonal(u, varargin)
% ANIMATE_EIKONAL Visualizes Eikonal equation travel times as an animated wave.
%
% Usage:
%   animate_eikonal(u)
%   animate_eikonal(u, v, 'Style', 'db', 'VideoFilename', 'out.mp4')
%
% Required Inputs:
%   u - 2D or 3D matrix of travel times (can contain Inf for unreachable areas)
%
% Optional Inputs:
%   v - ND matrix representing amplitude/direction field v. Default is 1.
%
% Optional Name-Value Pairs:
%   'VelocityField' - 2D matrix (H x W) representing the direction field v. 
%                     If provided, a faint streamline/quiver overlay is added.
%   'VideoFilename' - String. If provided (e.g., 'wave.mp4'), exports the animation.
%   'Title'         - String to display above the animation.
%   'PulseWidth'    - Spatial width of the wave packet (default: 80).
%   'Speed'         - Time step per frame (default: 0.5). Lower is smoother.
%   'Outline'       - Outline to overlay on top of the 2D image.
%   'Overlay'       - Overlay to put on top of the 2D image.
%   'RenderMode'    - For 3D only: 'slice' or 'isosurface' (default: 'slice').

    % --- Input Parsing ---
    p = inputParser;
    addRequired(p, 'u', @isnumeric);
    addOptional(p, 'v', 1, @isnumeric);
    addParameter(p, 'ColorMap', gray(256), @isnumeric);
    addParameter(p, 'VideoFilename', '', @ischar);
    addParameter(p, 'Title', 'EIKONAL WAVEFRONT', @ischar);
    addParameter(p, 'PulseWidth', 80, @isnumeric);
    addParameter(p, 'Speed', 0.5, @isnumeric);
    addParameter(p, 'Overlay', [], @(x) isnumeric(x) || islogical(x));
    addParameter(p, 'Outline', [], @(x) isnumeric(x) || islogical(x));
    addParameter(p, 'Style', 'real', @ischar);  % MUSTBEMEMBER
    addParameter(p, 'RenderMode', 'slice', @ischar);

    parse(p, u, varargin{:});
    opts = p.Results;

    % Check dimensionality
    is3D = ndims(u) == 3;

    % --- Safe Data Initialization ---
    maxv = gather(max(u(u < inf))); 
    if isempty(maxv) || isinf(maxv)
        warning('Travel time field u is completely Inf. Nothing to animate.');
        return;
    end
    
    if isempty(opts.v)
        opts.v = 1;
    end

    u_safe = single(u); 
    u_safe(isinf(u_safe)) = maxv + opts.PulseWidth; % Push boundaries far away

    % Handle Permutation for 3D (to fix the XY plane transposition bug)
    if is3D
        % Permute from [Z, X, Y] -> [Y, X, Z] for MATLAB's meshgrid standard
        u_safe = permute(u_safe, [3, 2, 1]);
        if ~isscalar(opts.v)
            v_anim = permute(opts.v, [3, 2, 1]);
        else
            v_anim = opts.v;
        end
        Nx = size(u, 2);
        Ny = size(u, 3);
        Nz = size(u, 1);
    else
        v_anim = opts.v;
    end

    % --- Setup Gabor Wave Packet ---
    pulse_x = linspace(-opts.PulseWidth/2, opts.PulseWidth/2, 400);
    freq = 6 * pi;     
    pulse_y = exp(1i * freq * pulse_x / opts.PulseWidth);
    pulse_y = pulse_y(:) .* hanning(numel(pulse_y));
    pulse_y = cast(pulse_y, 'like', u_safe);

    % --- Figure & Axis Setup ---
    fig = figure('Color', [1 1 1], 'Position', [100, 100, size(u,2)*1.5*1.5, size(u,1)*1.5*1.5]); 
    set(fig, 'InvertHardcopy', 'off'); 
    
    ax = gca;

    if strcmpi(opts.Style, 'real')
        clim = [-1, 1];
    elseif strcmpi(opts.Style, 'abs')
        clim = [0, 1];
    elseif strcmpi(opts.Style, 'db')
        clim = [-30, 0];
        db_ref = 20 * log10(abs(1));
    end

    % Setup Render Objects
    if is3D
        % Use meshgrid directly since data is now [Y, X, Z]
        [X_grid, Y_grid, Z_grid] = meshgrid(1:Nx, 1:Ny, 1:Nz);
        
        % Calculate a threshold for the expanding bubble
        if strcmpi(opts.Style, 'db')
            iso_val = -6; % -6 dB envelope
        else
            iso_val = 0.5; % 50% amplitude
        end
        sx = round(Nx/2); % Mid-plane for X
        sy = round(Ny/2); % Mid-plane for Y
        sz = round(Nz/2); % Mid-plane for Z
        if strcmpi(opts.RenderMode, 'slice')
            hSlices = slice(ax, X_grid, Y_grid, Z_grid, zeros(size(u_safe)), sx, sy, sz, 'cubic');
            set(hSlices, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
            shading(ax, 'interp');
            % set(hSlices, 'EdgeColor', 'none', 'FaceAlpha', 'interp');
        elseif strcmpi(opts.RenderMode, 'isosurface')
            % Create an empty patch object
            hPatch = patch(ax, 'Faces', [], 'Vertices', [], ...
            'FaceColor', [0.2, 0.6, 1.0], 'EdgeColor', 'none', ...
            'FaceAlpha', 0.7, 'FaceLighting', 'gouraud', 'BackFaceLighting', 'reverselit');
            camlight('right'); lighting gouraud;
        elseif strcmpi(opts.RenderMode, 'contourslice')
            % Initialize empty handle array for contours
            hContours = [];
        elseif strcmpi(opts.RenderMode, 'cloud')
            cloud_thresh = iso_val * 0.4; % Lower threshold to grab more wave volume
            % Initialize empty scatter3 plot
            hCloud = scatter3(ax, NaN, NaN, NaN, 15, NaN, 'filled', ...
                'MarkerFaceAlpha', 0.15, 'MarkerEdgeColor', 'none');
        elseif strcmpi(opts.RenderMode, 'volshow')
            if exist('volshow', 'file') ~= 2
                error('The ''volshow'' RenderMode requires the Image Processing Toolbox.');
            end
            % Initialize volume viewer object natively inside our axes
            ax = [];
            viewer = viewer3d(fig, 'BackgroundColor', [1 1 1]);
            viewer.CameraUpVector = [0, 0, -1];
            hVol = volshow(zeros(size(u_safe), 'single'), 'Parent', viewer);
            hVol.Colormap = opts.ColorMap;
            % Cubic alphamap: Low values are completely invisible, high values are solid
            hVol.Alphamap = linspace(0, 1, 256).^3; 
        end

        if not(isempty(ax))
            % Labels to verify physical axes.
            xlabel(ax, 'X (Lateral)', 'Interpreter', 'latex', 'FontSize', 18);
            ylabel(ax, 'Y (Elevation)', 'Interpreter', 'latex', 'FontSize', 18);
            zlabel(ax, 'Z (Depth)', 'Interpreter', 'latex', 'FontSize', 18);
    
            % Ultrasound convention: Depth increases downwards
            % set(ax, 'ZDir', 'reverse');
            %set(ax, 'ZDir', 'reverse', 'TickLabelInterpreter', 'latex', 'FontSize', 14);
            set(ax, 'ZDir', 'reverse', 'TickLabelInterpreter', 'latex', 'FontSize', 14, 'FontName', 'Times New Roman');
    
            view(3);
            xlim(ax, [1, Nx]); ylim(ax, [1, Ny]); zlim(ax, [1, Nz]);
            daspect(ax, [1 1 1]); % Equivalent to 'axis equal' but strictly respects manual limits
            axis vis3d  % freeze aspect ratio during rotation
            grid on;
            % box on;
    
            ax.XAxis.Color = 'k';
            ax.YAxis.Color = 'k';
            ax.ZAxis.Color = 'k';
    
            if ~strcmpi(opts.RenderMode, 'volshow')
                colormap(ax, opts.ColorMap);
                caxis(ax, clim); % Use clim(ax, clim) if on MATLAB R2022a or newer.

                % --- 3D Outline Handling ---
                if ~isempty(opts.Outline)
                    % Ensure Outline is also permuted to match our [Y, X, Z] volume
                    outline_3d = permute(logical(opts.Outline), [3, 2, 1]);
                    
                    % Generate isosurface for the outline
                    fv = isosurface(X_grid, Y_grid, Z_grid, double(outline_3d), 0.5);
                    
                    % Add as a patch object
                    patch(ax, fv, 'FaceColor', [1.0 0.8 0.8], ...
                          'EdgeColor', 'none', ...
                          'FaceAlpha', 0.3, ...
                          'FaceLighting', 'gouraud', ...
                          'BackFaceLighting', 'reverselit');
                end
            end
        end
    else
        % 2D Rendering: imagesc naturally places Dim 1 on the Y-axis going down.
        plt = imagesc(zeros(size(u)), clim);
        colormap(ax, opts.ColorMap);
        axis image; axis off;
        ax.XAxis.Color = 'k';
        ax.YAxis.Color = 'k';
        xlabel('X (Lateral)', 'Interpreter', 'latex', 'FontSize', 18);
        ylabel('Z (Depth)', 'Interpreter', 'latex', 'FontSize', 18);
        if not(isempty(opts.Outline))
            hold(ax, 'on');
            [~, hContour] = contour(ax, opts.Outline, [0 1], 'Color', 'k', 'LineWidth', 2);
            hold(ax, 'off');
        end
        if not(isempty(opts.Overlay))
            %overlayColor = opts.Overlay;
            %hold(ax, 'on');
            %pltOverlay = imagesc(ax, overlayColor);
            %set(pltOverlay, 'AlphaData', opts.Overlay>0);
            %colormap(ax, [opts.ColorMap; [1, 0, 0]]);
            %hold(ax, 'off');
            boundaries = bwboundaries(opts.Overlay);
            for i = 1:numel(boundaries)
                b = boundaries{i};
                
                % Draw a solid polygon
                patch(ax, b(:,2), b(:,1), 'k', ...    % 'r' for red, or use [R G B]
                      'FaceAlpha', 1, ...             % 30% opaque fill
                      'EdgeColor', 'k', ...           % Red outline (or 'none')
                      'LineWidth', 2.5);
            end
        end
    end
    % title(opts.Title, 'Color', [0.2 0.2 0.2], 'Interpreter', 'latex', 'FontSize', 24);
    if not(isempty(ax))
        set(ax, 'FontName', 'Times New Roman', 'FontSize', 14, 'TickLabelInterpreter', 'latex');
        title(ax, opts.Title, 'Color', [0 0 0], 'Interpreter', 'latex', 'FontSize', 32);
        hold on;
    end

    % --- Video Setup ---
    export_video = ~isempty(opts.VideoFilename);
    if export_video
        if endsWith(opts.VideoFilename, '.avi')
            vidObj = VideoWriter(opts.VideoFilename, 'Uncompressed AVI');
        else
            vidObj = VideoWriter(opts.VideoFilename, 'MPEG-4');
            vidObj.Quality = 100;
        end
        vidObj.FrameRate = 60;
        
        open(vidObj);
    end

    % --- Animation Loop ---
    time_steps = 0 : opts.Speed : (maxv + opts.PulseWidth);
    
    for t = time_steps
        % Shift coordinate space and interpolate.
        if 0
            img = interp1(pulse_x, pulse_y, u_safe - t, 'cubic', 0);
        else
            % --- Fast Analytical Pulse Calculation ---
            diff_t = u_safe - t;
            
            % Create a mask of only the active wavefront to save computation
            valid_mask = abs(diff_t) <= (opts.PulseWidth / 2);
            
            % Initialize empty image.
            img = zeros(size(u_safe), 'like', u_safe);
            envelope_full = zeros(size(u_safe), 'like', u_safe);
            
            if any(valid_mask, 'all')
                d_valid = diff_t(valid_mask);
                
                % Analytical Hanning envelope: 0.5 * (1 + cos(2*pi * x / width))
                envelope = 0.5 * (1 + cos(2 * pi * d_valid / opts.PulseWidth));
                envelope_full(valid_mask) = envelope;
                
                % Apply envelope and carrier phase.
                img(valid_mask) = envelope .* exp(1i * freq * d_valid / opts.PulseWidth);
            end
        end
        img = img .* v_anim;

        % Apply Style.
        if strcmpi(opts.Style, 'real')
            final_img = real(img);
        elseif strcmpi(opts.Style, 'abs')
            final_img = abs(img);
        elseif strcmpi(opts.Style, 'db')
            final_img = 20 * log10(abs(img)) - db_ref;
        end
        
        % Render.
        if is3D
            % Permute the volume to match meshgrid orientation: [Y, X, Z] -> [Dim 3, Dim 2, Dim 1]
            % final_img = permute(final_img, [3, 2, 1]);
            % final_img = permute(gather(final_img), [3, 2, 1]);

            if strcmpi(opts.RenderMode, 'slice')
                % Update CData directly based on slice axes
                hSlices(1).CData = gather(squeeze(final_img(:, sx, :))); % X-slice
                hSlices(2).CData = gather(squeeze(final_img(sy, :, :))); % Y-slice
                if not(isempty(sz))
                    hSlices(3).CData = gather(squeeze(final_img(:, :, sz))); % Z-slice
                end

                %set(hSlices(1), 'AlphaData', squeeze(envelope_full(:, sx, :))); 
                %set(hSlices(2), 'AlphaData', squeeze(envelope_full(sy, :, :))); 
                %set(hSlices(3), 'AlphaData', squeeze(envelope_full(:, :, sz))); 
            elseif strcmpi(opts.RenderMode, 'isosurface')
                fv = isosurface(X_grid, Y_grid, Z_grid, final_img, iso_val);
                if not(isempty(fv.faces))
                    set(hPatch, 'Faces', fv.faces, 'Vertices', fv.vertices, 'FaceVertexCData', []);
                end
                % patch(ax, fv, 'FaceColor', [0.2 0.6 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
            elseif strcmpi(opts.RenderMode, 'contourslice')
                % Contours must be recreated, but doing so is visually beautiful
                if ~isempty(hContours), delete(hContours); end
                hContours = contourslice(ax, X_grid, Y_grid, Z_grid, gather(final_img), sx, sy, sz, 12);
                set(hContours, 'EdgeColor', 'flat', 'LineWidth', 1.5);
            elseif strcmpi(opts.RenderMode, 'cloud')
                f_img = gather(final_img);
                % Find indices where wave intensity is high enough
                idx = find(abs(f_img) > cloud_thresh);
                % Downsample points to maintain 30+ FPS (stride of 2 means 1/8th the points)
                idx = idx(1:2:end); 
                % Update scatter plot data
                set(hCloud, 'XData', X_grid(idx), 'YData', Y_grid(idx), ...
                            'ZData', Z_grid(idx), 'CData', f_img(idx));
            elseif strcmpi(opts.RenderMode, 'volshow')
                % Normalize data to [0, 1] based on clim_vals for smooth volshow rendering
                norm_img = (final_img - clim(1)) / (clim(2) - clim(1));
                hVol.Data = gather(max(0, min(1, norm_img))); % Clamp and update
            end
            drawnow limitrate;
        else
            plt.CData = gather(final_img);
            drawnow;
        end

        if export_video
            writeVideo(vidObj, getframe(fig));
        end
    end

    if export_video
        close(vidObj);
        disp(['Video saved to ' opts.VideoFilename]);
    end
end