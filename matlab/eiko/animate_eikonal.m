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
%   'Extent'        - [xmin, xmax, zmin, zmax] (2D) or [xmin, xmax, ymin, ymax, zmin, zmax] (3D). Maps indices to physical units.
%   'VelocityField' - 2D matrix (H x W) representing the direction field v. 
%                     If provided, a faint streamline/quiver overlay is added.
%   'VideoFilename' - String. If provided (e.g., 'wave.mp4'), exports the animation.
%   'Title'         - String to display above the animation.
%   'PulseWidth'    - Spatial width of the wave packet (default: 80).
%   'Speed'         - Time step per frame (default: 0.5). Lower is smoother.
%   'Outline'       - Outline to overlay on top of the 2D/3D image.
%   'Overlay'       - Overlay to put on top of the 2D image.
%   'Style'         - 'real', 'abs', or 'db' (default: 'real').
%   'RenderMode'    - For 3D only: 'slice', 'isosurface', 'contourslice', 'cloud', or 'volshow' (default: 'slice').

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
    addParameter(p, 'Extent', [], @isnumeric);

    parse(p, u, varargin{:});
    opts = p.Results;

    % Check dimensionality
    is3D = ndims(u) == 3;

    % Format units if an Extent was passed
    has_extent = ~isempty(opts.Extent);
    if has_extent
        unit_suffix = ' [mm]';
    else
        unit_suffix = '';
    end

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
        Nx = size(u, 2); % Lateral dimension
        Ny = size(u, 3); % Elevation dimension
        Nz = size(u, 1); % Depth dimension

        if has_extent && numel(opts.Extent) == 6
            % Map Extent based on original array: [Depth, Lateral, Elevation]
            x_vals = linspace(opts.Extent(3), opts.Extent(4), Nx); % Lateral
            y_vals = linspace(opts.Extent(5), opts.Extent(6), Ny); % Elevation
            z_vals = linspace(opts.Extent(1), opts.Extent(2), Nz); % Depth
        else
            x_vals = 1:Nx;
            y_vals = 1:Ny;
            z_vals = 1:Nz;
        end

        [X_grid, Y_grid, Z_grid] = meshgrid(x_vals, y_vals, z_vals);
    else
        v_anim = opts.v;
    end

    % --- Setup Gabor Wave Packet Frequency ---
    freq = 6 * pi;     

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
        % Calculate mid-plane slices
        sx_idx = round(Nx/2);
        sy_idx = round(Ny/2);
        sz_idx = round(Nz/2);
        
        sx = x_vals(sx_idx);
        sy = y_vals(sy_idx);
        sz = z_vals(sz_idx);

        % Calculate a threshold for the expanding bubble
        if strcmpi(opts.Style, 'db')
            iso_val = -6; % -6 dB envelope
        else
            iso_val = 0.5; % 50% amplitude
        end

        if strcmpi(opts.RenderMode, 'slice')
            hSlices = slice(ax, X_grid, Y_grid, Z_grid, zeros(size(u_safe)), sx, sy, sz, 'cubic');
            % FaceAlpha set to 'interp' allowing dynamic AlphaData transparency later
            set(hSlices, 'EdgeColor', 'none', 'FaceAlpha', 'interp');
            shading(ax, 'interp');
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
            % Labels to verify physical axes
            xlabel(ax, ['X (Lateral)' unit_suffix], 'Interpreter', 'latex', 'FontSize', 18);
            ylabel(ax, ['Y (Elevation)' unit_suffix], 'Interpreter', 'latex', 'FontSize', 18);
            zlabel(ax, ['Z (Depth)' unit_suffix], 'Interpreter', 'latex', 'FontSize', 18);
    
            % Ultrasound convention: Depth increases downwards
            set(ax, 'ZDir', 'reverse', 'TickLabelInterpreter', 'latex', 'FontSize', 14, ...
                'FontName', 'Times New Roman', 'TickDir', 'out', 'Color', [0.5 0.5 0.5]);
    
            view(3);
            if has_extent
                xlim(ax, [min(x_vals), max(x_vals)]); 
                ylim(ax, [min(y_vals), max(y_vals)]); 
                zlim(ax, [min(z_vals), max(z_vals)]);
            else
                xlim(ax, [1, Nx]); ylim(ax, [1, Ny]); zlim(ax, [1, Nz]);
            end
            
            daspect(ax, [1 1 1]); % Strictly respects manual physical limits for true 1:1:1 aspect
            axis vis3d  % Freeze aspect ratio during rotation
            grid on;
    
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
                    
                    % Add as a patch object, restoring the clean red/pink original aesthetic
                    if ~isempty(fv.faces)
                        patch(ax, 'Faces', fv.faces, 'Vertices', fv.vertices, ...
                              'FaceColor', [1.0 0.8 0.8], ...
                              'EdgeColor', 'none', ...
                              'FaceAlpha', 0.3, ...
                              'FaceLighting', 'gouraud', ...
                              'BackFaceLighting', 'reverselit');
                    end
                end
            end
        end
    else
        % 2D Rendering: imagesc naturally places Dim 1 on the Y-axis going down.
        if has_extent && numel(opts.Extent) >= 4
            x_lims = [opts.Extent(3), opts.Extent(4)];
            z_lims = [opts.Extent(1), opts.Extent(2)];
            plt = imagesc(ax, x_lims, z_lims, zeros(size(u)));
        else
            plt = imagesc(ax, zeros(size(u)));
        end
        colormap(ax, opts.ColorMap);
        caxis(ax, clim);
        axis image;
        
        set(ax, 'TickDir', 'out', 'Color', [0.5 0.5 0.5]);
        ax.XAxis.Color = 'k';
        ax.YAxis.Color = 'k';
        
        xlabel(['X (Lateral)' unit_suffix], 'Interpreter', 'latex', 'FontSize', 18);
        ylabel(['Z (Depth)' unit_suffix], 'Interpreter', 'latex', 'FontSize', 18);
        
        if not(isempty(opts.Outline))
            hold(ax, 'on');
            if has_extent && numel(opts.Extent) >= 4
                x_vals_2d = linspace(x_lims(1), x_lims(2), size(u,2));
                z_vals_2d = linspace(z_lims(1), z_lims(2), size(u,1));
                [~, ~] = contour(ax, x_vals_2d, z_vals_2d, opts.Outline, [0.5 0.5], 'Color', 'cyan', 'LineWidth', 2);
            else
                [~, ~] = contour(ax, opts.Outline, [0.5 0.5], 'Color', 'cyan', 'LineWidth', 2);
            end
            hold(ax, 'off');
        end
        
        if not(isempty(opts.Overlay))
            boundaries = bwboundaries(opts.Overlay);
            for i = 1:numel(boundaries)
                b = boundaries{i};
                
                if has_extent && numel(opts.Extent) >= 4
                    x_coords = x_lims(1) + (b(:,2) - 1) * (x_lims(2) - x_lims(1)) / (size(u,2) - 1);
                    z_coords = z_lims(1) + (b(:,1) - 1) * (z_lims(2) - z_lims(1)) / (size(u,1) - 1);
                    patch(ax, x_coords, z_coords, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'r', 'LineWidth', 2.5);
                else
                    patch(ax, b(:,2), b(:,1), 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'r', 'LineWidth', 2.5);
                end
            end
        end
    end
    
    if not(isempty(ax))
        set(ax, 'FontName', 'Times New Roman', 'FontSize', 14, 'TickLabelInterpreter', 'latex');
        % Sanitize title string to prevent LaTeX crashing on special chars (e.g. °)
        latex_title = strrep(opts.Title, '°', '$^\circ$');
        title(ax, latex_title, 'Color', [0 0 0], 'Interpreter', 'latex', 'FontSize', 24);
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
        % --- Fast Analytical Pulse Calculation ---
        diff_t = u_safe - t;
        
        % Create a mask of only the active wavefront to save computation
        valid_mask = abs(diff_t) <= (opts.PulseWidth / 2);
        
        % Initialize empty image.
        img = complex(zeros(size(u_safe), 'like', u_safe));
        envelope_full = zeros(size(u_safe), 'like', u_safe);
        
        if any(valid_mask, 'all')
            d_valid = diff_t(valid_mask);
            
            % Analytical Hanning envelope: 0.5 * (1 + cos(2*pi * x / width))
            envelope = 0.5 * (1 + cos(2 * pi * d_valid / opts.PulseWidth));
            envelope_full(valid_mask) = envelope;
            
            % Apply envelope and carrier phase.
            img(valid_mask) = envelope .* exp(1i * freq * d_valid / opts.PulseWidth);
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
            if strcmpi(opts.RenderMode, 'slice')
                % Dynamically extract slices and cast envelope masks to double
                slice_x = gather(squeeze(final_img(:, sx_idx, :)));
                alpha_x = double(gather(squeeze(envelope_full(:, sx_idx, :))));
                
                % slice() internally transposes planes for certain axes. 
                % We explicitly map our CData/AlphaData shape to match the slice shape.
                if ~isequal(size(slice_x), size(hSlices(1).CData))
                    slice_x = slice_x.';
                    alpha_x = alpha_x.';
                end
                hSlices(1).CData = slice_x;
                set(hSlices(1), 'AlphaData', alpha_x); 
                
                slice_y = gather(squeeze(final_img(sy_idx, :, :)));
                alpha_y = double(gather(squeeze(envelope_full(sy_idx, :, :))));
                
                if ~isequal(size(slice_y), size(hSlices(2).CData))
                    slice_y = slice_y.';
                    alpha_y = alpha_y.';
                end
                hSlices(2).CData = slice_y;
                set(hSlices(2), 'AlphaData', alpha_y); 
                
                if not(isempty(sz_idx))
                    slice_z = gather(squeeze(final_img(:, :, sz_idx)));
                    alpha_z = double(gather(squeeze(envelope_full(:, :, sz_idx))));
                    
                    if ~isequal(size(slice_z), size(hSlices(3).CData))
                        slice_z = slice_z.';
                        alpha_z = alpha_z.';
                    end
                    hSlices(3).CData = slice_z;
                    set(hSlices(3), 'AlphaData', alpha_z); 
                end

            elseif strcmpi(opts.RenderMode, 'isosurface')
                fv = isosurface(X_grid, Y_grid, Z_grid, gather(final_img), iso_val);
                if not(isempty(fv.faces))
                    set(hPatch, 'Faces', fv.faces, 'Vertices', fv.vertices, 'FaceVertexCData', []);
                end
            elseif strcmpi(opts.RenderMode, 'contourslice')
                % Contours must be recreated
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
            set(plt, 'AlphaData', gather(envelope_full));
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