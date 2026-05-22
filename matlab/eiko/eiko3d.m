% EIKO Computes the shortest time-of-flight in an arbitrary 3D medium.
%
% Calculates the time-of-flight (u), given a slowness map (f = 1/c), and
% initial conditions (u_init, initialized as infinity at unknown points).
%
% EXAMPLE USAGES:
%   u = eiko3d(u_init, f);                               %  (Standard usage)
%   u = eiko3d(u_init, f, dx);                           % (Optional dx arg)
%   u = eiko3d(u_init, f, msfm=true);                    % (Named arguments)
%   u = eiko3d(u_init, f, dx, 'msfm', true);             % (Named arguments)
%   [u, v_out] = eiko3d(u_init, f, v_init=my_advection); %   (Advection arg)
%
% REQUIRED INPUTS:
%   u_init  - Initial conditions (known arrival times/delays).
%             Shape: (H, W, D) for a single image, or (H, W, B) for a batch.
%   f       - Slowness. Can be (H, W, D) [broadcast to batch] or (H, W, D, B).
%
% OPTIONAL INPUTS:
%   dx      - Input grid spacing. Default: 1.0.
%   v_init  - The initial advection field. Same size as u. Default: [] (not used).
%   msfm    - Whether to enable Multi-Stencil Fast Marching (MSFM).
%             Reduces bias along diagonal directions. Default: false.
%   gated   - Whether to enforce positive propagation along the first 
%             data dimension. Speeds up computations. It is valid when 
%             time only increases when moving axially. Default: false.
%
% OUTPUTS:
%   u   - Computed arrival time (time-of-flight) map. Shape matches u_init.
%   v   - Output advection vectors (returned only if v_init was supplied).
%
%  Click the links below to learn more about EIKO and the v (advection) argument.
%
%  See also: eiko_lib.doc.advection, eiko_lib.doc.theory, eiko
function [u_out, varargout] = eiko3d(u_init, f, dx, options)
    
    arguments
        u_init  % Untyped main arguments to preserve potential 'dlarray' objects.
        f       % Name-Value pairs defined using the 'options.' struct syntax:
        dx = 1.0
        options.v_init              = []
        options.msfm  (1,1) logical = false
        options.gated (1,1) logical = false
    end
    
    % VALIDATION: Ensure the input is strictly 2D or 2D-batched
    nd = ndims(u_init);
    if nd > 4
        error('Eiko:InvalidDimensions', ...
            'solve_eikonal expects a 3D grid (H, W, D) or batched 3D grid (H, W, D, B). Got %d dimensions.', nd);
    end
    
    % Route to the shared internal solver
    if nargout > 1
        [u_out, v_out] = eiko_lib.solve_eikonal_core(u_init, f, options.v_init, dx, options.msfm, options.gated, true);
        varargout{1} = v_out;
    else
        u_out = eiko_lib.solve_eikonal_core(u_init, f, options.v_init, dx, options.msfm, options.gated, true);
    end
end