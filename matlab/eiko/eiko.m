% EIKO Computes the shortest time-of-flight in an arbitrary medium.
%
% Calculates the time-of-flight (u), given a slowness map (f = 1/c), and
% initial conditions (u_init, initialized as infinity at unknown points).
%
% EXAMPLE USAGES:
%   u = eiko(u_init, f);                               %  (Standard usage)
%   u = eiko(u_init, f, dx);                           % (Optional dx arg)
%   u = eiko(u_init, f, msfm=true);                    % (Named arguments)
%   u = eiko(u_init, f, dx, 'msfm', true);             % (Named arguments)
%   [u, v_out] = eiko(u_init, f, v_init=my_advection); %   (Advection arg)
%
% REQUIRED INPUTS:
%   u_init  - Initial conditions (known arrival times/delays). 
%             Shape: (H, W) for a single image, or (H, W, B) for a batch.
%   f       - Slowness map. Can be (H, W) [broadcast to batch] or (H, W, B).
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
%  See also: eiko_lib.doc.advection, eiko_lib.doc.theory, eiko3d
function [u_out, varargout] = eiko(u_init, f, dx, options)
    
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
    if nd > 3
        error('Eiko:InvalidDimensions', ...
            'solve_eikonal expects a 2D grid (H, W) or batched 2D grid (H, W, B). Got %d dimensions.', nd);
    end
    
    % Route to the shared internal solver
    if nargout > 1
        [u_out, v_out] = eiko_lib.solve_eikonal_core(u_init, f, options.v_init, dx, options.msfm, options.gated, false);
        varargout{1} = v_out;
    else
        u_out = eiko_lib.solve_eikonal_core(u_init, f, options.v_init, dx, options.msfm, options.gated, false);
    end
end