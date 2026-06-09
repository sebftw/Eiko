% =========================================================================
% INTERNAL FUNCTION: solve_eikonal_core
% Solves the eikonal equation using the batched GPU Fast Iterative Method.
%
% Handles type casting, autograd routing, and broadcasting logic for both
% the 2D and 3D Eikonal interfaces.
%
% You're not not supposed to use this class. Just call solve_eikonal,
% which will correctly use this function.
% =========================================================================
function [u_out, varargout] = solve_eikonal_core(u_init, f, v, dx, msfm, gated_x, is_3d)
    
    eiko_lib.setup();
    
    % Whether v is supplied.
    has_v = ~isempty(v);

    % Broadcast Detection Logic
    % Detect if `f` is a single instance being applied across a batched `u_init`.
    broadcast_f = false;
    if ndims(f) == ndims(u_init) - 1
        broadcast_f = true; 
    elseif ndims(f) == ndims(u_init) && size(f, ndims(f)) == 1 && size(u_init, ndims(u_init)) > 1
        broadcast_f = true;
    end
    % ^ TODO: Make more robust.

    % Determine if we are tracking gradients for deep learning
    use_autograd = false;
    if exist('dlarray', 'class') == 8 
        use_autograd = isa(u_init, 'dlarray') || isa(f, 'dlarray') || isa(dx, 'dlarray');
    end
    
    if use_autograd
        % Calculate total outputs expected by the user + solver.
        num_outputs = 1 + (has_v && nargout > 1);
        
        % Instantiate the custom differentiable operation.
        eikonalOp = eiko_lib.EikonalDifferentiableOp(is_3d, msfm, gated_x, has_v, broadcast_f, num_outputs);
        
        % Evaluate directly (bypassing dlfeval to allow custom op to run).
        if num_outputs > 1
            [u_out, v_out] = eikonalOp(u_init, f, v, dx);
            varargout{1} = v_out;
        else
            u_out = eikonalOp(u_init, f, v, dx);
        end
    else
        % Standard execution for regular numeric/gpuArray matrices.
        solver = eiko_lib.BatchedFIMSolver.get_solver(is_3d, false, msfm, has_v, gated_x);
        
        if has_v && nargout > 1
            [u_out, v_out] = solver.solve(u_init, f, v, [], dx, broadcast_f);
            varargout{1} = v_out;
        else
            u_out = solver.solve(u_init, f, v, [], dx, broadcast_f);
        end
    end
end
