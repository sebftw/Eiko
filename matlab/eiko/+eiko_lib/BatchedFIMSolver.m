classdef BatchedFIMSolver < handle
    % =========================================================================
    % CLASS: BatchedFIMSolver
    % A lightweight wrapper for the C++ MEX binding of the Fast Iterative Method.
    % Manages the memory lifecycle (pointer creation and destruction).
    %
    % You're not not supposed to use this class. Just call solve_eikonal,
    % which will correctly use this class for you.
    % =========================================================================

    properties (Access = private)
        solver_ptr   (1,1) uint64   % Pointer to the underlying C++ object.
        is_3d        (1,1) logical  % Dimensionality flag.
        is_backward  (1,1) logical  % Whether this is the backward (adjoint) solver.
        msfm         (1,1) logical  % Multi-stencil flag.
        has_v        (1,1) logical  % Advection flag.
        gated_x      (1,1) logical  % Gated propagation flag.
    end
    
    methods
        function obj = BatchedFIMSolver(is_3d, is_backward, msfm, has_v, gated_x)
            % -----------------------------------------------------------------
            % Constructor: Instantiates the C++ solver and stores the pointer.
            % -----------------------------------------------------------------
            arguments
                is_3d       (1,1) logical
                is_backward (1,1) logical
                msfm        (1,1) logical
                has_v       (1,1) logical
                gated_x     (1,1) logical
            end
            
            obj.is_3d = is_3d;
            obj.is_backward = is_backward;
            obj.msfm = msfm;
            obj.has_v = has_v;
            obj.gated_x = gated_x;
            
            % Call MEX to instantiate C++ object and get pointer
            obj.solver_ptr = eiko_lib.mex_bindings('new', obj.is_3d, obj.is_backward, ...
                                             obj.msfm, obj.has_v, obj.gated_x);
        end
        
        function varargout = solve(obj, u, f, v, tof, dx, broadcast_f)
            % -----------------------------------------------------------------
            % Method: solve
            % Executes the MEX kernel. Uses an arguments block to strictly
            % enforce single precision before passing data to C++ backend.
            % -----------------------------------------------------------------
            arguments
                obj
                u           single
                f           single
                v           single
                tof         single
                dx          (1,1) single
                broadcast_f (1,1) logical
            end
            
            [varargout{1:nargout}] = ...
                eiko_lib.mex_bindings('solve', obj.solver_ptr, u, f, v, tof, ...
                                dx, broadcast_f, obj.is_3d, obj.is_backward, ...
                                obj.msfm, obj.has_v, obj.gated_x);
        end
        
        
        function delete(obj)
            % -----------------------------------------------------------------
            % Destructor: free C++ GPU memory when the object is cleared.
            % -----------------------------------------------------------------
            if obj.solver_ptr ~= 0
                eiko_lib.mex_bindings('delete', obj.solver_ptr, obj.is_3d, ...
                                obj.is_backward, obj.msfm, obj.has_v, obj.gated_x);
                obj.solver_ptr = uint64(0);
            end
        end
    end

    methods (Static)
        function solver = get_solver(is_3d, is_backward, msfm, has_v, gated_x)
            % =====================================================================
            % STATIC METHOD: get_solver
            % Singleton factory pattern. Caches initialized solver instances in
            % memory to avoid the overhead of crossing the MEX boundary repeatedly.
            % =====================================================================

            % Initialize the static map on the first call.
            persistent solvers;
            if isempty(solvers)
                solvers = containers.Map('KeyType', 'char', 'ValueType', 'any');
            end
            
            % Generate a unique string hash for these solver settings.
            key = sprintf('%d_%d_%d_%d_%d', is_3d, is_backward, msfm, has_v, gated_x);
            
            % If the solver doesn't exist, create and cache it.
            if ~isKey(solvers, key)
                solvers(key) = eiko_lib.BatchedFIMSolver(is_3d, is_backward, msfm, has_v, gated_x);
            end
            
            % Return cached solver.
            solver = solvers(key);
        end
    end
end