classdef EikonalDifferentiableOp < deep.DifferentiableFunction
    % =========================================================================
    % CLASS: EikonalDifferentiableOp
    % A custom differentiable operation for solving the Eikonal equation.
    % This class wraps a fast Eikonal solver CUDA kernel, enabling it to
    % be used within optimization loops or deep neural networks in MATLAB.
    %
    % You're not not supposed to use this class. Just call solve_eikonal,
    % which will correctly use this class for you.
    % =========================================================================
    
    properties(SetAccess = immutable)
        is_3d        % (logical) Whether the input is 3D (true) or 2D (false).
        msfm         % (logical) Whether to use the Multi-Stencil Fast Marching method.
        gated_x      % (logical) Whether to restrict propagation positively along the first data dimension.
        has_v        % (logical) Whether advection should also be computed.
        broadcast_f  % (logical) Whether f is broadcasted (one f applied to every input u in the batch).
        num_outputs  % (numeric) Number of outputs expected from the forward pass.
    end

    % Cache solvers privately to avoid instantiation overhead on each pass.
    properties (Access = private)
        forward_solver
        backward_solver
    end
    
    methods
        function obj = EikonalDifferentiableOp(is_3d, msfm, gated_x, has_v, broadcast_f, num_outputs)
            % -------------------------------------------------------------------------
            % Constructor: EikonalDifferentiableOp
            % Initializes the differentiable Eikonal solver operation and caches solvers.
            %
            % Inputs:
            %   is_3d       - True if input is 3D.
            %   msfm        - True to use multi-stencil fast marching.
            %   gated_x     - True to gate propagation along the axial axis.
            %   has_v       - True if computing advection.
            %   broadcast_f - True to broadcast the speed function across the batch.
            %   num_outputs - Number of outputs (1 for u_out, 2 for u_out + v_out).
            % -------------------------------------------------------------------------
            
            arguments
                is_3d       (1,1) logical
                msfm        (1,1) logical
                gated_x     (1,1) logical
                has_v       (1,1) logical
                broadcast_f (1,1) logical
                num_outputs (1,1) double
            end

            % Initialize the differentiable function object.
            % We instruct MATLAB to cache inputs and outputs for the backward pass automatically.
            obj@deep.DifferentiableFunction(num_outputs, ...
                'SaveInputsForBackward', true, ...
                'SaveOutputsForBackward', true);
            
            obj.is_3d = is_3d;
            obj.msfm = msfm;
            obj.gated_x = gated_x;
            obj.has_v = has_v;
            obj.broadcast_f = broadcast_f;
            obj.num_outputs = num_outputs;

            % Cache solvers during initialization (assumes they are essentially stateless, which they are).
            obj.forward_solver  = eiko_lib.BatchedFIMSolver.get_solver(is_3d, false, msfm, has_v, gated_x);
            obj.backward_solver = eiko_lib.BatchedFIMSolver.get_solver(is_3d, true,  msfm, false, gated_x);
            % Note: ^ Backwards pass does NOT support advection (has_v = false).
        end
        
        function [u_out, varargout] = forward(obj, u_init, f, v, dx)
            % -------------------------------------------------------------------------
            % Method: forward
            % Executes the forward pass of the Eikonal solver.
            %
            % Inputs:
            %   u_init - Matrix representing the initial conditions/boundary values.
            %   f      - Matrix representing the slowness function (f = 1/c).
            %            This is the right-hand side of the Eikonal equation: |∇u| = f.
            %   v      - Matrix representing the advected field.
            %   dx     - Scalar representing the grid spacing (step size).
            %
            % Outputs:
            %   u_out     - The computed arrival times / solution to the Eikonal equation.
            %   varargout - Optional secondary output (e.g., v_out) returned if has_v 
            %               is true and the network expects >1 output.
            % -------------------------------------------------------------------------

            % Execute the standard forward pass using the cached solver.
            if obj.has_v && obj.num_outputs > 1
                [u_out, v_out] = obj.forward_solver.solve(u_init, f, v, [], dx, obj.broadcast_f);
                varargout{1} = v_out;
            else
                u_out = obj.forward_solver.solve(u_init, f, v, [], dx, obj.broadcast_f);
            end
        end
        
        function [grad_u_init, grad_f, grad_v, grad_dx] = backward(obj, grad_u, varargin)
            % -------------------------------------------------------------------------
            % Method: backward
            % Executes the backward (adjoint) pass to compute gradients for backpropagation.
            %
            % Inputs:
            %   grad_u   - Gradient of the loss with respect to the primary output (u_out).
            %   varargin - Implicitly packed arguments provided by MATLAB's autodiff engine:
            %              Format: [grad_v_out (opt)], computeGradients, u_init, f, v, dx, u_out
            %              * computeGradients: logical array mapping to inputs [u_init, f, v, dx]
            %
            % Outputs:
            %   grad_u_init - Gradient w.r.t. the initial conditions (u_init).
            %   grad_f      - Gradient w.r.t. the speed function (f).
            %   grad_v      - Gradient w.r.t. the advection tensor (v).
            %   grad_dx     - Gradient w.r.t. the grid spacing (dx).
            % -------------------------------------------------------------------------
            
            % Determine the starting index in varargin. If the forward pass
            % produced multiple outputs, MATLAB passes their gradients first.
            base_idx = 1;
            if obj.num_outputs > 1 
               base_idx = 2; % Shift to skip grad_v_out
            end
            
            % Extract the boolean flags and the cached forward variables.
            computeGradients = varargin{base_idx};
            % u_init         = varargin{base_idx + 1}; % Unused in backward logic
            f                = varargin{base_idx + 2};
            % v              = varargin{base_idx + 3}; % Unused in backward logic
            dx               = varargin{base_idx + 4};
            u_out            = varargin{base_idx + 5};
            
            % Initialize target gradients as empty arrays.
            grad_u_init = []; grad_f = []; grad_v = []; grad_dx = [];
            
            % Gradient computation of v is not supported (for good reason).
            if computeGradients(3)
                error('EikonalOp:Backward', 'Gradient with respect to v is not supported/implemented.');
            end
            
            % Analytical Gradient w.r.t Step Size via Euler's Homogeneity Theorem
            if computeGradients(4)
                grad_dx = sum(grad_u .* u_out, 'all') / dx;
            end
            
            % Run backward kernel and capture returned adjoint output
            if computeGradients(1) || computeGradients(2)
                lambda = zeros(size(u_out), 'like', u_out);
                
                % Solve the adjoint equation.
                lambda = obj.backward_solver.solve(lambda, grad_u, [], u_out, dx, false);  
                % Note: ^ We DO NOT broadcast "f" here, as f is the residual (grad_u), in this context.
                
                % Assign gradient for the initial condition.
                if computeGradients(1)
                    grad_u_init = lambda; 
                end
                
                % Compute gradient for the slowness function (f).
                if computeGradients(2)
                    % OPTIMIZED: Scalar squaring first, then array multiplication
                    grad_f = (dx * dx) * (lambda .* f);
                    
                    if obj.broadcast_f
                        batch_dim = 3 + obj.is_3d; 
                        grad_f = sum(grad_f, batch_dim); 
                    end
                end
            end
        end
    end
end