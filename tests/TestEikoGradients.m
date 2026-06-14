classdef TestEikoGradients < matlab.unittest.TestCase
    % Validates the analytical gradients of the Eikonal solver.
    
    methods (Test)
        function test_eikonal_analytical_gradients_1d(testCase)
            
            N = 10;
            dx_val = single(0.1);
            f_val = single(1.5);
            
            % 1. Initialize standard numeric arrays [N, 1] instead of [1, N]
            u_init_num = inf(N, 1, 'single');
            u_init_num(1) = 0.0;  
            
            f_num = repmat(f_val, N, 1);
            
            % 2. Cast to dlarray
            u_init = dlarray(u_init_num);
            f = dlarray(f_num);
            dx = dlarray(dx_val);
            
            % 3. Evaluate Gradients via dlfeval
            [~, grad_u_dl, grad_f_dl, grad_dx_dl] = dlfeval(@eikoLoss, u_init, f, dx);
            
            grad_u = extractdata(grad_u_dl);
            grad_f = extractdata(grad_f_dl);
            grad_dx = extractdata(grad_dx_dl);
            
            % ==========================================
            % 4. Verify Gradients Analytically
            % ==========================================
            
            % --- A. Gradient of u_init ---
            expected_grad_u = zeros(N, 1, 'single');
            expected_grad_u(1) = 1.0;
            
            testCase.verifyEqual(double(gather(grad_u)), double(expected_grad_u), 'AbsTol', 1e-6, ...
                'u_init gradient failed!');
            
            % --- B. Gradient of f ---
            expected_grad_f = repmat(dx_val, N, 1);
            expected_grad_f(1) = 0.0;
            
            testCase.verifyEqual(double(gather(grad_f)), double(expected_grad_f), 'AbsTol', 1e-6, ...
                'f gradient failed!');
            
            % --- C. Gradient of dx ---
            expected_grad_dx = f_val * single(N - 1);
            
            testCase.verifyEqual(double(gather(grad_dx)), double(expected_grad_dx), 'AbsTol', 1e-6, ...
                'dx gradient failed!');
        end

        % ==========================================
        % Numerical Finite Difference Test
        % ==========================================
        function test_eikonal_numerical_gradients_1d(testCase)
            N = 10;
            dx_val = single(0.1);
            f_val = single(1.5);
            epsilon = single(1e-3);
            
            % 1. Initialize standard numeric arrays
            u_init_num = inf(N, 1, 'single');
            u_init_num(1) = 0.0;
            
            f_num = repmat(f_val, N, 1);
            
            % 2. Compute Automatic Differentiation Gradients
            u_init_dl = dlarray(u_init_num);
            f_dl = dlarray(f_num);
            dx_dl = dlarray(dx_val);
            
            [~, grad_u_dl, grad_f_dl, grad_dx_dl] = dlfeval(@eikoLoss, u_init_dl, f_dl, dx_dl);
            
            grad_u_ad = extractdata(grad_u_dl);
            grad_f_ad = extractdata(grad_f_dl);
            grad_dx_ad = extractdata(grad_dx_dl);
            
            % ==========================================
            % 3. Compute Numerical Gradients (Central Difference)
            % ==========================================
            
            % --- A. Numerical Gradient of u_init ---
            grad_u_num = zeros(N, 1, 'single');
            for i = 1:N
                % Only perturb the finite source node. Perturbing Inf does nothing.
                if isinf(u_init_num(i))
                    continue;
                end
                u_plus = u_init_num; u_plus(i) = u_plus(i) + epsilon;
                u_minus = u_init_num; u_minus(i) = u_minus(i) - epsilon;
                
                loss_plus = computeNumLoss(u_plus, f_num, dx_val);
                loss_minus = computeNumLoss(u_minus, f_num, dx_val);
                grad_u_num(i) = (loss_plus - loss_minus) / (2 * epsilon);
            end
            
            % --- B. Numerical Gradient of f ---
            grad_f_num = zeros(N, 1, 'single');
            for i = 1:N
                f_plus = f_num; f_plus(i) = f_plus(i) + epsilon;
                f_minus = f_num; f_minus(i) = f_minus(i) - epsilon;
                
                loss_plus = computeNumLoss(u_init_num, f_plus, dx_val);
                loss_minus = computeNumLoss(u_init_num, f_minus, dx_val);
                grad_f_num(i) = (loss_plus - loss_minus) / (2 * epsilon);
            end
            
            % --- C. Numerical Gradient of dx ---
            dx_plus = dx_val + epsilon;
            dx_minus = dx_val - epsilon;
            loss_plus = computeNumLoss(u_init_num, f_num, dx_plus);
            loss_minus = computeNumLoss(u_init_num, f_num, dx_minus);
            grad_dx_num = (loss_plus - loss_minus) / (2 * epsilon);
            
            % ==========================================
            % 4. Compare AD to Numerical
            % ==========================================
            
            % We use 1e-3 tolerance since single precision finite differences 
            % naturally exhibit floating point drift compared to exact autodiff.
            
            testCase.verifyEqual(double(gather(grad_u_ad)), double(grad_u_num), 'AbsTol', tol, ...
                'u_init AD gradient does not match numerical approximation!');
                
            testCase.verifyEqual(double(gather(grad_f_ad)), double(grad_f_num), 'AbsTol', tol, ...
                'f AD gradient does not match numerical approximation!');
                
            testCase.verifyEqual(double(gather(grad_dx_ad)), gather(double(grad_dx_num)), 'AbsTol', tol, ...
                'dx AD gradient does not match numerical approximation!');
        end
    end
end

% ==========================================
% Helper Functions for Automatic Differentiation
% ==========================================

function [loss, grad_u, grad_f, grad_dx] = eikoLoss(u_in, f_in, dx_in)
    msfm = false;
    u_out = eiko(u_in, f_in, dx_in, 'msfm', msfm);
    
    % Access the last element in the column vector
    loss = u_out(end, 1);
    
    [grad_u, grad_f, grad_dx] = dlgradient(loss, u_in, f_in, dx_in);
end

function loss = computeNumLoss(u_in, f_in, dx_in)
    % Evaluates the solver purely numerically without autodiff tracking
    msfm = false;
    u_out = eiko(u_in, f_in, dx_in, 'msfm', msfm);
    loss = u_out(end, 1);
end