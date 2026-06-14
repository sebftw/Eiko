classdef TestEiko < matlab.unittest.TestCase
    % Validates the 2D and 3D eiko solvers with MATLAB dimensional ordering.
    
    properties (TestParameter)
        shape2d = {[31, 31], [1, 31], [31, 1], [1, 1]};
        shape3d = {[31, 31, 31], [1, 31, 31], [31, 1, 31], [31, 31, 1], [1, 1, 1]};
        batch_size = {0, 1, 4};
        msfm = {true, false};
    end
    
    methods (Test)
        
        % ==========================================
        % 2D Solver Tests
        % ==========================================
        function test_eiko2d_constant_speed_of_sound(testCase, shape2d, batch_size, msfm)
            dx = 0.001;
            % Reversing spatial dimension assignments for MATLAB
            dim_1 = shape2d(2);
            dim_0 = shape2d(1); 
            
            center_1 = floor(dim_1 / 2) + 1;
            center_0 = floor(dim_0 / 2) + 1;
            
            coords_1 = ((1:dim_1) - center_1) * dx;
            coords_0 = ((1:dim_0) - center_0) * dx;
            
            [Grid_1, Grid_0] = ndgrid(coords_1, coords_0);
            R = sqrt(Grid_1.^2 + Grid_0.^2);
            
            if batch_size > 0
                if batch_size == 4
                    c_values = [1400.0; 1500.0; 1540.0; 1600.0];
                else
                    c_values = 1540.0;
                end
                
                % Shape: [1, 1, B]
                c_view = reshape(c_values, 1, 1, batch_size);
                
                % Shape: [dim_1, dim_0, B]
                f = 1.0 ./ repmat(c_view, dim_1, dim_0, 1);
                
                u_analytical = repmat(R, 1, 1, batch_size) ./ c_view;
                
                u_init = inf([dim_1, dim_0, batch_size], 'single');
                u_init(center_1, center_0, :) = 0.0;
            else
                c_val = 1540.0;
                c_values = c_val;
                
                f = repmat(1.0 / c_val, dim_1, dim_0);
                u_analytical = R / c_val;
                
                u_init = inf([dim_1, dim_0], 'single');
                u_init(center_1, center_0) = 0.0;
            end
            
            u_numerical = eiko(u_init, f, dx, 'msfm', msfm);
            
            error_map = abs(u_numerical - u_analytical);
            tolerances = 1.5 * (dx ./ c_values);
            
            if batch_size > 0
                for i = 1:batch_size
                    % Slicing the trailing dimension
                    slice_error = error_map(:, :, i);
                    max_error = max(slice_error(:));
                    tol = tolerances(i);
                    testCase.verifyLessThanOrEqual(max_error, tol, ...
                        sprintf('2D Batched %d failed! Error %.4e > %.4e', i, max_error, tol));
                end
            else
                max_error = max(error_map(:));
                tol = tolerances(1);
                testCase.verifyLessThanOrEqual(max_error, tol, ...
                    sprintf('2D Unbatched failed! Error %.4e > %.4e', max_error, tol));
            end
        end
        
        % ==========================================
        % 3D Solver Tests
        % ==========================================
        function test_eiko3d_constant_speed_of_sound(testCase, shape3d, batch_size, msfm)
            dx = 0.001;
            dim_2 = shape3d(3);
            dim_1 = shape3d(2);
            dim_0 = shape3d(1);
            
            center_2 = floor(dim_2 / 2) + 1;
            center_1 = floor(dim_1 / 2) + 1;
            center_0 = floor(dim_0 / 2) + 1;
            
            coords_2 = ((1:dim_2) - center_2) * dx;
            coords_1 = ((1:dim_1) - center_1) * dx;
            coords_0 = ((1:dim_0) - center_0) * dx;
            
            [Grid_2, Grid_1, Grid_0] = ndgrid(coords_2, coords_1, coords_0);
            R = sqrt(Grid_2.^2 + Grid_1.^2 + Grid_0.^2);
            
            if batch_size > 0
                if batch_size == 4
                    c_values = [1400.0; 1500.0; 1540.0; 1600.0];
                else
                    c_values = 1540.0;
                end
                
                % Shape: [1, 1, 1, B]
                c_view = reshape(c_values, 1, 1, 1, batch_size);
                f = 1.0 ./ repmat(c_view, dim_2, dim_1, dim_0, 1);
                
                u_analytical = repmat(R, 1, 1, 1, batch_size) ./ c_view;
                
                u_init = inf([dim_2, dim_1, dim_0, batch_size], 'single');
                u_init(center_2, center_1, center_0, :) = 0.0;
            else
                c_val = 1540.0;
                c_values = c_val;
                
                f = repmat(1.0 / c_val, dim_2, dim_1, dim_0);
                u_analytical = R / c_val;
                
                u_init = inf([dim_2, dim_1, dim_0], 'single');
                u_init(center_2, center_1, center_0) = 0.0;
            end
            
            u_numerical = eiko3d(u_init, f, dx, 'msfm', msfm);
            
            error_map = abs(u_numerical - u_analytical);
            tolerances = 1.75 * (dx ./ c_values);
            
            if batch_size > 0
                for i = 1:batch_size
                    slice_error = error_map(:, :, :, i);
                    max_error = max(slice_error(:));
                    tol = tolerances(i);
                    testCase.verifyLessThanOrEqual(max_error, tol, ...
                        sprintf('3D Batched %d failed! Error %.4e > %.4e', i, max_error, tol));
                end
            else
                max_error = max(error_map(:));
                tol = tolerances(1);
                testCase.verifyLessThanOrEqual(max_error, tol, ...
                    sprintf('3D Unbatched failed! Error %.4e > %.4e', max_error, tol));
            end
        end

        % ==========================================
        % Snell's Law Test
        % ==========================================
        function test_eiko2d_snells_law(testCase, msfm)
            dim_x = 201;
            dim_y = 201;
            dx = 0.01;
            
            c1 = 1000.0;
            c2 = 1500.0;
            
            y_int = 1.0;
            idx_int = floor(y_int / dx) + 1;
            
            theta1 = pi / 8;
            sin_t1 = sin(theta1);
            cos_t1 = cos(theta1);
            
            sin_t2 = (c2 / c1) * sin_t1;
            testCase.verifyTrue(sin_t2 < 1.0, 'Critical angle exceeded.');
            cos_t2 = sqrt(1.0 - sin_t2^2);
            
            coords_x = (0:dim_x-1) * dx;
            coords_y = (0:dim_y-1) * dx;
            [X, Y] = ndgrid(coords_x, coords_y);
            
            % Reversed array initialization [X, Y]
            f = zeros(dim_x, dim_y, 'single');
            f(:, 1:idx_int-1) = 1.0 / c1;
            f(:, idx_int:end) = 1.0 / c2;
            
            u_init = inf(dim_x, dim_y, 'single');
            u_top_analytical = (X .* sin_t1 + Y .* cos_t1) / c1;
            u_init(:, 1:idx_int-1) = u_top_analytical(:, 1:idx_int-1);
            
            u_numerical = eiko(u_init, f, dx, 'msfm', msfm);
            
            u_bottom_analytical = (X .* sin_t2 + (Y - y_int) .* cos_t2) / c2 + ...
                                  (X .* sin_t1 + y_int .* cos_t1) / c1;
            
            error_map = abs(u_numerical - u_bottom_analytical);
            
            c_start = floor(0.25 * dim_x) + 1;
            c_end = floor(0.75 * dim_x);
            % Slicing along the X-dimension (dim 1)
            test_region = error_map(c_start : c_end, idx_int + 5 : end - 5);
            
            max_error = max(test_region(:));
            tol = 60.0 * (dx / c1);
            
            testCase.verifyLessThanOrEqual(max_error, tol, ...
                sprintf('Snells Law failed (msfm=%d)! Expected <= %.4e, got %.4e', msfm, tol, max_error));
        end
        
    end
end