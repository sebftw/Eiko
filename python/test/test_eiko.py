import torch
import pytest
from eiko import eiko, eiko3d

# ==========================================
# 2D Solver Tests
# ==========================================

# Parameterize over the spatial grid shapes:
# - (31, 31) is a standard 2D plane.
# - (1, 31) and (31, 1) test the 2D solver's handling of 1D lines (singletons) 
#   to ensure implicit matrix operations don't collapse size-1 dimensions.
@pytest.mark.parametrize("spatial_shape", [(31, 31), (1, 31), (31, 1), (1, 1)])
# Parameterize over None (Unbatched 2D), B=1 (Batched 3D, single item), and B=4 (Batched 3D)
@pytest.mark.parametrize("batch_size", [None, 1, 4])
@pytest.mark.parametrize("msfm", [True, False])
def test_eiko2d_constant_speed_of_sound(spatial_shape, batch_size, msfm):
    """Validates the 2D Eiko solver against an analytical point-source solution."""
    
    # 1. Domain Setup
    device = torch.device("cuda")
    dx = 0.001          
    dim_0, dim_1 = spatial_shape
    
    # Determine the index of the point source (center of the domain)
    center_0, center_1 = dim_0 // 2, dim_1 // 2
    
    # 2. Coordinate Grid Construction
    # By subtracting (center * dx), we shift the coordinate system so the center 
    # index lies precisely at physical coordinate 0.0.
    coords_0 = torch.arange(dim_0, dtype=torch.float32, device=device) * dx - (center_0 * dx)
    coords_1 = torch.arange(dim_1, dtype=torch.float32, device=device) * dx - (center_1 * dx)
    
    # indexing='ij' ensures the meshgrid axes match the (dim_0, dim_1) array shapes, 
    # rather than transposing to the Cartesian (x, y) default.
    Grid_0, Grid_1 = torch.meshgrid(coords_0, coords_1, indexing='ij')
    
    # Physical distance from the point source for every node in the grid.
    R = torch.sqrt(Grid_0**2 + Grid_1**2)
    
    # 3. Input Initialization (Batched vs. Unbatched)
    if batch_size is not None:
        # --- BATCHED CASE (3D: B x H x W) ---
        # Provide varied speeds of sound to ensure batch slices aren't intermingling.
        if batch_size == 4:
            c_values = torch.tensor([1400.0, 1500.0, 1540.0, 1600.0], dtype=torch.float32, device=device)
        else: # batch_size == 1
            c_values = torch.tensor([1540.0], dtype=torch.float32, device=device)
        
        # Reshape to (B, 1, 1) so PyTorch can broadcast the division over the spatial dims.
        c_view = c_values.view(batch_size, 1, 1)
        
        # Slowness field (f = 1/c). We expand it to explicitly form the (B, H, W) tensor.
        f = 1.0 / c_view.expand(batch_size, dim_0, dim_1)
        
        # Travel time = Distance / Speed.
        # R is (H, W). R.unsqueeze(0) becomes (1, H, W). Divided by (B, 1, 1) -> (B, H, W).
        u_analytical = R.unsqueeze(0) / c_view
        
        # Set all nodes to infinity except the point source (set to 0.0) across all batches.
        u_init = torch.full((batch_size, dim_0, dim_1), float('inf'), dtype=torch.float32, device=device)
        u_init[:, center_0, center_1] = 0.0
        
    else:
        # --- UNBATCHED CASE (2D: H x W) ---
        c_val = 1540.0
        c_values = torch.tensor([c_val], dtype=torch.float32, device=device)
        
        f = torch.full((dim_0, dim_1), 1.0 / c_val, dtype=torch.float32, device=device)
        u_analytical = R / c_val
        
        u_init = torch.full((dim_0, dim_1), float('inf'), dtype=torch.float32, device=device)
        u_init[center_0, center_1] = 0.0

    # 4. Solver Execution
    with torch.no_grad():
        u_numerical = eiko(u_init, f, dx=dx, msfm=msfm)
        
    # 5. Validation
    error_map = torch.abs(u_numerical - u_analytical)
    
    # In a first-order scheme, the numerical error scales with the grid spacing (dx)
    # and the slowness (1/c). The multiplier 1.5 safely bounds the expected error in 2D.
    tolerances = 1.5 * (dx / c_values)
    
    if batch_size is not None:
        # Evaluate errors per batch slice for exact debugging if a test fails.
        for i in range(batch_size):
            max_error = torch.max(error_map[i]).item()
            tol = tolerances[i].item()
            assert max_error <= tol, (
                f"2D Shape {spatial_shape}, Batch {i} failed! Error {max_error:.4e} > {tol:.4e}"
            )
    else:
        max_error = torch.max(error_map).item()
        tol = tolerances[0].item()
        assert max_error <= tol, (
            f"2D Shape {spatial_shape}, Unbatched failed! Error {max_error:.4e} > {tol:.4e}"
        )


# ==========================================
# 3D Solver Tests
# ==========================================

# (31, 31, 31) is a standard 3D cube.
# The remaining shapes test the 3D solver's handling of 2D planes (singletons).
# The geometric logic mirrors the 2D tests identically.
@pytest.mark.parametrize("spatial_shape", [
    (31, 31, 31), 
    ( 1, 31, 31), 
    (31,  1, 31), 
    (31, 31,  1),
    ( 1,  1,  1)
])
@pytest.mark.parametrize("batch_size", [None, 1, 4])
@pytest.mark.parametrize("msfm", [True, False])
def test_eiko3d_constant_speed_of_sound(spatial_shape, batch_size, msfm):
    """Validates the 3D Eiko solver for standard 3D cubes and 2D singletons."""
    
    device = torch.device("cuda")
    dx = 0.001          
    dim_0, dim_1, dim_2 = spatial_shape
    center_0, center_1, center_2 = dim_0 // 2, dim_1 // 2, dim_2 // 2
    
    coords_0 = torch.arange(dim_0, dtype=torch.float32, device=device) * dx - (center_0 * dx)
    coords_1 = torch.arange(dim_1, dtype=torch.float32, device=device) * dx - (center_1 * dx)
    coords_2 = torch.arange(dim_2, dtype=torch.float32, device=device) * dx - (center_2 * dx)
    
    Grid_0, Grid_1, Grid_2 = torch.meshgrid(coords_0, coords_1, coords_2, indexing='ij')
    R = torch.sqrt(Grid_0**2 + Grid_1**2 + Grid_2**2)
    
    if batch_size is not None:
        if batch_size == 4:
            c_values = torch.tensor([1400.0, 1500.0, 1540.0, 1600.0], dtype=torch.float32, device=device)
        else:
            c_values = torch.tensor([1540.0], dtype=torch.float32, device=device)
        
        c_view = c_values.view(batch_size, 1, 1, 1)
        f = 1.0 / c_view.expand(batch_size, dim_0, dim_1, dim_2)
        u_analytical = R.unsqueeze(0) / c_view
        
        u_init = torch.full((batch_size, dim_0, dim_1, dim_2), float('inf'), dtype=torch.float32, device=device)
        u_init[:, center_0, center_1, center_2] = 0.0
    else:
        c_val = 1540.0
        c_values = torch.tensor([c_val], dtype=torch.float32, device=device)
        f = torch.full((dim_0, dim_1, dim_2), 1.0 / c_val, dtype=torch.float32, device=device)
        u_analytical = R / c_val
        
        u_init = torch.full((dim_0, dim_1, dim_2), float('inf'), dtype=torch.float32, device=device)
        u_init[center_0, center_1, center_2] = 0.0

    with torch.no_grad():
        u_numerical = eiko3d(u_init, f, dx=dx, msfm=msfm)
        
    error_map = torch.abs(u_numerical - u_analytical)
    
    # 3D grid diagonal is sqrt(3) * dx, so numerical error can be slightly higher than 2D.
    tolerances = 1.75 * (dx / c_values)
    
    if batch_size is not None:
        for i in range(batch_size):
            max_error = torch.max(error_map[i]).item()
            tol = tolerances[i].item()
            assert max_error <= tol, (
                f"3D Shape {spatial_shape}, Batch {i} failed! Error {max_error:.4e} > {tol:.4e}"
            )
    else:
        max_error = torch.max(error_map).item()
        tol = tolerances[0].item()
        assert max_error <= tol, (
            f"3D Shape {spatial_shape}, Unbatched failed! Error {max_error:.4e} > {tol:.4e}"
        )

def test_eiko2d_default_dx():
    """Tests that the 2D solver works correctly when dx is omitted (defaults to 1.0)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 11  # Small grid for a quick test
    center = N // 2
    
    # Slowness f = 1.0 (speed of sound c = 1.0)
    f = torch.ones((N, N), dtype=torch.float32, device=device)
    
    u_init = torch.full((N, N), float('inf'), dtype=torch.float32, device=device)
    u_init[center, center] = 0.0
    
    # Analytical solution is just the Euclidean index distance
    coords = torch.arange(N, dtype=torch.float32, device=device) - center
    Grid_0, Grid_1 = torch.meshgrid(coords, coords, indexing='ij')
    u_analytical = torch.sqrt(Grid_0**2 + Grid_1**2)
    
    # Execute WITHOUT the dx argument
    with torch.no_grad():
        u_numerical = eiko(u_init, f)
        
    max_error = torch.max(torch.abs(u_numerical - u_analytical)).item()
    
    # Tolerance is 1.5 * (dx / c) = 1.5 * (1.0 / 1.0) = 1.5
    assert max_error <= 1.5, f"2D Default dx test failed! Max error: {max_error:.4e}"


def test_eiko3d_default_dx():
    """Tests that the 3D solver works correctly when dx is omitted (defaults to 1.0)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 11
    center = N // 2
    
    f = torch.ones((N, N, N), dtype=torch.float32, device=device)
    
    u_init = torch.full((N, N, N), float('inf'), dtype=torch.float32, device=device)
    u_init[center, center, center] = 0.0
    
    coords = torch.arange(N, dtype=torch.float32, device=device) - center
    Grid_0, Grid_1, Grid_2 = torch.meshgrid(coords, coords, coords, indexing='ij')
    u_analytical = torch.sqrt(Grid_0**2 + Grid_1**2 + Grid_2**2)
    
    # Execute WITHOUT the dx argument
    with torch.no_grad():
        u_numerical = eiko3d(u_init, f)
        
    max_error = torch.max(torch.abs(u_numerical - u_analytical)).item()
    
    # Tolerance is 1.75 * (dx / c) = 1.75 * (1.0 / 1.0) = 1.75
    assert max_error <= 1.75, f"3D Default dx test failed! Max error: {max_error:.4e}"

# TODO: Add JAX tests, gradient tests, and advection tests.
