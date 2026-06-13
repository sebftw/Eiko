import torch
import pytest
from eiko import eiko, eiko3d
import numpy as np
import jax
import jax.numpy as jnp
import math

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

# ==========================================
# Backend Helper Functions
# ==========================================

def cast_to_backend(arrays, backend):
    """
    Converts a list of NumPy arrays to the specified computational backend.
    Ensures that PyTorch tensors are placed on the GPU if available.
    """
    if backend == "torch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return [torch.tensor(arr, device=device) for arr in arrays]
    elif backend == "jax":
        return [jnp.array(arr) for arr in arrays]
    raise ValueError(f"Unknown backend: {backend}")

def extract_to_numpy(tensor, backend):
    """
    Converts a framework-specific tensor back to a standard NumPy array 
    to unify the final error calculation and validation logic.
    """
    if backend == "torch":
        return tensor.detach().cpu().numpy()
    elif backend == "jax":
        return np.array(tensor)
    raise ValueError(f"Unknown backend: {backend}")

# ==========================================
# Snell's Law Test
# ==========================================

@pytest.mark.parametrize("msfm", [True, False])
@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_eiko2d_snells_law(msfm, backend):
    """
    Validates refraction according to Snell's Law. 
    A planar wavefront is initialized in a top medium and propagates across a 
    horizontal interface into a bottom medium with a different speed of sound.
    The resulting wave angle in the bottom medium is compared against the analytical solution.
    """
    # 1. Domain and Physics Setup
    dim_y, dim_x = 201, 201
    dx = 0.01
    
    # Define the speeds of sound for the top (c1) and bottom (c2) media
    c1 = 1000.0
    c2 = 1500.0
    
    # Define the physical depth of the interface and its corresponding grid index
    y_int = 1.0  
    idx_int = int(y_int / dx)
    
    # Define the incident angle of the plane wave in the top medium
    theta1 = math.pi / 8  # 22.5 degrees
    sin_t1 = math.sin(theta1)
    cos_t1 = math.cos(theta1)
    
    # Calculate the expected refracted angle in the bottom medium using Snell's Law:
    # sin(theta2) / c2 = sin(theta1) / c1
    sin_t2 = (c2 / c1) * sin_t1
    
    # Ensure the incident angle does not exceed the critical angle, 
    # which would cause total internal reflection (unsupported by standard eikonal models)
    assert sin_t2 < 1.0, "Critical angle exceeded, total internal reflection will occur."
    cos_t2 = math.sqrt(1.0 - sin_t2**2)
    
    # Construct the physical coordinate grids for the domain
    y_coords = np.arange(dim_y, dtype=np.float32) * dx
    x_coords = np.arange(dim_x, dtype=np.float32) * dx
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # 2. Slowness Field Initialization
    # Initialize the slowness (1/c) for both media in pure NumPy
    f_np = np.empty((dim_y, dim_x), dtype=np.float32)
    f_np[:idx_int, :] = 1.0 / c1
    f_np[idx_int:, :] = 1.0 / c2
    
    # 3. Input Condition Initialization
    # Initialize the travel time array with infinity
    u_init_np = np.full((dim_y, dim_x), np.inf, dtype=np.float32)
    
    # Populate the entire top medium with the exact analytical travel times 
    # for an incoming plane wave. This establishes a continuous boundary condition 
    # pushing down onto the interface.
    u_top_analytical = (X * sin_t1 + Y * cos_t1) / c1
    u_init_np[:idx_int, :] = u_top_analytical[:idx_int, :]
    
    # 4. Backend Execution
    # Cast the initialized NumPy arrays to PyTorch or JAX
    u_init, f = cast_to_backend([u_init_np, f_np], backend)
    
    # Execute the solver
    if backend == "torch":
        with torch.no_grad():
            u_numerical = eiko(u_init, f, dx=dx, msfm=msfm)
    elif backend == "jax":
        u_numerical = eiko(u_init, f, dx=float(dx), msfm=msfm)
        
    # Return the computed field to NumPy for validation
    u_numerical_np = extract_to_numpy(u_numerical, backend)
        
    # 5. Validation
    # Construct the analytical solution for the refracted plane wave in the bottom medium.
    # The total travel time is the time accumulated traveling through the bottom medium 
    # plus the specific entry time at the interface.
    u_bottom_analytical = (X * sin_t2 + (Y - y_int) * cos_t2) / c2 + (X * sin_t1 + y_int * cos_t1) / c1
    
    error_map = np.abs(u_numerical_np - u_bottom_analytical)
    
    # The finite grid domain introduces edge diffractions because the initialized plane wave 
    # abruptly ends at the left and right boundaries (x=0 and x=dim_x). 
    # To isolate and validate pure refraction, we crop the validation region to the 
    # center 50% of the bottom medium, away from the interface and lateral edges.
    c_start, c_end = int(0.25 * dim_x), int(0.75 * dim_x)
    test_region = error_map[idx_int + 5 : -5, c_start : c_end]
    
    max_error = np.max(test_region)
    
    # The expected error bounds scale with the grid spacing (dx) and the highest slowness (1/c1).
    tol = 60.0 * (dx / c1)
    # ^ High tolerance to account for the jump-discontinuity at the interface.
    
    assert max_error <= tol, (
        f"Snell's Law failed on {backend} (msfm={msfm})! "
        f"Expected error <= {tol:.4e}, got {max_error:.4e}"
    )

def test_eikonal_advection_plane_wave():
    """
    Validates that a transversely varying v_init field is pulled 
    along the characteristics without diffusing across them.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 50
    dx_val = 0.1
    
    # 1. Setup Plane Wave Source (Left Boundary)
    u_init = torch.full((N, N), float('inf'), dtype=torch.float32, device=device)
    u_init[:, 0] = 0.0  
    
    # 2. Setup Transverse v_init
    # v varies from -1.0 to 1.0 along the Y-axis (rows), but starts at X=0.
    v_init = torch.zeros((N, N), dtype=torch.float32, device=device)
    v_source = torch.linspace(-1.0, 1.0, steps=N, device=device)
    v_init[:, 0] = v_source
    
    # Constant slowness ensures characteristics are perfectly horizontal (X-direction)
    f = torch.ones((N, N), dtype=torch.float32, device=device)
    
    # 3. Execute Solver
    with torch.no_grad():
        # Adjust this call to match your high-level Python wrapper's exact signature
        u_out, v_out = eiko(u_init, f, v_init=v_init, dx=dx_val)
        
    # 4. Verify Travel Time (u)
    # The wave should just be the X-coordinate distance.
    x_coords = torch.arange(N, dtype=torch.float32, device=device) * dx_val
    u_expected = x_coords.unsqueeze(0).expand(N, N)
    
    max_u_error = torch.max(torch.abs(u_out - u_expected)).item()
    assert max_u_error <= 1e-5, f"u_out plane wave failed! Max error: {max_u_error:.4e}"
    
    # 5. Verify Advection (v)
    # Because rays travel perfectly left-to-right, every column should be an 
    # exact copy of the source column. There should be zero vertical mixing.
    v_expected = v_source.unsqueeze(1).expand(N, N)
    
    max_v_error = torch.max(torch.abs(v_out - v_expected)).item()
    assert max_v_error <= 1e-4, f"v_out advection failed! Max error: {max_v_error:.4e}"

@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_eikonal_analytical_gradients_1d(backend):
    """
    Validates the analytical gradients of the 1D Eikonal solver for u_init, f, and dx.
    Ensures gradients correctly flow back through the upwind logic on both PyTorch and JAX.
    """
    N = 10
    dx_val = 0.1
    f_val = 1.5
    
    # 1. Initialize variables in standard NumPy
    u_init_np = np.full((1, N), np.inf, dtype=np.float32)
    u_init_np[0, 0] = 0.0  
    
    f_np = np.full((1, N), f_val, dtype=np.float32)
    
    # 2. Compute Gradients based on Backend
    if backend == "torch":
        # Cast to PyTorch and enable autograd
        u_init, f = cast_to_backend([u_init_np, f_np], backend)
        dx = torch.tensor(dx_val, dtype=torch.float32, device=u_init.device)
        
        u_init.requires_grad_(True)
        f.requires_grad_(True)
        dx.requires_grad_(True)
        
        # Forward Pass
        u_out = eiko(u_init, f, dx=dx, msfm=False)
        
        # Backward Pass
        loss = u_out[0, -1]
        loss.backward()
        
        # Extract gradients back to NumPy
        grad_u_np = extract_to_numpy(u_init.grad, backend)
        grad_f_np = extract_to_numpy(f.grad, backend)
        grad_dx_np = extract_to_numpy(dx.grad, backend)

    elif backend == "jax":
        # Cast to JAX arrays
        u_init, f = cast_to_backend([u_init_np, f_np], backend)
        
        # Define a pure functional forward pass for JAX autodiff
        def loss_fn(u_in, f_in):
            u_out = eiko(u_in, f_in, dx=dx_val, msfm=False)
            return u_out[0, -1]

        # jax.grad computes gradients w.r.t specified arguments (0=u_in, 1=f_in)
        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        grad_u, grad_f = grad_fn(u_init, f)
        
        # Extract gradients back to NumPy
        grad_u_np = extract_to_numpy(grad_u, backend)
        grad_f_np = extract_to_numpy(grad_f, backend)
        # grad_dx_np = extract_to_numpy(grad_dx, backend)
        
    # ==========================================
    # 3. Verify Gradients Analytically
    # ==========================================
    
    # --- A. Gradient of u_init ---
    expected_grad_u = np.zeros_like(u_init_np)
    expected_grad_u[0, 0] = 1.0
    assert np.allclose(grad_u_np, expected_grad_u), f"u_init gradient failed on {backend}!"
    
    # --- B. Gradient of f ---
    # Assuming standard right-sided upwind: u[i] = u[i-1] + f[i]*dx
    expected_grad_f = np.full((1, N), dx_val, dtype=np.float32)
    expected_grad_f[0, 0] = 0.0  
    assert np.allclose(grad_f_np, expected_grad_f), f"f gradient failed on {backend}!"
    
    # --- C. Gradient of dx ---
    # We traverse (N-1) nodes, each adding f_val to the gradient.
    expected_grad_dx_val = f_val * (N - 1)
    expected_grad_dx = np.array(expected_grad_dx_val, dtype=np.float32)
    
    if backend == "torch":
        # Allow a tiny absolute tolerance for floating point summation drift
        assert np.allclose(grad_dx_np, expected_grad_dx, atol=1e-6), f"dx gradient failed on {backend}!"
