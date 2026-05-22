import sys

def animate_eikonal(u, v=1.0, color_map='gray', video_filename='', 
                    title='EIKONAL WAVEFRONT', pulse_width=80.0, 
                    speed=0.5, overlay=None, outline=None, 
                    style='real', render_mode='slice'):
    """
    ANIMATE_EIKONAL Visualizes Eikonal equation travel times as an animated wave.
    """
    
    # Required dependencies to run the visualization.
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        raise ImportError(
            "The animation utility requires 'numpy' and 'matplotlib'. "
            "Please install it using: pip install \"eiko[examples]\""
        )
    
    # Optional dependency for contours/isosurfaces
    try:
        from skimage import measure
        HAS_SKIMAGE = True
    except ImportError:
        HAS_SKIMAGE = False
        print("[Eiko] Warning: 'scikit-image' not found. Contours, overlays, and 3D isosurfaces will be disabled.", file=sys.stderr)
    
    # --- Data Type & Device Checks (e.g., PyTorch -> Numpy) ---
    if hasattr(u, 'cpu'): u = u.detach().cpu().numpy()
    if hasattr(v, 'cpu'): v = v.detach().cpu().numpy()
    if hasattr(overlay, 'cpu'): overlay = overlay.detach().cpu().numpy()
    if hasattr(outline, 'cpu'): outline = outline.detach().cpu().numpy()

    is_3d = u.ndim == 3
    
    # --- Safe Data Initialization ---
    valid_u = u[np.isfinite(u)]
    if len(valid_u) == 0:
        print("Warning: Travel time field u is completely Inf. Nothing to animate.", file=sys.stderr)
        return

    maxv = np.max(valid_u)
    u_safe = np.copy(u).astype(np.float32)
    u_safe[np.isinf(u_safe)] = maxv + pulse_width

    # Handle Permutation and setup for 3D
    if is_3d:
        # In Numpy, typical 3D ordering is [Z, Y, X]. We keep it as is, 
        # but map slices to dimensions.
        Nz, Ny, Nx = u.shape
        v_anim = v
    else:
        v_anim = v
        Ny, Nx = u.shape

    # --- Figure & Axis Setup ---
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    
    if style.lower() == 'real':
        clim = [-1, 1]
    elif style.lower() == 'abs':
        clim = [0, 1]
    elif style.lower() == 'db':
        clim = [-30, 0]
        db_ref = 20 * np.log10(1.0 + 1e-12) # +1e-12 prevents log10(0)

    # Setup Render Objects
    if is_3d:
        # 3D Axes
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X (Lateral)')
        ax.set_ylabel('Y (Elevation)')
        ax.set_zlabel('Z (Depth)')
        ax.invert_zaxis() # Ultrasound convention: Depth increases downwards
        
        # We'll handle the 3D drawing inside the update loop, but prepare indices
        sz, sy, sx = Nz//2, Ny//2, Nx//2
        iso_val = -6 if style.lower() == 'db' else 0.5
        
    else:
        # 2D Rendering
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=16, fontweight='bold', fontname='serif')
        ax.set_xlabel('X (Lateral)', fontsize=14, fontname='serif')
        ax.set_ylabel('Z (Depth)', fontsize=14, fontname='serif')
        
        # Initialize empty image
        im = ax.imshow(np.zeros((Ny, Nx)), vmin=clim[0], vmax=clim[1], cmap=color_map, aspect='equal')
        
        # In matplotlib, imshow puts (0,0) at top-left, which inherently matches 
        # the ultrasound convention (Depth going down).
        
        if outline is not None and HAS_SKIMAGE:
            ax.contour(outline, levels=[0.5], colors='k', linewidths=2)
            
        if overlay is not None and HAS_SKIMAGE:
            contours = measure.find_contours(overlay, 0.5)
            for contour in contours:
                # contour is (row, col) -> (Y, X)
                ax.fill(contour[:, 1], contour[:, 0], color='red', alpha=0.3, edgecolor='r', linewidth=2.5)

    freq = 6 * np.pi
    
    # --- Animation Update Logic ---
    def compute_frame(t):
        """Calculates the analytical wave packet for a given time step."""
        diff_t = u_safe - t
        valid_mask = np.abs(diff_t) <= (pulse_width / 2.0)
        
        img = np.zeros_like(u_safe, dtype=np.complex64)
        
        if np.any(valid_mask):
            d_valid = diff_t[valid_mask]
            
            # Analytical Hanning envelope
            envelope = 0.5 * (1.0 + np.cos(2 * np.pi * d_valid / pulse_width))
            
            # Apply envelope and carrier phase
            img[valid_mask] = envelope * np.exp(1j * freq * d_valid / pulse_width)
            
        img = img * v_anim
        
        # Apply Style
        if style.lower() == 'real':
            final_img = np.real(img)
        elif style.lower() == 'abs':
            final_img = np.abs(img)
        elif style.lower() == 'db':
            final_img = 20 * np.log10(np.abs(img) + 1e-12) - db_ref
            
        return final_img

    # time_steps = np.arange(0, maxv + pulse_width, speed)
    start_time = np.min(u_safe)
    time_steps = np.arange(start_time, maxv + pulse_width, speed)

    # Note: Global references for 3D plot collections so they can be removed/updated
    frame_objs = [] 

    
    def update(frame_idx):
        t = time_steps[frame_idx]
        final_img = compute_frame(t)
        
        #if frame_idx % 20 == 0: # Check every 20 frames to avoid spamming
        #    print(f"Frame {frame_idx} (t={t:.1f}): Data min/max = {final_img.min():.3f} / {final_img.max():.3f}")

        if is_3d:
            # Clear all collections (slices or isosurfaces) from the previous frame.
            for coll in ax.collections:
                coll.remove()
            
            # Define levels to force contouring (20 steps is usually enough for smooth waves)
            levels = np.linspace(clim[0], clim[1], 20)
            if render_mode == 'slice':
                # Re-draw the slices.
                X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
                ax.contourf(X, Y, final_img[sz, :, :], zdir='z', offset=sz, cmap=color_map, vmin=clim[0], vmax=clim[1], levels=levels)
                
                Y, Z = np.meshgrid(np.arange(Ny), np.arange(Nz))
                ax.contourf(final_img[:, :, sx], Y, Z, zdir='x', offset=sx, cmap=color_map, vmin=clim[0], vmax=clim[1], levels=levels)
                
                X, Z = np.meshgrid(np.arange(Nx), np.arange(Nz))
                ax.contourf(X, final_img[:, sy, :], Z, zdir='y', offset=sy, cmap=color_map, vmin=clim[0], vmax=clim[1], levels=levels)
                
            elif render_mode == 'isosurface' and HAS_SKIMAGE:
                try:
                    # Use marching cubes to find the isosurface.
                    verts, faces, _, _ = measure.marching_cubes(final_img, level=iso_val)
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    mesh = Poly3DCollection(verts[faces], alpha=0.7)
                    mesh.set_facecolor([0.2, 0.6, 1.0])
                    mesh.set_edgecolor('none')
                    ax.add_collection3d(mesh)
                except ValueError:
                    pass # Fails cleanly if the volume doesn't contain the isovalue yet
            
            return ax.collections
        else:
            im.set_data(final_img)
            return [im]

    # --- Run / Export Animation ---
    # Blit=True massively speeds up 2D rendering by only redrawing changed pixels.
    # Blit is not well-supported for matplotlib 3D axes.
    anim = FuncAnimation(fig, update, frames=len(time_steps), 
                                   interval=1000/60, blit=not is_3d)

    if video_filename:
        print(f"Exporting video to {video_filename} (this may take a moment)...")
        # Ensure you have ffmpeg installed on your Ubuntu machine (sudo apt install ffmpeg)
        writer = animation.FFMpegWriter(fps=60, bitrate=2000)
        anim.save(video_filename, writer=writer)
        print("Video export complete.")
    else:
        plt.show()

    return anim