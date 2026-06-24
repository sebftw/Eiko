## Project Layout
The files are as follows:
```
Eiko/
├── README.md       # Primary project description
├── PROJECT.md      # This file
├── THEORY.md       # Mathematical background and Eiko algorithm details
├── LICENSE         # License information
├── src/            # Core CUDA C++ implementation and API interfaces
├── matlab/         # MATLAB Eiko library
├── python/         # Python Eiko library
├── examples/       # Example scripts (e.g., tomography, lens design)
│   ├── matlab/     #   MATLAB examples
│   └── python/     #   Python examples
├── tests/          # Automated testing and validation scripts
├── images/         # Various image assets
├── .github/        # Build and release automation (Github Actions)
├── pyproject.toml  # Python package configuration (for pip install)
├── setup.py        # Python package build script
└── tox.ini         # Python test automation configuration
```

## Future Plans
A few features that would be nice to add:

* **Double-Precision Floating-Point Support:** Eiko currently operates exclusively in single-precision (32-bit). While double-precision computation will incur a performance penalty, particularly on gaming GPUs, it is helpful for numerical validation, such as estimating rounding errors or validating Eiko against numerical gradients.
* **Robust Backward Pass:** The forward pass (computing time-of-flight) is highly robust and deterministic due to monotonic convergence: time of flight decreases every iteration. Conversely, the backward pass (computing gradients) currently exhibits non-monotonic behavior because residuals can be positive or negative. The simple solution is to compute two separate backward passes: one for positive and one for negative residuals, and then sum the results (the backward pass is linear). While this doubles memory requirements, it will guarantee deterministic convergence and likely accelerate the backward pass.
* **Auto-Tuning:** Current optimizations are primarily tuned for the `NVIDIA GeForce RTX 5070 Ti`, with extensive tuning limited to 2D cases. A future update could include an auto-tuning function that lets users optimize Eiko for their specific hardware, alongside out-of-the-box presets for various GPU generations to improve baseline performance.
* **Differentiable Beamforming:** Because Eiko computes time-of-flight and apodizations based on electronic emission delays without relying on closed-form geometric distance equations, it can seamlessly handle arbitrary or unconventional emission sequences and non-uniform media. This black-box functionality will pair well with a fully differentiable beamformer, allowing users to image through acoustic lenses or aberrations without deriving any time-of-flight equations manually.
* **Simulation Framework:** Eiko's required inputs (sound speed maps and initial delays) closely mirror those of simulation tools like Field II and k-Wave. Future examples could simulate RF data and let Eiko compute the time-of-flight and apodization. Combining these outputs allows an image to be beamformed with ideal aberration correction.


A few features which probably will never be added:
* **Non-GPU implementation:** CPUs are slow, so maintaining a separate codebase for CPU-only users makes little sense.




