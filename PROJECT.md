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
A few features that would be nice to have:

* **Double-Precision Floating-Point Support:** Eiko currently operates exclusively in single-precision (32-bit). While double-precision is slower, particularly on gaming GPUs, it can be helpful for numerical validation, such as estimating rounding errors or validating Eiko against numerical (finite difference) gradients.
* **Robust Backward Pass:** The forward pass (computing time-of-flight) is robust and deterministic due to monotonic convergence: time of flight decreases every iteration. This makes it easier to parallelize. Conversely, the backward pass (computing gradients) currently exhibits non-monotonic behavior because input residuals may be positive or negative. The simple solution is to perform two backward passes: one for positive and one for negative residuals. Then, the two results can be summed since the backward pass is linear. While this doubles the memory requirements, it guarantees deterministic convergence and likely accelerates the backward pass.
* **Auto-Tuning:** The code is currently optimized for a `NVIDIA GeForce RTX 5070 Ti` GPU, with extensive tuning only for the to 2D input case. A future update could include an auto-tuning function that lets users optimize Eiko for their specific hardware, alongside out-of-the-box presets for various GPU generations to improve baseline performance.
* **Beamforming:** Because Eiko can compute time-of-flight and apodizations based on electronic emission delays without relying on geometric distance equations, it can seamlessly handle arbitrary or unconventional emission sequences inside non-uniform media. This black-box functionality pairs well with a beamformer, allowing users to image through acoustic lenses or aberrations without deriving any time-of-flight equations manually.
* **Simulation Framework:** Eiko's inputs (a sound speed map and initial delays) closely mirror those of simulation tools like Field II and k-Wave. Future examples could simulate RF data and let Eiko compute the time-of-flight and apodization. Combining these outputs allows an image to be beamformed with ideal aberration correction.
* **More Comparisons:** Add more comparisons with other libraries (maybe TomoATT, if comparable).

A few features which probably will probably never be added:
* **Non-GPU implementation:** CPUs are slow, so maintaining a separate codebase for CPU-only users makes little sense.




