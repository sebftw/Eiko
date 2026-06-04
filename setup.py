import sys
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Add 'python' to sys.path so we can import 'eiko.build_config'
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

from eiko.build_config import CXX_ARGS, NVCC_ARGS, EXTRA_INCLUDE_PATHS

# 1. Determine Build Mode
# If EIKO_LOCAL_VERSION is set, we are in the CI wheel-building pipeline.
local_version = os.environ.get("EIKO_LOCAL_VERSION", "")
IS_CI_BUILD = bool(local_version)

ext_modules = []

if IS_CI_BUILD:
    print(f"\n[Eiko setup.py] CI environment detected (Suffix: {local_version}).")
    print("[Eiko setup.py] Compiling PyTorch and JAX AOT extensions...\n")
    
    # Mandate JAX and pybind11 availability for CI builds
    import jax
    import pybind11
    
    # Define PyTorch Extension
    ext_modules.append(
        CUDAExtension(
            name="eiko.eiko_torch_impl",
            sources=["src/bindings/torch_bindings.cu"],
            extra_compile_args={
                "cxx": CXX_ARGS,
                "nvcc": NVCC_ARGS,
            },
            include_dirs=EXTRA_INCLUDE_PATHS,
        )
    )
    
    # Define JAX Extension
    jax_includes = EXTRA_INCLUDE_PATHS + [pybind11.get_include()]
    ext_modules.append(
        CUDAExtension(
            name="eiko.eiko_jax_impl",
            sources=["src/bindings/jax_bindings.cu"],
            extra_compile_args={
                "cxx": CXX_ARGS,
                "nvcc": NVCC_ARGS,
            },
            include_dirs=jax_includes,
        )
    )

# 2. Process Dynamic Version
def get_base_version():
    import re
    init_path = os.path.join(os.path.dirname(__file__), "python", "eiko", "__init__.py")
    with open(init_path, "r") as f:
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.M)
        if match:
            return match.group(1)
    return "0.0.0"

base_version = get_base_version()
full_version = f"{base_version}{local_version}"


# 3. Execute setup
setup(
    version=full_version,
    ext_modules=ext_modules,
    license="BSD-3-Clause",
    cmdclass={
        "build_ext": BuildExtension
    } if IS_CI_BUILD else {}  # Avoid forcing torch dependency checks on basic sdist installs
)
