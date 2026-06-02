# setup.py
import sys
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Add 'python' to sys.path so we can import 'eiko.build_config'
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

from eiko.build_config import CXX_ARGS, NVCC_ARGS, EXTRA_INCLUDE_PATHS

# 1. Define the base PyTorch Extension
ext_modules = [
    CUDAExtension(
        name="eiko.eiko_torch_impl",
        sources=["src/bindings/torch_bindings.cu"],
        extra_compile_args={
            "cxx": CXX_ARGS,
            "nvcc": NVCC_ARGS,
        },
        include_dirs=EXTRA_INCLUDE_PATHS,
    )
]

# 2. Conditionally attempt to add the JAX Extension
try:
    import jax
    import pybind11
    
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
    print("\n[Eiko setup.py] JAX environment detected. Queuing JAX extension for build.\n")
    
except ImportError:
    print("\n[Eiko setup.py] JAX or pybind11 not found. Skipping JAX extension build.\n")


# 3. Process Dynamic Version
def get_base_version():
    # Read __init__.py as plain text to avoid the import paradox
    import re
    init_path = os.path.join(os.path.dirname(__file__), "python", "eiko", "__init__.py")
    with open(init_path, "r") as f:
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.M)
        if match:
            return match.group(1)
    return "0.0.0"

# Get the clean base version (e.g., "0.1.0")
base_version = get_base_version()

# Get the CI environment suffix (e.g., "+pt2.12.0cu126")
local_version = os.environ.get("EIKO_LOCAL_VERSION", "")

# Combine them into a PEP-440 compliant string (e.g., "0.1.0+pt2.12.0cu126")
full_version = f"{base_version}{local_version}"


# 4. Execute setup
setup(
    version=full_version, # Injected into the wheel filename and metadata
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension
    }
)
