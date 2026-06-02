# setup.py
import sys
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Add 'python' to sys.path so we can import 'eiko.build_config'
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

from eiko import __version__
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
base_version = __version__
local_version = os.environ.get("EIKO_LOCAL_VERSION", "")
full_version = f"{base_version}{local_version}"


# 4. Execute setup
setup(
    version=full_version, # Injected into the wheel filename and metadata
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension
    }
)
