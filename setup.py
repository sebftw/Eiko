import sys
import os
import shutil
import glob
from setuptools import setup, Extension
from setuptools.command.build_py import build_py

# ------------------------------------------------------------------------
# 1. Bootstrapping Build Config
# ------------------------------------------------------------------------
# Add the local 'python' directory to the path so we can import our 
# build machinery from build_config.py without installing first.
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

from eiko.build_config import (
    CXX_ARGS, 
    NVCC_ARGS, 
    EXTRA_INCLUDE_PATHS,
    IS_RELEASE_BUILD,
    get_full_version,
    get_jax_includes,
    compile_raw_nvcc_shared_lib
)

# ------------------------------------------------------------------------
# 2. Package Data Staging
# ------------------------------------------------------------------------
class CustomBuildPy(build_py):
    """
    Custom build step to copy necessary C++/CUDA source files into the 
    Python package staging area before building the wheel. This ensures 
    local JIT fallback environments have access to the raw headers.
    """
    def run(self):
        # Run the standard build_py first to set up the build_lib directory
        super().run()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(base_dir, 'src')
        
        # Target the temporary staging directory (self.build_lib) instead of the source tree
        dest_dir = os.path.join(self.build_lib, 'eiko', 'src')
        dest_bindings_dir = os.path.join(dest_dir, 'bindings')

        os.makedirs(dest_bindings_dir, exist_ok=True)

        # Copy all raw header files needed for JIT compiling
        for cuh_file in glob.glob(os.path.join(src_dir, '*.cuh')):
            shutil.copy(cuh_file, dest_dir)

        # Copy the binding entrypoints
        for b_file in ['torch_bindings.cu', 'jax_bindings.cu']:
            src_file = os.path.join(src_dir, 'bindings', b_file)
            if os.path.exists(src_file):
                shutil.copy(src_file, dest_bindings_dir)


# ------------------------------------------------------------------------
# 3. Extension Definitions & Routing
# ------------------------------------------------------------------------
ext_modules = []
cmdclass_dict = {"build_py": CustomBuildPy}
# If false, pip will install a pure-Python package (relies on JIT at runtime).
# If true, it triggers Ahead-of-Time (AOT) C++ compilation for the wheel.
if IS_RELEASE_BUILD:
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        
        # Define PyTorch Extension
        # We use PyTorch's native CUDAExtension because it perfectly handles 
        # the complexities of linking libtorch and libtorch_python.
        ext_modules.append(
            CUDAExtension(
                name="eiko.eiko_torch_impl",
                sources=["src/bindings/torch_bindings.cu"],
                extra_compile_args={"cxx": CXX_ARGS, "nvcc": NVCC_ARGS},
                include_dirs=EXTRA_INCLUDE_PATHS,
            )
        )
    except ImportError:
        print("\n[WARNING] PyTorch could not be found. Skipping PyTorch Eiko extension compilation.", file=sys.stderr)

    try:
        # Define JAX Extension
        # We define this as a standard setuptools Extension, but we will hijack 
        # its build process below so PyTorch doesn't try to link libtorch to it.
        ext_modules.append(
            Extension(
                name="eiko.eiko_jax_impl",
                sources=["src/bindings/jax_bindings.cu"],
                include_dirs=get_jax_includes(),
            )
        )
    except ImportError:
        print("\n[WARNING] JAX could not be found. Skipping JAX Eiko extension compilation.", file=sys.stderr)

    class MixedBuildExt(BuildExtension):
        """
        A traffic-cop build class. It routes extensions to different compilers
        based on their name, preventing dependency cross-contamination.
        """
        def build_extension(self, ext):
            if ext.name == "eiko.eiko_jax_impl":
                try:
                    # Intercept the JAX extension and compile it directly via raw NVCC
                    # in a subprocess, completely bypassing PyTorch's build machinery.
                    ext_path = self.get_ext_fullpath(ext.name)
                    compile_raw_nvcc_shared_lib(ext.sources[0], ext_path, ext.include_dirs)
                except Exception as e:
                    jax_ver = "unknown"
                    try:
                        import jax
                        jax_ver = jax.__version__
                    except ImportError:
                        pass
                    diagnose_build_failure(e, framework="JAX (AOT Wheel Build)", framework_version=jax_ver)
            else:
                try:
                    # Fall back to PyTorch's standard build logic for the torch extension
                    super().build_extension(ext)
                except Exception as e:
                    diagnose_build_failure(e, framework="PyTorch (AOT Wheel Build)", framework_version=torch.__version__)

    cmdclass_dict["build_ext"] = MixedBuildExt

# ------------------------------------------------------------------------
# 4. Final Setup Execution
# ------------------------------------------------------------------------
setup(
    version=get_full_version(),
    ext_modules=ext_modules,
    license="BSD-3-Clause",
    cmdclass=cmdclass_dict,
    # Ensure standard setuptools knows to bundle these files in the final wheel
    package_data={
        "eiko": [
            "src/*.cuh", 
            "src/bindings/*.cu"
        ],
    }
)
