import sys
import os
import shutil
import glob
import subprocess
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_py import build_py

# Add 'python' to sys.path so we can import 'eiko.build_config'
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

from eiko.build_config import CXX_ARGS, NVCC_ARGS, EXTRA_INCLUDE_PATHS

class CustomBuildPy(build_py):
    """Custom build step to copy necessary CUDA source files into the package."""
    def run(self):
        # Run the standard build_py first to set up the build_lib directory
        super().run()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(base_dir, 'src')
        
        # Target the temporary staging directory (self.build_lib) instead of the source tree
        dest_dir = os.path.join(self.build_lib, 'eiko', 'src')
        dest_bindings_dir = os.path.join(dest_dir, 'bindings')

        # Ensure destination directories exist in the build tree
        os.makedirs(dest_bindings_dir, exist_ok=True)

        # Copy src/*.cuh
        for cuh_file in glob.glob(os.path.join(src_dir, '*.cuh')):
            shutil.copy(cuh_file, dest_dir)

        # Copy specific binding files
        binding_files = ['torch_bindings.cu', 'jax_bindings.cu']
        for b_file in binding_files:
            src_file = os.path.join(src_dir, 'bindings', b_file)
            if os.path.exists(src_file):
                shutil.copy(src_file, dest_bindings_dir)

# 1. Determine Build Mode
local_version = os.environ.get("EIKO_LOCAL_VERSION", "")
IS_CI_BUILD = bool(local_version)

ext_modules = []
cmdclass_dict = {"build_py": CustomBuildPy}

if IS_CI_BUILD:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    print(f"\n[Eiko setup.py] CI environment detected (Suffix: {local_version}).")
    print("[Eiko setup.py] Compiling PyTorch and JAX AOT extensions...\n")
    
    # Mandate JAX and pybind11 availability for CI builds
    import jax
    import pybind11
    
    # Define PyTorch Extension (Leaves PyTorch dependency intact for Torch users)
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
    
    # Define JAX Extension as a STANDARD setuptools Extension
    jax_includes = EXTRA_INCLUDE_PATHS + [pybind11.get_include(), sysconfig.get_path("include")]
    ext_modules.append(
        Extension(
            name="eiko.eiko_jax_impl",
            sources=["src/bindings/jax_bindings.cu"],
            include_dirs=jax_includes,
        )
    )

    # Custom Build class to route compiling logic based on the extension name
    class MixedBuildExt(BuildExtension):
        # TODO: Also use this class inside eiko.eiko_jax.
        def build_extension(self, ext):
            if ext.name == "eiko.eiko_jax_impl":
                # Bypass PyTorch completely and use raw NVCC (adapted from your JIT script)
                ext_path = self.get_ext_fullpath(ext.name)
                os.makedirs(os.path.dirname(ext_path), exist_ok=True)
                
                nvcc_path = shutil.which("nvcc")
                if not nvcc_path:
                    fallback_path = "/usr/local/cuda/bin/nvcc"
                    if os.path.exists(fallback_path):
                        nvcc_path = fallback_path
                    else:
                        raise FileNotFoundError("nvcc not found in PATH for JAX compilation.")

                # Force -fPIC for shared library compilation
                cxx_args_pic = CXX_ARGS + ["-fPIC"] if "-fPIC" not in CXX_ARGS else CXX_ARGS
                include_flags = [f"-I{path}" for path in ext.include_dirs if os.path.exists(path)]

                cmd = [nvcc_path, "-shared", "-std=c++17", ext.sources[0], "-o", ext_path]
                cmd += NVCC_ARGS
                cmd += [f"-Xcompiler={arg}" for arg in cxx_args_pic]
                cmd += include_flags

                print(f"[Eiko] Compiling JAX extension via raw nvcc...\nCMD: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            else:
                # Force C++17 for PyTorch bindings
                def remove_cxx_20_flag(compiler_args):
                    bad_flags = ['-std=c++20', '/std:c++20', '-std=c++14', '/std:c++14']
                    return [arg for arg in compiler_args if arg not in bad_flags]

                if hasattr(ext, 'extra_compile_args') and isinstance(ext.extra_compile_args, dict):
                    if 'cxx' in ext.extra_compile_args:
                        ext.extra_compile_args['cxx'] = remove_cxx_20_flag(ext.extra_compile_args['cxx'])
                    if 'nvcc' in ext.extra_compile_args:
                        ext.extra_compile_args['nvcc'] = remove_cxx_20_flag(ext.extra_compile_args['nvcc'])
                
                # Fall back to standard PyTorch BuildExtension logic for eiko_torch_impl
                super().build_extension(ext)

    cmdclass_dict["build_ext"] = MixedBuildExt

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
    cmdclass=cmdclass_dict,
    package_data={
        "eiko": [
            "src/*.cuh", 
            "src/bindings/torch_bindings.cu", 
            "src/bindings/jax_bindings.cu"
        ],
    }
)