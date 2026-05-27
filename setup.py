# setup.py
import sys
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Add 'python' to sys.path so we can import 'eiko.build_config'
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

from eiko.build_config import CXX_ARGS, NVCC_ARGS, EXTRA_INCLUDE_PATHS

sources = ["src/bindings/torch_bindings.cu"]

setup(
    ext_modules=[
        CUDAExtension(
            name="eiko.eiko_torch_impl",
            sources=sources,
            extra_compile_args={
                "cxx": CXX_ARGS,
                "nvcc": NVCC_ARGS,
            },
            include_dirs=EXTRA_INCLUDE_PATHS,
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
