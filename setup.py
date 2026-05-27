from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from build_config import CXX_ARGS, NVCC_ARGS, EXTRA_INCLUDE_PATHS

# Map source file relative to the project root
sources = ["src/bindings/torch_bindings.cu"]

setup(
    ext_modules=[
        CUDAExtension(
            name="eiko.eiko_torch_impl",  # Important: Must match the import name!
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
