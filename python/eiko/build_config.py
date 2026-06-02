# build_release.py
import os
import sys

# Check if we are building a release wheel
IS_RELEASE_BUILD = os.environ.get("EIKO_RELEASE_BUILD", "0") == "1"

if sys.platform == "win32":
    CXX_ARGS = [# '/std:c++17',
                '/Zc:preprocessor', '/DNOMINMAX', '/wd3189']
    NVCC_ARGS = [
        # '-std=c++17', 
        '-allow-unsupported-compiler', 
        '-Xcompiler', '/Zc:preprocessor', 
        '-Xcompiler', '/wd3189', 
        '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', 
        '-DNOMINMAX', 
        '--use_fast_math'
    ]
else:
    CXX_ARGS = [#'-std=c++17',
                '-O3']
    NVCC_ARGS = [#'-std=c++17', 
                 '--use_fast_math']

if IS_RELEASE_BUILD:
    NVCC_ARGS.append('-arch=all-major')
else:
    NVCC_ARGS.append('-arch=native')
EXTRA_INCLUDE_PATHS = [os.path.abspath("src")]
