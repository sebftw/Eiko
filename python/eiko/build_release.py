# build_release.py
import os

# Define your flags here
CXX_ARGS = ["-O3"]
NVCC_ARGS = ["-O3", "--use_fast_math"]
EXTRA_INCLUDE_PATHS = [os.path.abspath("src")]
