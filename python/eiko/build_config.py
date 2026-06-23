import re
import os
import sys
import glob
import uuid
import shutil
import platform
import subprocess
from pathlib import Path

link_emoji = ""  # Fallback for older Windows terminals/CP1252
try:
    "\U0001f517".encode(sys.stdout.encoding or "utf-8")
    link_emoji = "\U0001f517"
except UnicodeEncodeError:
    pass

# ------------------------------------------------------------------------
# 1. Base Compiler Arguments
# ------------------------------------------------------------------------
NVCC_ARGS = ["-O3", "--use_fast_math"]

if sys.platform == "win32":
    # Microsoft Visual C++ uses /O2 for maximum speed
    CXX_ARGS = ["/O2"]
else:
    # GCC/Clang uses -O3
    CXX_ARGS = ["-O3"]

# ------------------------------------------------------------------------
# 2. Shared Binary Cache Directory
# ------------------------------------------------------------------------
if sys.platform == "win32":
    _cache_base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
else:
    _cache_base = Path.home() / ".cache"

BIN_CACHE_DIR = str(_cache_base / "eiko" / "binaries")

# ------------------------------------------------------------------------
# 3. Source Path Resolution
# ------------------------------------------------------------------------
# Base directory where this configuration file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Option A: Path when the package is installed via pip (src is inside eiko/).
SRC_DIR_INSTALLED = os.path.join(BASE_DIR, 'src')

# Option B: Path when running locally during development (src is sibling to python/).
SRC_DIR_DEV = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'src'))

if os.path.exists(SRC_DIR_INSTALLED):
    SRC_DIR = SRC_DIR_INSTALLED
elif os.path.exists(SRC_DIR_DEV):
    SRC_DIR = SRC_DIR_DEV
else:
    raise FileNotFoundError(
        f"Could not find Eiko C++/CUDA source directory. Tried:\n"
        f"1. {SRC_DIR_INSTALLED}\n"
        f"2. {SRC_DIR_DEV}"
    )

EXTRA_INCLUDE_PATHS = [SRC_DIR]

# ------------------------------------------------------------------------
# 4. NVCC Discovery & Dynamic CUDA_HOME Bootstrapping
# ------------------------------------------------------------------------
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or os.environ.get("CUDA_ROOT")

# Initial fallback to PyTorch's native resolution
if not cuda_home:
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        cuda_home = CUDA_HOME
    except ImportError:
        pass

def _resolve_best_nvcc() -> str:
    """
    Searches system paths, environment variables, and standard installation 
    directories to find all available nvcc executables. Returns the path 
    to the compiler with the highest version number.
    """
    candidates = set()
    ext = ".exe" if sys.platform == "win32" else ""

    # Strategy 1: Environment Variables & Pre-resolved CUDA_HOME
    if cuda_home and os.path.exists(os.path.join(cuda_home, "bin", f"nvcc{ext}")):
        candidates.add(os.path.realpath(os.path.join(cuda_home, "bin", f"nvcc{ext}")))

    # Strategy 2: System Path (which / where)
    path_nvcc = shutil.which("nvcc")
    if path_nvcc:
        candidates.add(os.path.realpath(path_nvcc))

    # Strategy 3: Common Default Installation Paths
    if sys.platform == "win32":
        base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.exists(base):
            for path in glob.glob(os.path.join(base, "v*", "bin", "nvcc.exe")):
                candidates.add(os.path.realpath(path))
    else:
        for base in ["/usr/local", "/opt"]:
            if os.path.exists(base):
                for path in glob.glob(os.path.join(base, "cuda*", "bin", "nvcc")):
                    candidates.add(os.path.realpath(path))

    if not candidates:
        raise FileNotFoundError(
            "nvcc compiler not found in PATH, CUDA_HOME, or standard directories. "
            "Please ensure the NVIDIA CUDA toolkit is installed."
        )

    # Evaluate all candidates; best match wins.
    best_nvcc = ""
    max_ver = (-1,)  # Tuple for reliable semantic version comparison

    for current_path in candidates:
        try:
            # Suppress console popups on Windows during background subprocess execution
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            result = subprocess.run(
                [current_path, "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True, 
                check=True,
                startupinfo=startupinfo
            )
            
            match = re.search(r"release (\d+(?:\.\d+)+)", result.stdout)
            if match:
                # Convert "12.2" -> (12, 2) or "11.8.89" -> (11, 8, 89)
                current_ver = tuple(map(int, match.group(1).split(".")))
                if current_ver > max_ver:
                    max_ver = current_ver
                    best_nvcc = current_path
        except (subprocess.CalledProcessError, OSError, ValueError):
            continue

    # Fallback to the first found candidate if version parsing fails entirely
    if not best_nvcc:
        return sorted(list(candidates))[0]

    return best_nvcc


# --- EXECUTE DISCOVERY ONCE UPON IMPORT ---
NVCC_PATH = None
try:
    NVCC_PATH = _resolve_best_nvcc()
    # If we didn't have a CUDA_HOME, derive it by going up two directories from bin/nvcc
    if not cuda_home and NVCC_PATH:
        cuda_home = os.path.dirname(os.path.dirname(NVCC_PATH))
    
    # Enforce it globally so PyTorch's cpp_extension picks it up natively
    if cuda_home:
        os.environ["CUDA_HOME"] = cuda_home
except FileNotFoundError:
    pass  # Fail silently here; the runtime JIT fallback will catch it and explain.

def get_nvcc_executable() -> str:
    """Returns the pre-resolved nvcc executable path, or raises if missing."""
    if not NVCC_PATH:
        raise FileNotFoundError(
            "nvcc compiler not found in PATH, CUDA_HOME, or standard directories. "
            "Please ensure the NVIDIA CUDA toolkit is installed."
        )
    return NVCC_PATH

# If found, add NVIDIA's Core Compute Libraries (Thrust, CUB, libcudacxx)
if cuda_home and os.path.exists(cuda_home):
    cccl_base = os.path.join(cuda_home, 'include', 'cccl')
    if os.path.exists(cccl_base):
        EXTRA_INCLUDE_PATHS.extend([
            cccl_base,
            os.path.join(cccl_base, 'thrust'),
            os.path.join(cccl_base, 'libcudacxx', 'include'),
            os.path.join(cccl_base, 'cub'),
        ])

# ------------------------------------------------------------------------
# 5. OS & Release Specific Flags
# ------------------------------------------------------------------------
local_version = os.environ.get("EIKO_LOCAL_VERSION", "")
IS_RELEASE_BUILD = bool(local_version)

if sys.platform == "win32":
    CXX_ARGS.extend([
        '/permissive',
        '/EHsc', '/MD', '/DVERSION_INFO', '/DTHRUST_IGNORE_CUB_VERSION_CHECK', 
        '/DTHRUST_FORCE_COMPATIBILITY', '/Zc:preprocessor', '/DNOMINMAX',
        '/D_ENABLE_EXTENDED_ALIGNED_STORAGE'
    ])
    NVCC_ARGS.extend([
        '-allow-unsupported-compiler', 
        '-D_WIN32=1', '-DUSE_CUDA=1',
        '-Xcompiler', '/Zc:preprocessor', 
        '-Xcompiler', '/permissive',
        '-DTHRUST_IGNORE_CUB_VERSION_CHECK', 
        '-DTHRUST_FORCE_COMPATIBILITY',
        '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', 
        '-DNOMINMAX',
        '-D_ENABLE_EXTENDED_ALIGNED_STORAGE',
        '-Xcompiler', '/MD,/EHsc'
    ])
else:
    # Safely detect the ABI flag from the active PyTorch installation
    try:
        import torch
        abi_val = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
    except ImportError:
        # Fallback to "0" (or "1" depending on your project's primary target) if torch isn't installed yet
        abi_val = "0"

    CXX_ARGS.extend([f'-D_GLIBCXX_USE_CXX11_ABI={abi_val}', '-fPIC', '-fvisibility=hidden'])
    NVCC_ARGS.extend(['-Xcompiler', f'-D_GLIBCXX_USE_CXX11_ABI={abi_val}', '-Xcompiler', '-fPIC', '-Xcompiler', '-Wno-deprecated-declarations'])

# Handle release/debug configurations and warning suppressions
if IS_RELEASE_BUILD:
    NVCC_ARGS.append('-arch=all-major')
    
    if sys.platform == "win32":
        # MSVC-specific warning suppression
        CXX_ARGS.extend(['/wd4101'])  # Unused variable
        
        # NVCC on Windows
        NVCC_ARGS.extend([
            '-Xcudafe', '--diag_suppress=3189',  # "module" keyword
            '-Xcudafe', '--diag_suppress=177',   # Unused variable
            '-Xcudafe', '--diag_suppress=550',   # Variable was set but never used
            '-Xcompiler', '/wd4101'              # Pass unused variable to host compiler
        ])
    else:
        # GCC/Clang specific warning suppression
        CXX_ARGS.extend(['-Wno-unused-variable'])
        
        # NVCC on Linux/macOS
        NVCC_ARGS.extend([
            '-Xcompiler', '-Wno-deprecated-declarations',
            '-Xcompiler', '-Wno-unused-variable',
            '-Xcudafe', '--diag_suppress=3189',  # "module" keyword
            '-Xcudafe', '--diag_suppress=550',    # Variable was set but never used in CUDA files
            '-Xcudafe', '--diag_suppress=177'    # Unused variable in CUDA files
        ])
else:
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "native"
    NVCC_ARGS.append('-arch=native')

# ------------------------------------------------------------------------
# 6. Build Machinery & Centralized Diagnostics
# ------------------------------------------------------------------------
# Version Resolution
def get_full_version() -> str:
    init_path = os.path.join(BASE_DIR, "__init__.py")
    base_version = "0.0.0"
    with open(init_path, "r") as f:
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.M)
        if match:
            base_version = match.group(1)
    local_version = os.environ.get("EIKO_LOCAL_VERSION", "")
    return f"{base_version}{local_version}"

# JAX Include bundle
def get_jax_includes() -> list:
    import pybind11
    import sysconfig
    return EXTRA_INCLUDE_PATHS + [pybind11.get_include(), sysconfig.get_path("include")]

def compile_raw_nvcc_shared_lib(src_file: str, output_path: str, include_dirs: list, is_jit: bool = False):
    """
    Executes the raw nvcc shared-library compilation step.
    If is_jit=True, compiles atomically to prevent race conditions during runtime imports.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nvcc_bin = get_nvcc_executable()
    
    # Windows MSVC does not use -fPIC
    cxx_pic = CXX_ARGS + ["-fPIC"] if "-fPIC" not in CXX_ARGS and os.name != "nt" else CXX_ARGS
    inc_flags = [f"-I{p}" for p in include_dirs if os.path.exists(p)]

    # Use a temp file for JIT to avoid locking/race conditions
    target_out = output_path
    if is_jit:
        target_out = output_path + f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"

    cmd = [nvcc_bin, "-shared", "-std=c++17", src_file, "-o", target_out]
    cmd += NVCC_ARGS
    cmd += [f"-Xcompiler={arg}" for arg in cxx_pic]
    cmd += inc_flags

    # Capturing stderr to PIPE ensures the diagnostic helper below can parse compiler crashes
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if is_jit:
        try:
            os.replace(target_out, output_path)
        except PermissionError:
            # Another process locked the file. Verify the other process actually finished building it.
            if not os.path.exists(output_path):
                raise RuntimeError(f"Race condition failure: {output_path} is missing and temp file could not be renamed.")
            
            if os.path.exists(target_out):
                try:
                    os.remove(target_out)
                except OSError:
                    pass # Windows file handles can be stubborn; safe to ignore if output_path exists

def diagnose_build_failure(error: Exception, framework: str, framework_version: str):
    """Parses C++/CUDA compilation errors and prints a structured, user-friendly diagnosis."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    os_name = platform.system()

    # Safely extract the raw error, combining stdout and stderr
    if isinstance(error, subprocess.CalledProcessError):
        out = error.stdout.decode('utf-8', errors='replace') if error.stdout else ""
        err = error.stderr.decode('utf-8', errors='replace') if error.stderr else ""
        raw_error = f"{out}\n{err}".strip()
    else:
        raw_error = str(error)

    error_msg = raw_error.lower()

    print("\n" + "="*75)
    print(f"[Eiko] FATAL ERROR: Local C++/CUDA compilation for {framework} failed.")
    print("="*75)
    print("1. We could not find a compatible precompiled wheel for your exact system.")
    print(f"2. We attempted to compile the {framework} extension from source, but it failed.\n")
    
    # Always print the raw error first
    print("--- RAW COMPILER OUTPUT ---")
    print(raw_error if raw_error else "<No output captured>")
    print("---------------------------\n")

    # --- DIAGNOSIS ROUTINES ---
    print("--- AUTOMATED DIAGNOSIS ---")
    if sys.platform == "win32" and ("cl.exe" in error_msg or "['where', 'cl']" in error_msg or "compiler" in error_msg):
        print("DIAGNOSIS: Microsoft Visual Studio C++ compiler ('cl.exe') was not found.")
        print("FIX: 1) Install the 'Desktop development with C++' workload via the Visual Studio Installer.")
        print("     2) Ensure you run Python inside the 'x64 Native Tools Command Prompt for VS'.")
    elif sys.platform != "win32" and any(x in error_msg for x in ["['which', 'c++']", "['which', 'g++']", "gcc", "g++", "c++"]):
        print("DIAGNOSIS: A C++ host compiler (like GCC or G++) was not found.")
        print("FIX: Install build tools on your system (e.g., run 'sudo apt install build-essential').")
    elif "nvcc" in error_msg or isinstance(error, FileNotFoundError):
        print("DIAGNOSIS: The NVIDIA CUDA compiler ('nvcc') was not found on your system path.")
        print("FIX: Ensure the NVIDIA CUDA Toolkit is installed and 'nvcc' is in your system PATH.")
        print("🔗 Download CUDA here: https://developer.nvidia.com/cuda-downloads")
    elif any(x in error_msg for x in ["sm_", "compute_", "compatibility", "undefined symbol"]):
        print("DIAGNOSIS: Hardware/Software architecture compatibility mismatch.")
        print("The compilation failed because your framework or CUDA driver doesn't align with your GPU capability.")
    else:
        print("DIAGNOSIS: Unknown compilation error. See the raw output above.")
        
    if framework.lower() == "pytorch":
        print("\n" + "-"*75)
        print("FASTEST FIX: UPDATE PYTORCH")
        print("Eiko provides precompiled wheels for the newest PyTorch releases.")
        print("Updating PyTorch to the latest version will likely bypass this compilation step entirely.")
        print(f"{link_emoji} https://pytorch.org/get-started/")

    # Grab the last 15 lines of the error for the issue template to keep it concise
    error_snippet = "\n".join(raw_error.splitlines()[-15:]) if raw_error else "N/A"

    print("\n" + "-"*75)
    print("STILL STUCK? REQUEST A PRECOMPILED WHEEL:")
    print("Open an issue here: 🔗 https://github.com/sebftw/Eiko/issues")
    print("Please copy and paste the following system information into your issue description:\n")
    print("```text")
    print(f"OS:      {os_name}")
    print(f"Python:  {py_ver}")
    print(f"Target:  {framework} {framework_version}")
    print("\n--- Error Snippet ---")
    print(error_snippet)
    print("```")
    print("="*75 + "\n")
    
    raise RuntimeError(f"Eiko {framework} initialization failed due to compilation errors.") from None
