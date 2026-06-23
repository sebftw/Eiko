import os
import sys
import json
import uuid
import platform
import urllib.request
import urllib.error
import hashlib
import zipfile
import shutil
import tempfile
import re
import sysconfig

def fetch_precompiled_wheel(
    package_version, 
    backend_name=None,      # 'torch' or 'jax'
    backend_version=None,   # e.g. '2.3.0' or '0.4.28'
    cuda_version=None,      # e.g. 'cu121'
    target_dir=None, 
    target_impl="eiko_torch_impl"
):
    if target_dir is None:
        raise ValueError("[Eiko] target_dir must be explicitly provided.")
    if package_version is None:
        ValueError("[Eiko] package_version must be explicitly provided.")
    # Auto-infer backend from target_impl if not explicitly passed
    if not backend_name:
        backend_name = "jax" if "jax" in target_impl.lower() else "torch"
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Clean the package version key to match semver format stored in generator
    clean_pkg_ver = package_version.lstrip("v").split("+")[0]

    # Get the exact C-extension suffix for the current environment
    # e.g., '.cpython-310-x86_64-linux-gnu.so' or '.cp310-win_amd64.pyd'
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    
    # Fallback to generic extensions if sysconfig returns None (rare edge case)
    valid_suffixes = (ext_suffix,) if ext_suffix else ('.so', '.pyd', '.dll')

    # Early Exit: Check if valid binaries already exist
    existing_binaries = [
        f for f in os.listdir(target_dir) 
        if target_impl in f and f.endswith(valid_suffixes)
    ]

    if existing_binaries:
        return True
    
    # 1. Load Local Registry
    try:
        registry_path = os.path.join(os.path.dirname(__file__), "registry.json")
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    except Exception as e:
        print(f"[Eiko] Failed to read local registry ({e}). Falling back to JIT.")
        return False
    
    # 2. Match Environment (OS & Python)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    os_name = platform.system().lower()
    
    builds = registry.get("versions", {}).get(clean_pkg_ver, [])
    valid_builds = []
    
    # Normalize target backend version (match up to minor version: '2.3')
    target_b_minor = ""
    if backend_version:
        target_b_minor = ".".join(backend_version.split("+")[0].split(".")[:2])
    
    for b in builds:
        if b.get("os") != os_name or b.get("python") != py_ver:
            continue
            
        # Framework Version Matching
        wheel_b_ver = b.get(backend_name, "")
        
        # Hard Reject: Wheel does not support this backend at all
        if not wheel_b_ver:
            continue
        
        # Version Check: Match target version unless wheel is universally compatible ("any")
        if target_b_minor and wheel_b_ver != "any":
            if not wheel_b_ver.startswith(target_b_minor):
                continue
                
        valid_builds.append(b)

    if not valid_builds:
        cuda_str = (", CUDA: " + cuda_version) if cuda_version else ""
        print(f"[Eiko] No precompiled wheels matching OS: {os_name}, Python: {py_ver}, {backend_name}: {target_b_minor}{cuda_str} Falling back to JIT.")
        return False

    # 3. Select Variant (CUDA Scoring)
    def parse_cuda_num(cu_str):
        digits = re.findall(r'\d+', cu_str)
        return int(digits[0]) if digits else 0

    target_cu_num = parse_cuda_num(cuda_version) if cuda_version else float('inf')
    
    matched_build = None
    if cuda_version:
        matched_build = next((b for b in valid_builds if b.get("cuda") == cuda_version), None)
        
    if not matched_build:
        # Filter out wheels that require a higher CUDA version than the host supports
        compatible_cuda_builds = [
            b for b in valid_builds 
            if parse_cuda_num(b.get("cuda", "")) <= target_cu_num
        ]
        # Pick the highest available compatible CUDA version
        fallback_pool = compatible_cuda_builds if compatible_cuda_builds else valid_builds
        matched_build = max(fallback_pool, key=lambda b: parse_cuda_num(b.get("cuda", "")))

    # 4. Multiprocessing-Safe Download & Extract
    print(f"[Eiko] Downloading precompiled {backend_name.upper()} kernels...")
    wheel_url = matched_build["url"]
    eiko_tmp_dir = os.path.join(tempfile.gettempdir(), "eiko_cache")
    os.makedirs(eiko_tmp_dir, exist_ok=True)
    
    wheel_path = os.path.join(eiko_tmp_dir, matched_build["filename"])
    tmp_suffix = f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    tmp_wheel_path = wheel_path + tmp_suffix

    expected_hash = matched_build.get("sha256")
    max_allowed_size = 50 * 1024 * 1024  # 50 MB
    chunk_size = 256 * 1024 
    
    try:
        # Only download if not already cached in temp
        if not os.path.exists(wheel_path):
            req = urllib.request.Request(wheel_url, headers={'User-Agent': 'eiko-bootstrap'})
            with urllib.request.urlopen(req, timeout=10.0) as response:
                content_length = response.getheader('Content-Length')
                if content_length and int(content_length) > max_allowed_size:
                    raise ValueError(f"Reported size exceeds maximum allowed limit.")

                sha256_hash = hashlib.sha256()
                downloaded_size = 0
                
                with open(tmp_wheel_path, 'wb') as out_file:
                    while chunk := response.read(chunk_size):
                        downloaded_size += len(chunk)
                        if downloaded_size > max_allowed_size:
                            raise ValueError("Download exceeded maximum allowed size.")
                        out_file.write(chunk)
                        sha256_hash.update(chunk)
            
            if expected_hash and sha256_hash.hexdigest() != expected_hash:
                raise ValueError(f"Hash mismatch for {matched_build['filename']}.")
            
            os.replace(tmp_wheel_path, wheel_path)
        
        # Atomic Extraction Phase
        tmp_target_path = None
        try:
            with zipfile.ZipFile(wheel_path, 'r') as z:
                binary_files = [
                    f for f in z.namelist() 
                    if target_impl in f and f.endswith(('.so', '.pyd', '.dll'))
                ]
                if not binary_files:
                    raise ValueError(f"Target artifact '{target_impl}' missing from wheel.")
                    
                for bf in binary_files:
                    filename = os.path.basename(bf)
                    final_target_path = os.path.join(target_dir, filename)
                    tmp_target_path = final_target_path + tmp_suffix
                    
                    with z.open(bf) as source, open(tmp_target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    
                    os.replace(tmp_target_path, final_target_path)
                        
            return True
        except zipfile.BadZipFile:
            print(f"[Eiko] Precompiled wheel (zip) was corrupted. Deleting and falling back to JIT.")
            os.remove(wheel_path)
            return False
    
    except Exception as e:
        print(f"[Eiko] Binary bootstrap failed: {e}. Falling back to JIT.")
        if tmp_target_path and os.path.exists(tmp_target_path):
            os.remove(tmp_target_path)
        return False
