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

def fetch_precompiled_wheel(package_version, torch_version, cuda_version, target_dir, target_impl="eiko_torch_impl"):
    os.makedirs(target_dir, exist_ok=True)
    
    # Early Exit
    existing_binaries = [f for f in os.listdir(target_dir) if target_impl in f and f.endswith(('.so', '.pyd', '.dll'))]
    if existing_binaries:
        return True

    print(f"[Eiko] Downloading precompiled CUDA kernels for you... (This may take a minute)")
    
    # 1. Load Local Registry
    try:
        # Resolves the path to the registry.json sitting next to this script
        registry_path = os.path.join(os.path.dirname(__file__), "registry.json")
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except FileNotFoundError:
        print("[Eiko] Local registry.json not found. Falling back to JIT.")
        return False
    except Exception as e:
        print(f"[Eiko] Failed to parse local registry ({e}). Falling back to JIT.")
        return False

    # 2. Match Environment
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    os_name = "windows" if platform.system().lower() == "windows" else "linux"
    
    builds = registry.get("versions", {}).get(package_version, [])
    valid_builds = []
    for b in builds:
        if b["os"] == os_name and b["python"] == py_ver:
            if torch_version is not None:
                t_ver = ".".join(torch_version.split("+")[0].split(".")[:2])
                if not b["torch"].startswith(t_ver):
                    continue
            valid_builds.append(b)

    if not valid_builds:
        print(f"[Eiko] No precompiled wheels found matching OS: {os_name}, Python: {py_ver}. Falling back to JIT.")
        return False

    def get_cuda_score(build_item):
        digits = re.findall(r'\d+', build_item.get("cuda", ""))
        return int(digits[0]) if digits else 0

    # 3. Select Variant
    matched_build = None
    if cuda_version is not None:
        matched_build = next((b for b in valid_builds if b["cuda"] == cuda_version), None)
    if not matched_build:
        matched_build = max(valid_builds, key=get_cuda_score)

    # 4. Multiprocessing-Safe Download & Extract
    wheel_url = matched_build["url"]
    eiko_tmp_dir = os.path.join(tempfile.gettempdir(), "eiko_cache")
    os.makedirs(eiko_tmp_dir, exist_ok=True)
    
    wheel_path = os.path.join(eiko_tmp_dir, matched_build["filename"])
    
    # Generate a unique suffix for this specific process/thread
    tmp_suffix = f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    tmp_wheel_path = wheel_path + tmp_suffix

    expected_hash = matched_build.get("sha256")
    max_allowed_size = 50 * 1024 * 1024  # 50 MB safety limit
    chunk_size = 256 * 1024  # 256 KB chunks 
    
    try:
        if not os.path.exists(wheel_path):
            req = urllib.request.Request(wheel_url, headers={'User-Agent': 'eiko-bootstrap'})
            
            with urllib.request.urlopen(req, timeout=5.0) as response:
                content_length = response.getheader('Content-Length')
                if content_length and int(content_length) > max_allowed_size:
                    raise ValueError(f"Reported size ({content_length} bytes) exceeds maximum allowed size.")

                sha256_hash = hashlib.sha256()
                downloaded_size = 0
                
                with open(tmp_wheel_path, 'wb') as out_file:
                    while chunk := response.read(chunk_size):
                        downloaded_size += len(chunk)
                        
                        if downloaded_size > max_allowed_size:
                            raise ValueError(f"Download exceeded the maximum allowed size of {max_allowed_size} bytes.")
                        
                        out_file.write(chunk)
                        sha256_hash.update(chunk)
            
            if expected_hash:
                calculated_hash = sha256_hash.hexdigest()
                if calculated_hash != expected_hash:
                    raise ValueError(
                        f"Hash mismatch!\nExpected: {expected_hash}\nGot:      {calculated_hash}\n"
                        "The file may be corrupted or compromised."
                    )
            
            os.replace(tmp_wheel_path, wheel_path)
        
        # Atomic Extraction Phase
        with zipfile.ZipFile(wheel_path, 'r') as z:
            binary_files = [f for f in z.namelist() if target_impl in f and f.endswith(('.so', '.pyd', '.dll'))]
            if not binary_files:
                raise ValueError(f"Target binary artifact '{target_impl}' not found inside the wheel footprint. Falling back to JIT.")
                
            for bf in binary_files:
                filename = os.path.basename(bf)
                final_target_path = os.path.join(target_dir, filename)
                tmp_target_path = final_target_path + tmp_suffix
                
                with z.open(bf) as source, open(tmp_target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                
                os.replace(tmp_target_path, final_target_path)
                    
        return True
        
    except Exception as e:
        print(f"[Eiko] Download or extraction failed: {e}. Falling back to JIT.")
        if os.path.exists(tmp_wheel_path):
            os.remove(tmp_wheel_path)
        return False
