import os
import sys
import json
import re
import hashlib
import urllib.request
import urllib.error
import platform
import zipfile
import tempfile
import shutil
import uuid

REGISTRY_URL = "https://sebftw.github.io/Eiko/registry.json"

def fetch_precompiled_wheel(package_version, torch_version, cuda_version, target_dir, target_impl="eiko_torch_impl"):
    os.makedirs(target_dir, exist_ok=True)
    
    # Early Exit
    existing_binaries = [f for f in os.listdir(target_dir) if target_impl in f and f.endswith(('.so', '.pyd', '.dll'))]
    if existing_binaries:
        # print(f"[Eiko] Binary already exists in {target_dir}. Skipping download.")
        return True

    # print(f"[Eiko] Looking for precompiled binaries for {target_impl}...")
    
    # 1. Fetch Registry
    try:
        req = urllib.request.Request(REGISTRY_URL, headers={'User-Agent': 'eiko-bootstrap'})
        with urllib.request.urlopen(req, timeout=5.0) as response:
            registry = json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"[Eiko] Network error reaching registry ({e.reason}). Falling back to JIT.")
        return False
    except Exception:
        print(f"[Eiko] Failed to parse registry. Falling back to JIT.")
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
        print(f"[Eiko] No precompiled wheels found matching OS: {os_name}, Python: {py_ver}")
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

    # print(f"[Eiko] Selected wheel: {matched_build['filename']} (CUDA target: {matched_build['cuda']})")

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
    
    try:
        if not os.path.exists(wheel_path):
            # print(f"[Eiko] Downloading wheel to temporary cache...")
            req = urllib.request.Request(wheel_url, headers={'User-Agent': 'eiko-bootstrap'})
            
            sha256_hash = hashlib.sha256()
            downloaded_size = 0
            
            with urllib.request.urlopen(req, timeout=15.0) as response, open(tmp_wheel_path, 'wb') as out_file:
                # Read and write the file in 8KB chunks
                while chunk := response.read(8192):
                    downloaded_size += len(chunk)
                    
                    # Defend against infinite streams / DoS
                    if downloaded_size > max_allowed_size:
                        raise ValueError(f"Download exceeded the maximum allowed size of {max_allowed_size} bytes.")
                    
                    out_file.write(chunk)
                    sha256_hash.update(chunk)
            
            # Verify the hash before moving it to the active path
            if expected_hash:
                calculated_hash = sha256_hash.hexdigest()
                if calculated_hash != expected_hash:
                    raise ValueError(
                        f"Hash mismatch!\nExpected: {expected_hash}\nGot:      {calculated_hash}\n"
                        "The file may be corrupted or compromised."
                    )
            
            # Atomically rename the verified tmp file to the final wheel path.
            os.replace(tmp_wheel_path, wheel_path)
        # else:
        #    print(f"[Eiko] Using cached wheel from {wheel_path}")
        
        # Atomic Extraction Phase
        with zipfile.ZipFile(wheel_path, 'r') as z:
            binary_files = [f for f in z.namelist() if target_impl in f and f.endswith(('.so', '.pyd', '.dll'))]
            if not binary_files:
                raise ValueError(f"Target binary artifact '{target_impl}' not found inside the wheel footprint.")
                
            for bf in binary_files:
                filename = os.path.basename(bf)
                final_target_path = os.path.join(target_dir, filename)
                tmp_target_path = final_target_path + tmp_suffix
                
                # Extract to a temporary unique file first
                with z.open(bf) as source, open(tmp_target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                
                # Atomically swap the temporary binary into its final location
                os.replace(tmp_target_path, final_target_path)
                    
        # print("[Eiko] Successfully extracted cached precompiled binaries.")
        return True
        
    except Exception as e:
        print(f"[Eiko] Download or extraction failed: {e}. Falling back to JIT.")
        # Cleanup unique temp files if something went wrong
        if os.path.exists(tmp_wheel_path):
            os.remove(tmp_wheel_path)
        # Avoid deleting the shared wheel_path on error, as another process might be reading it!
        return False
