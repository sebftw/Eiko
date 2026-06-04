import os
import sys
import json
import re
import urllib.request
import urllib.error
import platform
import zipfile
from pathlib import Path

REGISTRY_URL = "https://sebftw.github.io/Eiko/registry.json"

def fetch_precompiled_wheel(package_version, torch_version, cuda_version, target_dir, target_impl="eiko_torch_impl"):
    """
    Attempts to download and extract the precompiled wheel.
    If cuda_version is None, it defaults to the highest available CUDA version for the environment.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"[Eiko] Looking for precompiled binaries for {target_impl}...")
    
    # 1. Fetch Registry
    try:
        req = urllib.request.Request(REGISTRY_URL, headers={'User-Agent': 'eiko-bootstrap'})
        with urllib.request.urlopen(req, timeout=3.0) as response:
            registry = json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"[Eiko] Network error reaching registry ({e.reason}). Falling back to JIT.")
        return False
    except Exception as e:
        print(f"[Eiko] Failed to parse registry. Falling back to JIT.")
        return False

    # 2. Match Environment (OS and Python)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    os_name = "windows" if platform.system().lower() == "windows" else "linux"
    
    builds = registry.get("versions", {}).get(package_version, [])
    
    # Filter by base system compatibility
    valid_builds = []
    for b in builds:
        if b["os"] == os_name and b["python"] == py_ver:
            # If a PyTorch version constraint is given, respect it
            if torch_version is not None:
                t_ver = ".".join(torch_version.split("+")[0].split(".")[:2])
                if not b["torch"].startswith(t_ver):
                    continue
            valid_builds.append(b)

    if not valid_builds:
        print(f"[Eiko] No precompiled wheels found matching OS: {os_name}, Python: {py_ver}")
        return False

    # Helper to rank wheels by CUDA version string (e.g., "cu124" -> 124, "cpu" -> 0)
    def get_cuda_score(build_item):
        digits = re.findall(r'\d+', build_item.get("cuda", ""))
        return int(digits[0]) if digits else 0

    # 3. Select the Best Variant
    matched_build = None
    if cuda_version is not None:
        # Try exact match first (useful for PyTorch backend tracking)
        matched_build = next((b for b in valid_builds if b["cuda"] == cuda_version), None)
        
    if not matched_build:
        # Fallback / JAX track: Sort and pick the newest compiled CUDA platform version
        matched_build = max(valid_builds, key=get_cuda_score)

    print(f"[Eiko] Selected wheel: {matched_build['filename']} (CUDA target: {matched_build['cuda']})")

    # 4. Download & Extract
    wheel_url = matched_build["url"]
    wheel_path = os.path.join(target_dir, matched_build["filename"])
    
    try:
        urllib.request.urlretrieve(wheel_url, wheel_path)
        
        with zipfile.ZipFile(wheel_path, 'r') as z:
            binary_files = [f for f in z.namelist() if target_impl in f and f.endswith(('.so', '.pyd', '.dll'))]
            if not binary_files:
                raise ValueError(f"Target binary artifact '{target_impl}' not found inside the wheel footprint.")
                
            for bf in binary_files:
                filename = os.path.basename(bf)
                with z.open(bf) as source, open(os.path.join(target_dir, filename), "wb") as target:
                    target.write(source.read())
                    
        os.remove(wheel_path)
        print("[Eiko] Successfully downloaded and cached precompiled binaries.")
        return True
        
    except Exception as e:
        print(f"[Eiko] Download or extraction failed: {e}. Falling back to JIT.")
        if os.path.exists(wheel_path):
            os.remove(wheel_path)
        return False
