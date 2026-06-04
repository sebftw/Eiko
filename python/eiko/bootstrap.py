import os
import sys
import json
import urllib.request
import urllib.error
import platform
import zipfile
from pathlib import Path

REGISTRY_URL = "https://sebftw.github.io/Eiko/registry.json"

def fetch_precompiled_wheel(package_version, torch_version, cuda_version, target_dir, target_impl="eiko_torch_impl"):
    """
    Attempts to download and extract the precompiled wheel.
    Returns True if successful, False if it should fallback to JIT.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if we already downloaded it previously
    if any(f.endswith(('.so', '.pyd', '.dll')) for f in os.listdir(target_dir)):
        return True

    print(f"[Eiko] Looking for precompiled binaries (PyTorch {torch_version}, CUDA {cuda_version or 'cpu'})...")
    
    # 1. Fetch Registry
    try:
        req = urllib.request.Request(REGISTRY_URL, headers={'User-Agent': 'eiko-bootstrap'})
        with urllib.request.urlopen(req, timeout=3.0) as response:
            registry = json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"[Eiko] Network error reaching registry ({e.reason}). Falling back to JIT compilation.")
        return False
    except Exception as e:
        print(f"[Eiko] Failed to parse registry. Falling back to JIT compilation.")
        return False

    # 2. Match Environment
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    os_name = "windows" if platform.system().lower() == "windows" else "linux"
    
    # Extract major/minor PyTorch versions (e.g., "2.4.0+cu121" -> "2.4")
    t_ver = ".".join(torch_version.split("+")[0].split(".")[:2])
    
    # FIX: Normalize CUDA version string to match "cu121" or "cpu" exactly
    if cuda_version and cuda_version.lower() != "cpu":
        c_ver = f"cu{cuda_version.replace('.', '')}" # "12.1" -> "cu121"
    else:
        c_ver = "cpu"
    
    builds = registry.get("versions", {}).get(package_version, [])
    matched_build = None
    
    for b in builds:
        # If torch_version is None, we ignore the PyTorch check and just match OS, Python, and CUDA
        torch_match = True if torch_version is None else b["torch"].startswith(t_ver)
        
        if torch_match and b["cuda"] == c_ver and b["os"] == os_name and b["python"] == py_ver:
            matched_build = b
            break
          

    if not matched_build:
        print(f"[Eiko] No precompiled wheel found for your setup (PyTorch ~{t_ver}, CUDA {c_ver}, Python {py_ver}, OS {os_name}).")
        print(f"[Eiko] Falling back to local JIT compilation.")
        return False

    # 3. Download & Extract
    wheel_url = matched_build["url"]
    wheel_path = os.path.join(target_dir, matched_build["filename"])
    
    try:
        print(f"[Eiko] Downloading precompiled binary from remote registry...")
        urllib.request.urlretrieve(wheel_url, wheel_path)
        
        # Extract only the specific compiled binary from the wheel
        with zipfile.ZipFile(wheel_path, 'r') as z:
            binary_files = [f for f in z.namelist() if target_impl in f and f.endswith(('.so', '.pyd', '.dll'))]
            if not binary_files:
                raise ValueError("Binary compiled object missing from the downloaded wheel file structure.")
                
            for bf in binary_files:
                filename = os.path.basename(bf)
                source = z.open(bf)
                target = open(os.path.join(target_dir, filename), "wb")
                with source, target:
                    target.write(source.read())
                    
        # Clean up the wheel archive footprint
        os.remove(wheel_path)
        print("[Eiko] Successfully downloaded and cached precompiled binaries.")
        return True
        
    except Exception as e:
        print(f"[Eiko] Download or extraction failed: {e}. Falling back to JIT compilation.")
        if os.path.exists(wheel_path):
            os.remove(wheel_path)
        return False
