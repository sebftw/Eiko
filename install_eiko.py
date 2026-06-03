#!/usr/bin/env python3
import sys
import platform
import subprocess
import shutil

# --- Configuration ---
EIKO_BASE_URL = "https://your-username.github.io/your-repo/whl"
EIKO_VERSION = "0.1.0" # You can update this or leave it out to grab the latest

def get_cuda_version():
    """Attempts to detect local CUDA version via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return "cpu"
    try:
        # nvidia-smi usually prints the max supported CUDA version in its header
        output = subprocess.check_output(["nvidia-smi"], text=True)
        if "CUDA Version:" in output:
            version_str = output.split("CUDA Version:")[1].split()[0]
            major, minor = version_str.split('.')[:2]
            return f"cu{major}{minor}"
    except Exception:
        pass
    return "cpu"

def get_pytorch_info():
    """Checks if PyTorch is installed and what version/backend it is using."""
    try:
        import torch
        pt_version = torch.__version__.split('+')[0] # Strips +cu121 if present
        
        # Check if the installed torch actually sees the GPU
        if torch.cuda.is_available():
            # e.g., '12.1' -> 'cu121'
            cu_version = "cu" + torch.version.cuda.replace(".", "") 
            return pt_version, cu_version
        return pt_version, "cpu"
    except ImportError:
        return None, None

def main():
    print("=== Eiko Environment Auto-Detector ===")
    
    # 1. Check Python Version (Just for sanity, pip handles the actual wheel matching)
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Detected Python: {py_version} on {platform.system()}")

    # 2. Check for existing PyTorch
    pt_version, pt_cuda = get_pytorch_info()
    
    # 3. Determine the target state
    if pt_version:
        print(f"Detected PyTorch: {pt_version} (Backend: {pt_cuda})")
        target_pt = pt_version
        target_cu = pt_cuda
        install_torch = False
    else:
        print("PyTorch not found. Scanning hardware...")
        target_cu = get_cuda_version()
        target_pt = "2.12.0" # Default to your latest supported version
        install_torch = True
        print(f"Detected Hardware Target: {target_cu}")

    # 4. Construct the commands
    commands = []
    
    if install_torch:
        torch_cmd = f"{sys.executable} -m pip install torch=={target_pt} --index-url https://download.pytorch.org/whl/{target_cu}"
        commands.append(torch_cmd)

    eiko_cmd = f"{sys.executable} -m pip install \"eiko=={EIKO_VERSION}+pt{target_pt}.{target_cu}\" --extra-index-url {EIKO_BASE_URL}/{target_cu}"
    commands.append(eiko_cmd)

    print("\n--- Recommended Installation Commands ---")
    for cmd in commands:
        print(f"> {cmd}")
    print("-----------------------------------------\n")

    # 5. Offer to execute
    response = input("Would you like to execute these commands now in the current environment? [y/N]: ")
    
    if response.lower() in ['y', 'yes']:
        for cmd in commands:
            print(f"Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
        print("\nInstallation Complete!")
    else:
        print("\nAborted. You can copy/paste the commands above when ready.")

if __name__ == "__main__":
    main()
