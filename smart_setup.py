import subprocess
import sys
import platform
import os

def run_command(command):
    print(f"Executing: {command}")
    try:
        subprocess.check_call(f"{sys.executable} -m pip install {command}", shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        return False

def get_system_info():
    os_type = platform.system()
    is_m_chip = False
    has_nvidia = False
    
    if os_type == "Windows":
        try:
            cmd = 'powershell -Command "Get-CimInstance Win32_VideoController | Select-Object Name"'
            output = subprocess.check_output(cmd, shell=True).decode()
            has_nvidia = "NVIDIA" in output.upper() or os.path.exists(r"C:\Program Files\NVIDIA Corporation")
        except: pass
    elif os_type == "Linux":
        try:
            output = subprocess.check_output("lspci | grep -i nvidia", shell=True).decode()
            has_nvidia = "NVIDIA" in output.upper()
        except: pass
    elif os_type == "Darwin": # Mac
        try:
            output = subprocess.check_output("sysctl -n hw.optional.arm64", shell=True).decode()
            is_m_chip = output.strip() == "1"
        except: pass

    return os_type, has_nvidia, is_m_chip

def smart_install():
    print("\n--- XAI Universal Smart Setup ---")
    os_type, has_nvidia, is_m_chip = get_system_info()
    print(f"System: {os_type} | NVIDIA: {has_nvidia} | Apple Silicon: {is_m_chip}")

    if has_nvidia:
        print("Optimizing for NVIDIA GPU (CUDA)...")
        # cu118 is the most compatible for older/mid-range cards like 1050M
        torch_cmd = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall"
    elif is_m_chip:
        print("Optimizing for Apple Silicon (MPS)...")
        torch_cmd = "torch torchvision torchaudio --force-reinstall"
    else:
        print("Using standard CPU/Universal version.")
        torch_cmd = "torch torchvision torchaudio --force-reinstall"

    if run_command(torch_cmd):
        print("\nInstalling remaining dependencies...")
        run_command("-r requirements.txt")
        print("\n--- SETUP COMPLETE! ---")
    else:
        print("Setup failed.")

if __name__ == "__main__":
    smart_install()
