import sys
import subprocess
import torch

print(sys.version)

# Check GCC version
def check_gcc_version():
    try:
        gcc_version = subprocess.check_output(['gcc', '--version'], stderr=subprocess.STDOUT)
        gcc_version = gcc_version.decode('utf-8').splitlines()[0]
        return gcc_version
    except Exception as e:
        return f"Error checking GCC version: {str(e)}"

# Print GCC version
gcc_version = check_gcc_version()
print(f"GCC Version: {gcc_version}")

def check_gpp_version():
    try:
        gpp_version = subprocess.check_output(['g++', '--version'], stderr=subprocess.STDOUT)
        gpp_version = gpp_version.decode('utf-8').splitlines()[0]
        return gpp_version
    except Exception as e:
        return f"Error checking G++ version: {str(e)}"

# Print G++ version
gpp_version = check_gpp_version()
print(f"G++ Version: {gpp_version}")


# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Get the version of CUDA being used by PyTorch
cuda_version = torch.version.cuda

# Output the results
print(f"CUDA Available: {cuda_available}")
print(f"CUDA Version: {cuda_version}")

def get_nvcc_version():
    try:
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # The version info will be in the stdout
            return result.stdout
        else:
            return f"Error: {result.stderr}"
    except FileNotFoundError:
        return "nvcc is not installed or not in the system path."

# Call the function to get the version
print(get_nvcc_version())
