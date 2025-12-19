#!/bin/bash

# Install system dependencies
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    echo "Detected Ubuntu/Debian system"
    sudo apt-get update
    sudo apt-get install -y cmake ninja-build build-essential python3-dev
    if [ "$1" = "cuda" ]; then
        sudo apt-get install -y nvidia-cuda-toolkit
    fi
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    echo "Detected CentOS/RHEL system"
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y cmake ninja-build python3-devel
    if [ "$1" = "cuda" ]; then
        sudo yum install -y cuda-toolkit
    fi
elif command -v pacman &> /dev/null; then
    # Arch Linux/Manjaro
    echo "Detected Arch Linux/Manjaro system"
    sudo pacman -Sy --noconfirm cmake ninja base-devel python
    if [ "$1" = "cuda" ]; then
        sudo pacman -Sy --noconfirm cuda
    fi
else
    echo "Unsupported package manager. Please install dependencies manually:"
    echo "  - cmake"
    echo "  - ninja"
    echo "  - build-essential (Ubuntu) or base-devel (Arch) or Development Tools (CentOS)"
    echo "  - python3-dev (Ubuntu) or python3-devel (CentOS) or python (Arch)"
    if [ "$1" = "cuda" ]; then
        echo "  - nvidia-cuda-toolkit (Ubuntu) or cuda (Arch) or cuda-toolkit (CentOS)"
    fi
    exit 1
fi

# Check if CUDA installation was requested and verify
if [ "$1" = "cuda" ]; then
    echo "Verifying CUDA installation..."
    if command -v nvcc &> /dev/null; then
        echo "CUDA compiler (nvcc) found: $(nvcc --version | head -n 4)"
    else
        echo "Warning: nvcc not found in PATH. CUDA may not be properly installed."
        echo "You may need to add CUDA to your PATH:"
        echo "  export PATH=/usr/local/cuda/bin:\$PATH"
        echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    fi
fi

# Verify Python headers are available
echo "Verifying Python development headers..."
python3 -c "
import sysconfig
import os
include_dir = sysconfig.get_path('include')
if not os.path.exists(include_dir):
    print(f'ERROR: Python include directory not found: {include_dir}')
    print('Please install Python development headers for your system:')
    print('  Ubuntu/Debian: python3-dev')
    print('  CentOS/RHEL: python3-devel') 
    print('  Arch/Manjaro: python')
    exit(1)
else:
    print(f'Python include directory found: {include_dir}')
    
# Check for Python.h
python_h = os.path.join(include_dir, 'Python.h')
if not os.path.exists(python_h):
    print(f'ERROR: Python.h not found at: {python_h}')
    exit(1)
else:
    print('Python.h found ✓')
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch numpy scikit-build-core

# Additional dependencies that might be needed
pip install nanobind

echo ""
echo "Dependencies installed successfully!"
echo ""
echo "Development workflow:"
echo "1. Install in development mode: pip install -e ."
echo "2. Run tests: python -m pytest tests/"
echo "3. Or use the test runner: python tests/run_tests.py"
echo ""
echo "For build debugging only:"
echo "  python tests/test_build_only.py --build-test"
