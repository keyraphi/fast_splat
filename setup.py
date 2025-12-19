from setuptools import setup, find_packages

import subprocess
import sys
import os

def build_and_install_cpp_extension():
    """Build and install the C++ extension using CMake"""
    print("Building fast_splat_2d C++ extension...")
    
    # Create build directory
    build_dir = "build_setup"
    os.makedirs(build_dir, exist_ok=True)
    
    try:
        # Configure with CMake
        subprocess.check_call([
            'cmake', '..', 
            f'-DPython_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release'
        ], cwd=build_dir)
        
        # Build and install
        subprocess.check_call([
            'cmake', '--build', '.', '--target', 'fast_splat_2d'
        ], cwd=build_dir)
        
        subprocess.check_call([
            'cmake', '--install', '.'
        ], cwd=build_dir)
        
        print("Successfully built and installed fast_splat_2d")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to build C++ extension: {e}")
        print("Please ensure CMake and CUDA are installed")
        sys.exit(1)

# Build and install the C++ extension before setting up the Python package
build_and_install_cpp_extension()

setup(
    name="fast_splat_2D",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    author="Raphael Braun",
    author_email="keyraphi@gmail.com",
    description="A fast 2D splatting function, to splat a batch of splattlets into a given target image.",
)
