# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Include nanobind
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG master
)
FetchContent_MakeAvailable(nanobind)

# Enable CUDA
enable_language(CUDA)
set(CUDA_ARCHITECTURES 75 80 86 89 90)
# set(CUDA_ARCHITECTURES 75) # TEMPORAYR for faster compilation
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# Single shared library for the Python extension
nanobind_add_module(fast_splat_2d
    fast_splat_2d.cpp
    cpu/fast_splat_2d_cpu.cpp
    cuda/fast_splat_2d_cuda.cu
    cuda/utils.cu
)

# CUDA compilation flags
target_compile_options(
    fast_splat_2d
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# Link dependencies
target_link_libraries(fast_splat_2d PRIVATE CUDA::cudart)
# C++ standard
target_compile_features(fast_splat_2d PRIVATE cxx_std_20)
target_compile_options(
    fast_splat_2d
    PRIVATE -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas="-v">
)
# for clangd
target_include_directories(fast_splat_2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# use OpenMP in c++ implementation
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fast_splat_2d PUBLIC OpenMP::OpenMP_CXX)
endif()

# RPATH for Unix
if(UNIX)
    set_target_properties(fast_splat_2d PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

# Install the C++ extension to the current Python's site-packages
# Prefer Python from the current environment
if(DEFINED ENV{VIRTUAL_ENV})
    message(STATUS "Using virtual environment: $ENV{VIRTUAL_ENV}")
    set(Python_ROOT_DIR $ENV{VIRTUAL_ENV})
endif()

# Find Python - this should now find the venv Python
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Python include: ${Python_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python_LIBRARIES}")

# Get Python package installation directory
execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")

install(TARGETS fast_splat_2d LIBRARY DESTINATION ${Python_SITEARCH})
