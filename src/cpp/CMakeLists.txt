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
nanobind_add_module(fast_splat_2d_backend
    fast_splat_2d.cpp
    cpu/fast_splat_2d_cpu.cpp
    cuda/fast_splat_2d_cuda.cu
    cuda/utils.cu
)

# CUDA compilation flags
target_compile_options(
    fast_splat_2d_backend
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# Link dependencies
target_link_libraries(fast_splat_2d_backend PRIVATE CUDA::cudart)
# C++ standard
target_compile_features(fast_splat_2d_backend PRIVATE cxx_std_17)
target_compile_options(
    fast_splat_2d_backend
    PRIVATE -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas="-v">
)
# for clangd
target_include_directories(fast_splat_2d_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# use OpenMP in c++ implementation
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fast_splat_2d_backend PUBLIC OpenMP::OpenMP_CXX)
endif()

# RPATH for Unix
if(UNIX)
    set_target_properties(fast_splat_2d_backend PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS fast_splat_2d_backend LIBRARY DESTINATION fast_splat_2d)
