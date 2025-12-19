#include "utils.h"
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <vector>
#include <vector_functions.h>

namespace nb = nanobind;

namespace cuda {
// CUDA memory allocation
void *cuda_allocate(size_t size) {
  void *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

// CUDA memory deallocation
void cuda_free(void *ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

// CUDA memcpy
bool cuda_memcpy(void* dest, void* src, size_t bytes) {
  cudaError_t err = cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    return false;
  }
  return true;
}

// Helper to get CUDA device from nanobind ndarray
auto get_cuda_device_from_ndarray(const void *data_ptr) -> int {
  cudaPointerAttributes attributes;
  cudaError_t result = cudaPointerGetAttributes(&attributes, data_ptr);

  if (result != cudaSuccess) {
    throw ::std::runtime_error("Failed to get CUDA pointer attributes");
  }

  // attributes.device contains the device ID where the memory is allocated
  return attributes.device;
}

// Overload for multiple arrays - picks the first valid one
auto get_common_cuda_device(const PatchListTypeCUDA &patch_list,
                            const PositionListTypeCUDA &position_list,
                            const TargetImageTypeCUDA &target_image) -> int {
  // Check all arrays and return the first valid device
  ::std::vector<int> devices;

  if (patch_list.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(patch_list.data()));
  }
  if (position_list.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(position_list.data()));
  }
  if (target_image.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(target_image.data()));
  }
  // If no arrays have data (shouldn't happen in practice), default to device
  // 0
  if (devices.empty()) {
    return 0;
  }
  int common_device = 0;
  // Verify all devices are the same
  common_device = devices[0];
  for (size_t i = 1; i < devices.size(); ++i) {
    if (devices[i] != common_device) {
      throw ::std::runtime_error("Input tensors are on different GPUs. All "
                                 "inputs must be on the same GPU.");
    }
  }
  return common_device;
}

} // namespace cuda
