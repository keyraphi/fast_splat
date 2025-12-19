#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace cuda {
// Forward declarations
using TargetImageTypeCUDA = nb::ndarray<nb::array_api,float, nb::shape<-1, -1, 3>,
                                        nb::c_contig, nb::device::cuda>;
using PatchListTypeCUDA = nb::ndarray<const float, nb::shape<-1, -1, -1, 3>,
                                      nb::c_contig, nb::device::cuda>;
using PositionListTypeCUDA =
    nb::ndarray<const float, nb::shape<-1, 2>, nb::c_contig, nb::device::cuda>;

// CUDA memory management functions
void *cuda_allocate(size_t size);
void cuda_free(void *ptr);
bool cuda_memcpy(void *dest, void* src, size_t bytes);

// Device handling utils
auto get_cuda_device_from_ndarray(const nb::ndarray<nb::device::cuda> &arr)
    -> int;
// set cuda device to common device of inputs
auto get_common_cuda_device(const PatchListTypeCUDA &patch_list,
                            const PositionListTypeCUDA &position_list,
                            const TargetImageTypeCUDA &target_image) -> int;

class ScopedCudaDevice {
private:
  int original_device_;

public:
  ScopedCudaDevice(int new_device) {
    cudaGetDevice(&original_device_);
    cudaSetDevice(new_device);
  }

  ~ScopedCudaDevice() { cudaSetDevice(original_device_); }

  // Disallow copying
  ScopedCudaDevice(const ScopedCudaDevice &) = delete;
  ScopedCudaDevice &operator=(const ScopedCudaDevice &) = delete;
};

} // namespace cuda
