#include <cstddef>
#include <cstring>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sstream>
#include <stdexcept>

#include "cpu/fast_splat_2d_cpu.h"
#include "cuda/fast_splat_2d_cuda.h"
#include "cuda/utils.h"

namespace nb = nanobind;
using namespace nb::literals;

// Device-specific type aliases with exact constraints

using TargetImageTypeCPU =
    nb::ndarray<nb::array_api, float, nb::shape<-1, -1, 3>, nb::c_contig,
                nb::device::cpu>;
using TargetImageTypeCUDA =
    nb::ndarray<nb::array_api, float, nb::shape<-1, -1, 3>, nb::c_contig,
                nb::device::cuda>;
using PatchListTypeCPU = nb::ndarray<const float, nb::shape<-1, -1, -1, 3>,
                                     nb::c_contig, nb::device::cpu>;
using PatchListTypeCUDA = nb::ndarray<const float, nb::shape<-1, -1, -1, 3>,
                                      nb::c_contig, nb::device::cuda>;
using PositionListTypeCPU =
    nb::ndarray<const float, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;
using PositionListTypeCUDA =
    nb::ndarray<const float, nb::shape<-1, 2>, nb::c_contig, nb::device::cuda>;

size_t check_dimensions(const PatchListTypeCPU &patch_list,
                        const PositionListTypeCPU &position_list) {
  size_t patch_list_size = patch_list.shape(0);
  size_t position_list_size = position_list.shape(0);
  if (position_list_size != patch_list_size) {
    std::stringstream ss;
    ss << "fast_splat_2d: The number of positions (" << position_list_size
       << ") must match the number of patches (" << patch_list_size << ").";

    throw nb::value_error(ss.str().c_str());
  }

  return patch_list_size;
}

size_t check_dimensions(const PatchListTypeCUDA &patch_list,
                        const PositionListTypeCUDA &position_list) {
  size_t patch_list_size = patch_list.shape(0);
  size_t position_list_size = position_list.shape(0);
  if (patch_list_size != position_list_size) {
    std::stringstream ss;
    ss << "fast_splat_2d: The number of positions (" << position_list_size
       << ") must match the number of patches (" << patch_list_size << ").";
    throw nb::value_error(ss.str().c_str());
  }

  return patch_list_size;
}

// CPU function
auto fast_splat_2d_cpu(const PatchListTypeCPU &patch_list,
                       const PositionListTypeCPU &position_list,
                       TargetImageTypeCPU &target_image) -> TargetImageTypeCPU {

  size_t n_patches = check_dimensions(patch_list, position_list);

  // create a copy of the target image to splat into
  float *result_data;
  result_data = new float[target_image.shape(0) * target_image.shape(1) *
                          target_image.shape(2)];
  std::memcpy(result_data, target_image.data(),
              sizeof(float) * target_image.shape(0) * target_image.shape(1) *
                  target_image.shape(2));
  fast_splat_2d_cpu_impl(patch_list.data(), position_list.data(), n_patches,
                         patch_list.shape(2), patch_list.shape(1), result_data,
                         target_image.shape(1), target_image.shape(0));
  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return TargetImageTypeCPU(
      result_data,
      {target_image.shape(0), target_image.shape(1), target_image.shape(2)},
      owner);
}

auto fast_splat_2d_cuda(const PatchListTypeCUDA &patch_list,
                        const PositionListTypeCUDA &position_list,
                        TargetImageTypeCUDA &target_image)
    -> TargetImageTypeCUDA {
  size_t n_patches = check_dimensions(patch_list, position_list);

  // First ensure that correct gpu is used for all allocations and computations
  int target_device =
      cuda::get_common_cuda_device(patch_list, position_list, target_image);
  cuda::ScopedCudaDevice device(target_device);

  // copy target image
  float *result_data;
  result_data = static_cast<float *>(
      cuda::cuda_allocate(sizeof(float) * target_image.shape(0) *
                          target_image.shape(1) * target_image.shape(2)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  if (!cuda::cuda_memcpy(result_data, target_image.data(),
                         sizeof(float) * target_image.shape(0) *
                             target_image.shape(1) * target_image.shape(2))) {
    throw std::runtime_error("Failed to copy Target image for output");
  }

  fast_splat_2d_cuda_impl(patch_list.data(), position_list.data(), n_patches,
                          patch_list.shape(2), patch_list.shape(1), result_data,
                          target_image.shape(1), target_image.shape(0));

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return TargetImageTypeCUDA(
      result_data,
      {target_image.shape(0), target_image.shape(1), target_image.shape(2)},
      owner);
}

NB_MODULE(fast_splat_2d_backend, module) {
  module.doc() = "This module contains a fast splatting function for splatting "
                 "to a target image. "
                 "It either runs on CPU or on GPU, depending if the input "
                 "tensors are on GPU or system memory.";

  // Two separate overloads - nanobind will automatically dispatch based on
  // device type
  module.def(
      "splat", &fast_splat_2d_cpu,
      "CPU implementation of the fast splatting function.\n\n"
      "This is used when all arguments are on cpu.\n"
      "Args:\n"
      "    patch_list: The image patches that will be splatted to the target "
      "image. [N, patch_heigt, patch_width, 3]\n"
      "    position_list: List of pixel coordinates (x, y) to which the "
      "patches in the patch list are supposed to splatted to. The patches "
      "center points are placed at those positions. Uses bilinear "
      "interpolation for non integer positions. [N, 2]\n"

      "    target_image: The image which to splat to. The patches will be "
      "added to this image at the given positions.\n\n"
      "Returns:\n"
      "    Copy or reference to the target image (depending on is_inplace) "
      "with the patches splatted to the given positions.",
      "patch_list"_a, "position_list"_a, "target_image"_a);

  module.def(
      "splat", &fast_splat_2d_cuda,
      "CUDA implementation of the fast splatting function.\n\n"
      "This is used when all arguments are on gpu.\n"
      "Args:\n"
      "    patch_list: The image patches that will be splatted to the target "
      "image. [N, patch_heigt, patch_width, 3]\n"
      "    position_list: List of pixel coordinates (x, y) to which the "
      "patches in the patch list are supposed to splatted to. The patches "
      "center points are placed at those positions. Uses bilinear "
      "interpolation for non integer positions. [N, 2]\n"

      "    target_image: The image which to splat to. The patches will be "
      "added to this image at the given positions.\n\n"
      "Returns:\n"
      "    Copy or reference to the target image (depending on is_inplace) "
      "with the patches splatted to the given positions.",
      "patch_list"_a, "position_list"_a, "target_image"_a);
}
