#include "fast_splat_2d_cuda.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cub/util_type.cuh>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <string>
#include <sys/types.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <tuple>

#define TILE_SIZE_X 32
#define TILE_SIZE_Y 32

void cuda_debug_print(const std::string &kernel_name) {
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("%s: CUDA Error: %s\n", kernel_name.c_str(),
           cudaGetErrorString(err));
  } else {
    printf("Kernel %s finished successfully!\n", kernel_name.c_str());
  }
}

/** Finds out if the given positions in the position list (x, y) have an effect
 * on a target patch. The target patch size is defined by TILE_SIZE_X and
 * TILE_SIZE_Y. The target position is implicit from the thread id. There is one
 * thread for every targe patch and every position list entry. The result is a
 * bit map of shape MxN, where M is the total number of target positions and N
 * is the number of positions in the position list (patch_count). Each input
 * position has a radius of influence (rectangular) with the given
 * patch_radius_x and patch_radius_y shape. If the influence area of the input
 * position and the target patch overlap in any way the corresponding bitmap
 * value should be set to 1, otherwise 0;
 */
__global__ void find_source_patches_for_target_tiles(
    const float *__restrict__ position_list, const uint32_t patch_count,
    const float patch_radius_x, const float patch_radius_y,
    const uint32_t target_width, const uint32_t target_count, uint8_t *bitmap) {
  uint32_t position_n = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t target_m = threadIdx.y + blockIdx.y * blockDim.y;
  if (position_n >= patch_count || target_m >= target_count) {
    return;
  }
  uint32_t tid = target_m * patch_count + position_n;

  const auto T_SIZE_X = static_cast<uint32_t>(TILE_SIZE_X);
  const auto T_SIZE_Y = static_cast<uint32_t>(TILE_SIZE_Y);

  uint32_t tiles_per_row = (target_width + T_SIZE_X - 1) / T_SIZE_X;
  uint32_t target_patch_y = target_m / tiles_per_row;
  uint32_t target_patch_x = target_m % tiles_per_row;
  auto target_x_left = static_cast<float>(target_patch_x * T_SIZE_X);
  auto target_y_top = static_cast<float>(target_patch_y * T_SIZE_Y);
  float target_x_right = target_x_left + static_cast<float>(T_SIZE_X);
  float target_y_bottom = target_y_top + static_cast<float>(T_SIZE_Y);

  float pos_x = position_list[position_n];
  float pos_y = position_list[position_n + patch_count];
  if (target_m == 63) {
    printf("patch %u, pos_x: %f, pos_y: %f\n", position_n, pos_x, pos_y);
    if (position_n == 0) {
      printf("tile %u, (%f, %f), (%f, %f)\n", target_m, target_x_left,
             target_y_top, target_x_right, target_y_bottom);
    }
  }

  uint8_t is_in_m = 0;

  // Calculate extended influence area for bilinear interpolation
  float source_left = pos_x - ceilf(patch_radius_x);
  float source_right = pos_x + ceilf(patch_radius_x);
  float source_top = pos_y - ceilf(patch_radius_y);
  float source_bottom = pos_y + ceilf(patch_radius_y);

  // Check for overlap (using half-open intervals: [left, right) x [top,
  // bottom))
  bool x_overlap =
      (source_right > target_x_left) && (source_left < target_x_right);
  bool y_overlap =
      (source_bottom > target_y_top) && (source_top < target_y_bottom);

  if (x_overlap && y_overlap) {
    is_in_m = 1;
  }

  bitmap[tid] = is_in_m;
}

auto compute_indices_from_bitmap(thrust::device_vector<uint8_t> &bitmap,
                                 const size_t rows, const size_t columns)
    -> std::tuple<thrust::device_vector<uint32_t>,
                  thrust::device_vector<uint32_t>,
                  thrust::device_vector<uint32_t>> {
  // prefix sum over each row
  auto make_key = [columns] __host__ __device__(uint32_t idx) {
    return idx / static_cast<uint32_t>(columns);
  };
  auto keys_begin = thrust::make_transform_iterator(
      thrust::counting_iterator<uint32_t>(0), make_key);

  thrust::device_vector<uint32_t> prefix_sum(rows * columns);
  thrust::exclusive_scan_by_key(keys_begin, keys_begin + (rows * columns),
                                bitmap.begin(), prefix_sum.begin());

  // number of entries per row
  thrust::device_vector<uint32_t> row_sums(rows);
  thrust::reduce_by_key(keys_begin, keys_begin + (rows * columns),
                        bitmap.begin(), thrust::discard_iterator<>(),
                        row_sums.begin());

  // start of each row
  thrust::device_vector<uint32_t> row_offsets(rows);
  thrust::exclusive_scan(row_sums.begin(), row_sums.end(), row_offsets.begin());

  //  write indices of patches together
  auto column_indices_begin = thrust::make_transform_iterator(
      thrust::counting_iterator<uint32_t>(0),
      [columns] __host__ __device__(uint32_t idx) {
        return idx % static_cast<uint32_t>(columns);
      });

  // Use copy_if to extract only indices where bitmap is 1
  // Count the number of 1s first to allocate the right size
  // equivalent to summing up sums of rows
  uint32_t total_ones = thrust::reduce(row_sums.begin(), row_sums.end());
  thrust::device_vector<uint32_t> result(total_ones);

  thrust::copy_if(column_indices_begin, // Source: column indices
                  column_indices_begin + (rows * columns), // End
                  bitmap.begin(), // Stencil: bitmap values (0 or 1)
                  result.begin(), // Destination
                  [] __host__ __device__(uint8_t val) { return val == 1; });

  return {result, row_sums, row_offsets};
}

__device__ inline void
bilinear_splat(const float src_red, const float src_green, const float src_blue,
               const float x_in_tile, const float y_in_tile, float *tile) {
  const int left = floorf(x_in_tile);
  const int right = left + 1;
  const int top = floorf(y_in_tile);
  const int bottom = top + 1;
  const uint32_t pixels_in_tile = TILE_SIZE_X * TILE_SIZE_Y;

  if (left >= 0 && left < TILE_SIZE_X) {
    const float weight_left = static_cast<float>(right) - x_in_tile;
    if (top >= 0 && top < TILE_SIZE_Y) {
      const float weight_top = static_cast<float>(bottom) - y_in_tile;
      const float weight = weight_left * weight_top;
      uint32_t tile_idx = left + top * TILE_SIZE_X;
      atomicAdd(tile + tile_idx, src_red * weight);
      atomicAdd(tile + tile_idx + pixels_in_tile, src_green * weight);
      atomicAdd(tile + tile_idx + 2 * pixels_in_tile, src_blue * weight);
    }
    if (bottom >= 0 && bottom < TILE_SIZE_Y) {
      const float weight_bottom = y_in_tile - static_cast<float>(top);
      const float weight = weight_left * weight_bottom;
      uint32_t tile_idx = left + bottom * TILE_SIZE_X;
      atomicAdd(tile + tile_idx, src_red * weight);
      atomicAdd(tile + tile_idx + pixels_in_tile, src_green * weight);
      atomicAdd(tile + tile_idx + 2 * pixels_in_tile, src_blue * weight);
    }
  }
  if (right >= 0 && right < TILE_SIZE_X) {
    const float weight_right = x_in_tile - static_cast<float>(left);
    if (top >= 0 && top < TILE_SIZE_Y) {
      const float weight_top = static_cast<float>(bottom) - y_in_tile;
      const float weight = weight_right * weight_top;
      uint32_t tile_idx = right + top * TILE_SIZE_X;
      atomicAdd(tile + tile_idx, src_red * weight);
      atomicAdd(tile + tile_idx + pixels_in_tile, src_green * weight);
      atomicAdd(tile + tile_idx + 2 * pixels_in_tile, src_blue * weight);
    }
    if (bottom >= 0 && bottom < TILE_SIZE_Y) {
      const float weight_bottom = y_in_tile - static_cast<float>(top);
      const float weight = weight_right * weight_bottom;
      uint32_t tile_idx = right + bottom * TILE_SIZE_X;
      atomicAdd(tile + tile_idx, src_red * weight);
      atomicAdd(tile + tile_idx + pixels_in_tile, src_green * weight);
      atomicAdd(tile + tile_idx + 2 * pixels_in_tile, src_blue * weight);
    }
  }
}

__global__ void fast_splat_2d_kernel(
    const float *__restrict__ patch_list, const size_t patch_width,
    const size_t patch_height, const size_t patch_count,
    const float *__restrict__ position_list, const uint32_t *indices,
    const uint32_t *patches_per_tile, const uint32_t *tile_index_offsets,
    float *__restrict__ result, const size_t target_width,
    const size_t target_height) {
  uint32_t tile_id = blockIdx.x;

  uint32_t tiles_per_width = (target_width + TILE_SIZE_X - 1) / TILE_SIZE_X;
  uint32_t tile_x = tile_id % tiles_per_width;
  uint32_t tile_y = tile_id / tiles_per_width;

  uint32_t tile_x_px = tile_x * TILE_SIZE_X;
  uint32_t tile_y_px = tile_y * TILE_SIZE_Y;

  const uint32_t patch_pixel_count = patch_width * patch_height;

  float patch_radius_x = patch_width / 2.F;
  float patch_radius_y = patch_height / 2.F;

  // initialize a tile of shared memory with zeros (neutral element for
  // addition in the end)
  __shared__ float tile[TILE_SIZE_X * TILE_SIZE_Y * 3];
  for (uint32_t idx_in_tile = threadIdx.x;
       idx_in_tile < TILE_SIZE_X * TILE_SIZE_Y * 3; idx_in_tile += blockDim.x) {
    tile[idx_in_tile] = 0.f;
  }
  __syncthreads();

  // iterate over all patches that need to be splatet into this tile
  uint32_t patches_for_this_tile = patches_per_tile[tile_id];
  uint32_t tile_index_offsets_for_this_tile = tile_index_offsets[tile_id];
  for (uint32_t i = 0; i < patches_for_this_tile; i++) {
    uint32_t patch_id = indices[tile_index_offsets_for_this_tile + i];
    float patch_center_pos_x = position_list[patch_id];
    float patch_center_pos_y = position_list[patch_id + patch_count];
    float patch_left = patch_center_pos_x - patch_radius_x;
    float patch_top = patch_center_pos_y - patch_radius_y;
    float patch_left_in_tile = patch_left - tile_x_px;
    float patch_top_in_tile = patch_top - tile_y_px;
    // attomicly add the pixels of this patch to this tile at the correct
    // positions using bilinear interpolation
    for (uint32_t idx_in_patch = threadIdx.x;
         idx_in_patch < patch_height * patch_width;
         idx_in_patch += blockDim.x) {
      float src_red =
          patch_list[patch_id * patch_width * patch_height * 3 + idx_in_patch];
      float src_green = patch_list[patch_id * patch_width * patch_height * 3 +
                                   idx_in_patch + patch_pixel_count];
      float src_blue = patch_list[patch_id * patch_width * patch_height * 3 +
                                  idx_in_patch + 2 * patch_pixel_count];

      uint32_t x_in_patch = idx_in_patch % patch_width;
      uint32_t y_in_patch = idx_in_patch / patch_width;
      float x_in_tile = x_in_patch + patch_left_in_tile;
      float y_in_tile = y_in_patch + patch_top_in_tile;
      if (ceilf(x_in_tile) >= 0 && floorf(x_in_tile) < TILE_SIZE_X &&
          ceilf(y_in_tile) >= 0 && floorf(y_in_tile) < TILE_SIZE_Y) {
        bilinear_splat(src_red, src_green, src_blue, x_in_tile, y_in_tile,
                       tile);
      }
    }
  }
  __syncthreads();

  // add tile on top of the result. No attomic needed, as tiles don't overlap
  const uint32_t target_pixels = target_width * target_height;
  for (uint32_t idx_in_tile = threadIdx.x;
       idx_in_tile < TILE_SIZE_X * TILE_SIZE_Y * 3; idx_in_tile += blockDim.x) {
    uint32_t color_idx = idx_in_tile % 3;
    uint32_t pos_in_tile = idx_in_tile / 3;
    uint32_t x_in_tile = pos_in_tile % TILE_SIZE_X;
    uint32_t y_in_tile = pos_in_tile / TILE_SIZE_X;
    uint32_t x_in_result = tile_x_px + x_in_tile;
    uint32_t y_in_result = tile_y_px + y_in_tile;
    if (x_in_result >= target_width || y_in_result >= target_height) {
      continue;
    }
    uint32_t idx_in_result =
        y_in_result * target_width + x_in_result + color_idx * target_pixels;

    result[idx_in_result] += tile[idx_in_tile];
  }
}

extern "C" void
fast_splat_2d_cuda_impl(const float *__restrict__ patch_list,
                        const float *__restrict__ position_list,
                        const size_t patch_count, const size_t patch_width,
                        const size_t patch_height, float *__restrict__ result,
                        const size_t target_width, const size_t target_height) {
  // Determine how many target patches there will be
  size_t tiles_X = (target_width + TILE_SIZE_X - 1) / TILE_SIZE_X;
  size_t tiles_Y = (target_height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
  size_t total_tiles = tiles_X * tiles_Y;
  thrust::device_vector<uint8_t> used_patches_bitmap(total_tiles * patch_count);
  fflush(stdout);
  float patch_radius_x = patch_width / 2.F;
  float patch_radius_y = patch_height / 2.F;

  printf("DEBUG: 1. tiles_X: %lu, tiles_Y: %lu, "
         "total_tiles: %lu, patch_count: %lu, target_width: %lu, "
         "target_height: %lu\n",
         tiles_X, tiles_Y, total_tiles, patch_count, target_width,
         target_height);

  // one thread for every Patch*Target_patch
  const dim3 threads_find_kernel(64, 64);
  const dim3 grid_dim(
      (patch_count + threads_find_kernel.x - 1) / threads_find_kernel.x,
      (total_tiles + threads_find_kernel.y - 1) / threads_find_kernel.y);
  // DEBUG
  printf("DEBUG: threads_find_kernel: (%u, %u), grid_dim: (%u, %u)\n",
         threads_find_kernel.x, threads_find_kernel.y, grid_dim.x, grid_dim.y);
  // DEBUG
  find_source_patches_for_target_tiles<<<grid_dim, threads_find_kernel>>>(
      position_list, static_cast<uint32_t>(patch_count), patch_radius_x,
      patch_radius_y, static_cast<uint32_t>(target_width),
      static_cast<uint32_t>(total_tiles), used_patches_bitmap.data().get());

  // DEBUG
  thrust::host_vector<uint8_t> bitmap_cpu = used_patches_bitmap;
  printf("DEBUG: 4. used_patches_bitmap\n");
  for (size_t m = 0; m < total_tiles; m++) {
    printf("%lu: ", m);
    for (size_t n = 0; n < patch_count; n++) {
      printf("%u ", bitmap_cpu[m * patch_count + n]);
    }
    printf("\n");
  }
  // DEBUG

  const auto [indices, patches_per_tile, tile_index_offsets] =
      compute_indices_from_bitmap(used_patches_bitmap, total_tiles,
                                  patch_count);

  const size_t THREADS_SPLAT_KERNEL = 128;
  fast_splat_2d_kernel<<<total_tiles, THREADS_SPLAT_KERNEL>>>(
      patch_list, patch_width, patch_height, patch_count, position_list,
      indices.data().get(), patches_per_tile.data().get(),
      tile_index_offsets.data().get(), result, target_width, target_height);

  fflush(stdout);
}
