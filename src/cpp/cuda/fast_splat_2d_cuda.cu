#include "fast_splat_2d_cuda.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

#define N_THREADS_X 64
#define N_THREADS_Y 64

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
 * on a target patch. The target patch size is defined by N_THREADS_X and
 * N_THREADS_Y. The target position is implicit from the thread id. There is one
 * thread for every targe patch and every position list entry. The result is a
 * bit map of shape MxN, where M is the total number of target positions and N
 * is the number of positions in the position list (patch_count). Each input
 * position has a radius of influence (rectangular) with the given
 * patch_radius_x and patch_radius_y shape. If the influence area of the input
 * position and the target patch overlap in any way the corresponding bitmap
 * value should be set to 1, otherwise 0;
 */
__global__ void find_source_patches_for_target_patches(
    const float *__restrict__ position_list, const size_t patch_count,
    const float patch_radius_x, const float patch_radius_y,
    const size_t target_width, const size_t target_count, uint32_t *bitmap) {

  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t target_m = tid / patch_count;
  size_t position_n = tid % patch_count;
  if (position_n >= patch_count || target_m >= target_count) {
    return;
  }

  size_t patches_per_row = (target_width + N_THREADS_X - 1) / N_THREADS_X;
  size_t target_patch_y = target_m / patches_per_row;
  size_t target_patch_x = target_m % patches_per_row;
  auto target_x_left = static_cast<float>(target_patch_x * N_THREADS_X);
  auto target_y_top = static_cast<float>(target_patch_y * N_THREADS_Y);
  float target_x_right = target_x_left + N_THREADS_X;
  float target_y_bottom = target_y_top + N_THREADS_Y;

  float pos_x = position_list[position_n * 2];
  float pos_y = position_list[position_n * 2 + 1];

  uint32_t is_in_m = 0;

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

auto compute_indices_from_bitmap(thrust::device_vector<uint32_t> &bitmap,
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

  // DEBUG 5:
  thrust::host_vector<uint32_t> prefix_sums_cpu = prefix_sum;
  printf("DEBUG: 5. prefix_sums\n");
  for (size_t m = 0; m < rows; m++) {
    printf("%lu: ", m);
    for (size_t n = 0; n < columns; n++) {
      printf("%u ", prefix_sums_cpu[m * columns + n]);
    }
    printf("\n");
  }
  // END DEBUG

  // number of entries per row
  thrust::device_vector<uint32_t> row_sums(rows);
  thrust::reduce_by_key(keys_begin, keys_begin + (rows * columns),
                        bitmap.begin(), thrust::discard_iterator<>(),
                        row_sums.begin());

  // DEBUG 6:
  thrust::host_vector<uint32_t> row_sums_cpu = row_sums;
  printf("DEBUG: 6. row_sums\n");
  for (size_t m = 0; m < rows; m++) {
    printf("%lu: %u ", m, row_sums_cpu[m]);
    printf("\n");
  }
  // END DEBUG
  // start of each row
  thrust::device_vector<uint32_t> row_offsets(rows);
  thrust::exclusive_scan(row_sums.begin(), row_sums.end(), row_offsets.begin());

  // DEBUG 7:
  thrust::host_vector<uint32_t> row_offsets_cpu = row_offsets;
  printf("DEBUG: 7. row_offsets\n");
  for (size_t m = 0; m < rows; m++) {
    printf("%lu: %u ", m, row_offsets_cpu[m]);
    printf("\n");
  }
  // END DEBUG

  //  write indices of patches together
  auto column_indices_begin = thrust::make_transform_iterator(
      thrust::counting_iterator<uint32_t>(0),
      [columns] __host__ __device__(uint32_t idx) {
        return idx % static_cast<uint32_t>(columns);
      });

  // Use copy_if to extract only indices where bitmap is 1
  // Count the number of 1s first to allocate the right size
  uint32_t total_ones = thrust::reduce(bitmap.begin(), bitmap.end());
  thrust::device_vector<uint32_t> result(total_ones);

  thrust::copy_if(column_indices_begin, // Source: column indices
                  column_indices_begin + (rows * columns), // End
                  bitmap.begin(), // Stencil: bitmap values (0 or 1)
                  result.begin(), // Destination
                  [] __host__ __device__(uint32_t val) { return val == 1; });
  // DEBUG 8:
  thrust::host_vector<uint32_t> result_cpu = result;
  printf("DEBUG: 8. result\n");
  size_t i = 0;
  for (size_t m = 0; m < rows; m++) {
    printf("%lu: ", m);
    for (size_t n = 0; n < row_sums_cpu[m]; n++, i++) {
      printf("%u ", result_cpu[i]);
    }
    printf("\n");
  }
  // END DEBUG

  return {result, row_sums, row_offsets};
}

__device__ inline void
bilinear_splat(const float src_red, const float src_green, const float src_blue,
               const float x_in_tile, const float y_in_tile, float *tile) {
  const int left = x_in_tile;
  const int right = left + 1;
  const int top = y_in_tile;
  const int bottom = top + 1;

  if (left >= 0 && left < N_THREADS_X) {
    const float weight_left = static_cast<float>(right) - x_in_tile;
    if (top >= 0 && top < N_THREADS_Y) {
      const float weight_top = static_cast<float>(bottom) - y_in_tile;
      const float weight = weight_left * weight_top;
      uint32_t tile_idx = left * 3 + top * N_THREADS_X * 3;
      atomicAdd(tile + tile_idx, src_red * weight);
      atomicAdd(tile + tile_idx + 1, src_red * weight);
      atomicAdd(tile + tile_idx + 2, src_red * weight);
    }
    if (bottom >= 0 && bottom < N_THREADS_Y) {
      const float weight_bottom = y_in_tile - static_cast<float>(top);
      const float weight = weight_left * weight_bottom;
      uint32_t tile_idx = left * 3 + bottom * N_THREADS_X * 3;
      atomicAdd(tile + tile_idx, src_red * weight);
      atomicAdd(tile + tile_idx + 1, src_red * weight);
      atomicAdd(tile + tile_idx + 2, src_red * weight);
    }
  }
  if (right >= 0 && right < N_THREADS_X) {
    const float weight_right = x_in_tile - static_cast<float>(left);
    if (top >= 0 && top < N_THREADS_Y) {
      const float weight_top = static_cast<float>(bottom) - y_in_tile;
      const float weight = weight_right * weight_top;
      uint32_t tile_idx = right * 3 + top * N_THREADS_X * 3;
      atomicAdd(tile + tile_idx, src_red * weight);
      atomicAdd(tile + tile_idx + 1, src_red * weight);
      atomicAdd(tile + tile_idx + 2, src_red * weight);
    }
    if (bottom >= 0 && bottom < N_THREADS_Y) {
      const float weight_bottom = y_in_tile - static_cast<float>(top);
      const float weight = weight_right * weight_bottom;
      uint32_t tile_idx = right * 3 + bottom * N_THREADS_X * 3;
      atomicAdd(tile + tile_idx, src_red * weight);
      atomicAdd(tile + tile_idx + 1, src_red * weight);
      atomicAdd(tile + tile_idx + 2, src_red * weight);
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

  uint32_t tiles_per_width = target_width / N_THREADS_X;
  uint32_t tile_x = tile_id % tiles_per_width;
  uint32_t tile_y = tile_id / tiles_per_width;

  uint32_t tile_x_px = tile_x * N_THREADS_X;
  uint32_t tile_y_px = tile_y * N_THREADS_X;

  float patch_radius_x = patch_width / 2.F;
  float patch_radius_y = patch_height / 2.F;

  // initialize a tile of shared memory with zeros (neutral element for
  // addition in the end)
  __shared__ float tile[N_THREADS_X * N_THREADS_Y * 3];
  for (uint32_t idx_in_tile = threadIdx.x;
       idx_in_tile < N_THREADS_X * N_THREADS_Y * 3; idx_in_tile += blockDim.x) {
    tile[idx_in_tile] = 0.f;
  }
  __syncthreads();

  // iterate over all patches that need to be splatet into this tile
  uint32_t patches_for_this_tile = patches_per_tile[tile_id];
  // if (threadIdx.x == 0) {
  //   printf("patches_per_tile[%u]: %u\n", tile_id, patches_for_this_tile);
  // }
  uint32_t tile_index_offsets_for_this_tile = tile_index_offsets[tile_id];
  for (uint32_t i = 0; i < patches_for_this_tile; i++) {
    uint32_t patch_idx = indices[tile_index_offsets_for_this_tile + i];
    float patch_center_pos_x = position_list[patch_idx * 2];
    float patch_center_pos_y = position_list[patch_idx * 2 + 1];
    float patch_left = patch_center_pos_x - patch_radius_x;
    float patch_top = patch_center_pos_y - patch_radius_y;
    float patch_left_in_tile = patch_left - tile_x_px;
    float patch_top_in_tile = patch_top - tile_y_px;
    // if (threadIdx.x == 0) {
    //   printf("patch center: (%f, %f),  top left: (%f, %f), tile pos: (%u, %u), in tile: (%f, %f)\n",
    //          patch_center_pos_x, patch_center_pos_y, patch_left, patch_top,
    //          tile_x_px, tile_y_px, patch_left_in_tile, patch_top_in_tile);
    // }

    // add the pixels of this patch to this tile at the correct positions using
    // bilinear interpolation
    for (uint32_t idx_in_patch = 0; idx_in_patch < patch_height * patch_width;
         idx_in_patch += blockDim.x) {
      float src_red = patch_list[patch_idx * patch_width * patch_height * 3 +
                                 idx_in_patch * 3];
      float src_green = patch_list[patch_idx * patch_width * patch_height * 3 +
                                   idx_in_patch * 3 + 1];
      float src_blue = patch_list[patch_idx * patch_width * patch_height * 3 +
                                  idx_in_patch * 3 + 2];

      uint32_t x_in_patch = idx_in_patch % patch_width;
      uint32_t y_in_patch = idx_in_patch % patch_height;
      float x_in_tile = x_in_patch + patch_left_in_tile;
      float y_in_tile = y_in_patch + patch_top_in_tile;
      if (ceilf(x_in_tile) >= 0 && floorf(x_in_tile) < N_THREADS_X &&
          ceilf(y_in_tile) >= 0 && floorf(y_in_tile) < N_THREADS_Y) {
        bilinear_splat(src_red, src_green, src_blue, x_in_tile, y_in_tile,
                       tile);
        // if(threadIdx.x == 0) {
        //   printf("tile[%f]: %f\n", floorf(x_in_tile), tile[int(floorf(x_in_tile))]);
        // }
      }
    }
  }
  __syncthreads();
  // add tile on top of the result. No attomic needed, as tiles don't overlap
  for (uint32_t idx_in_tile = threadIdx.x;
       idx_in_tile < N_THREADS_X * N_THREADS_Y; idx_in_tile += blockDim.x) {
    uint32_t y_in_tile = idx_in_tile / N_THREADS_X;
    uint32_t x_in_tile = idx_in_tile % N_THREADS_X;
    uint32_t x_in_result = tile_x_px + x_in_tile;
    uint32_t y_in_result = tile_y_px + y_in_tile;
    // if (tile_id == 15) {
    //   printf("tile: (%u, %u): %f\n", x_in_tile, y_in_tile, tile[idx_in_tile]);
    // }
    if (x_in_result >= target_width || y_in_result >= target_height) {
      continue;
    }
    uint32_t idx_in_result = y_in_result * target_width * 3 + x_in_result * 3;

    // if (tile[idx_in_tile] > 0) {
    //   printf("nonzero tile\n");
    // }
    result[idx_in_result] += tile[idx_in_tile];
    result[idx_in_result + 1] += tile[idx_in_tile + 1];
    result[idx_in_result + 2] += tile[idx_in_tile + 2];
  }
}

extern "C" void
fast_splat_2d_cuda_impl(const float *__restrict__ patch_list,
                        const float *__restrict__ position_list,
                        const size_t patch_count, const size_t patch_width,
                        const size_t patch_height, float *__restrict__ result,
                        const size_t target_width, const size_t target_height) {
  // Determine how many target patches there will be
  size_t target_patches_X = (target_width + N_THREADS_X - 1) / N_THREADS_X;
  size_t target_patches_Y = (target_height + N_THREADS_Y - 1) / N_THREADS_Y;
  size_t m_target_patches = target_patches_X * target_patches_Y;
  thrust::device_vector<uint32_t> used_patches_bitmap(m_target_patches *
                                                      patch_count);
  fflush(stdout);
  printf("DEBUG: 1. target_patches_X: %lu, target_patches_Y: %lu, "
         "m_target_patches: %lu\n",
         target_patches_X, target_patches_Y, m_target_patches);
  float patch_radius_x = patch_width / 2.F;
  float patch_radius_y = patch_height / 2.F;
  printf("DEBUG: 2. patch_radius_x: %f, patch_radius_y: %f\n", patch_radius_x,
         patch_radius_y);

  // one thread for every Patch*Target_patch
  const size_t THREADS = 256;
  const size_t NM_BLOCKS =
      (m_target_patches * patch_count + THREADS - 1) / THREADS;
  printf("DEBUG: 3. THREADS: %lu, NM_BLOCKS: %lu\n", THREADS, NM_BLOCKS);
  find_source_patches_for_target_patches<<<NM_BLOCKS, THREADS>>>(
      position_list, patch_count, patch_radius_x, patch_radius_y, target_width,
      m_target_patches, used_patches_bitmap.data().get());
  // DEBUG 4:
  thrust::host_vector<uint32_t> bitmap_cpu = used_patches_bitmap;
  printf("DEBUG: 4. used_patches_bitmap\n");
  for (size_t m = 0; m < m_target_patches; m++) {
    printf("%lu: ", m);
    for (size_t n = 0; n < patch_count; n++) {
      printf("%u ", bitmap_cpu[m * patch_count + n]);
    }
    printf("\n");
  }
  // END DEBUG

  const auto [indices, patches_per_tile, tile_index_offsets] =
      compute_indices_from_bitmap(used_patches_bitmap, m_target_patches,
                                  patch_count);

  fast_splat_2d_kernel<<<m_target_patches, 256>>>(
      patch_list, patch_width, patch_height, patch_count, position_list,
      indices.data().get(), patches_per_tile.data().get(),
      tile_index_offsets.data().get(), result, target_width, target_height);

  fflush(stdout);
}
