#include "fast_splat_2d_cpu.h"
#include <cmath>
#include <cstddef>

void bilinear_splat_pixel(const float red, const float green, const float blue,
                          const float x, const float y, float *image,
                          size_t width, size_t height) {
  int x_left = static_cast<int>(std::floor(x));
  int x_right = x_left + 1;
  int y_top = static_cast<int>(std::floor(y));
  int y_bottom = y_top + 1;
  size_t pixel_count = width * height;

  if (x_left >= 0 && x_left < static_cast<int>(width)) {
    const float weight_left = static_cast<float>(x_right) - x;
    if (y_top >= 0 && y_top < static_cast<int>(height)) {
      const float weight_top = static_cast<float>(y_bottom) - y;
      const float weight_top_left = weight_left * weight_top;
#pragma omp atomic
      image[y_top * width + static_cast<size_t>(x_left)] +=
          weight_top_left * red;
#pragma omp atomic
      image[y_top * width + static_cast<size_t>(x_left) + pixel_count] +=
          weight_top_left * green;
#pragma omp atomic
      image[y_top * width + static_cast<size_t>(x_left) + 2 * pixel_count] +=
          weight_top_left * blue;
    }
    if (y_bottom >= 0 && y_bottom < static_cast<int>(height)) {
      const float weight_bottom = y - static_cast<float>(y_top);
      const float weight_bottom_left = weight_left * weight_bottom;
#pragma omp atomic
      image[y_bottom * width + static_cast<size_t>(x_left)] +=
          weight_bottom_left * red;
#pragma omp atomic
      image[y_bottom * width + static_cast<size_t>(x_left) + pixel_count] +=
          weight_bottom_left * green;
#pragma omp atomic
      image[y_bottom * width + static_cast<size_t>(x_left) + 2 * pixel_count] +=
          weight_bottom_left * blue;
    }
  }
  if (x_right >= 0 && x_right < static_cast<int>(width)) {
    const float weight_right = x - static_cast<float>(x_left);
    if (y_top >= 0 && y_top < static_cast<int>(height)) {
      const float weight_top = static_cast<float>(y_bottom) - y;
      const float weight_top_right = weight_right * weight_top;
#pragma omp atomic
      image[y_top * width + static_cast<size_t>(x_right) ] +=
          weight_top_right * red;
#pragma omp atomic
      image[y_top * width + static_cast<size_t>(x_right) + pixel_count] +=
          weight_top_right * green;
#pragma omp atomic
      image[y_top * width + static_cast<size_t>(x_right) + 2*pixel_count] +=
          weight_top_right * blue;
    }
    if (y_bottom >= 0 && y_bottom < static_cast<int>(height)) {
      const float weight_bottom = y - static_cast<float>(y_top);
      const float weight_bottom_right = weight_right * weight_bottom;
#pragma omp atomic
      image[y_bottom * width + static_cast<size_t>(x_right) ] +=
          weight_bottom_right * red;
#pragma omp atomic
      image[y_bottom * width + static_cast<size_t>(x_right) + pixel_count] +=
          weight_bottom_right * green;
#pragma omp atomic
      image[y_bottom * width + static_cast<size_t>(x_right) + 2*pixel_count] +=
          weight_bottom_right * blue;
    }
  }
}

void fast_splat_2d_cpu_impl(const float *__restrict__ patch_list,
                            const float *__restrict__ position_list,
                            const size_t patch_count, const size_t patch_width,
                            const size_t patch_height,
                            float *__restrict__ result,
                            const size_t target_width,
                            const size_t target_height) {
  float patch_radius_x = static_cast<float>(patch_width) / 2.F;
  float patch_radius_y = static_cast<float>(patch_height) / 2.F;
  const size_t patch_pixel_count = patch_width*patch_height;

#pragma omp parallel for
  for (size_t i = 0; i < patch_count; ++i) {
    const float *patch = patch_list + i * patch_height * patch_width * 3;
    const float pos_x = position_list[i];
    const float pos_y = position_list[i + patch_count];

    const float start_pos_x = pos_x - patch_radius_x;
    const float start_pos_y = pos_y - patch_radius_y;
    for (size_t y = 0; y < patch_height; y++) {
      for (size_t x = 0; x < patch_width; x++) {
        const float red = patch[y * patch_width + x];
        const float green =
            patch[y * patch_width + x + 1 * patch_pixel_count];
        const float blue =
            patch[y * patch_width + x + 2 * patch_pixel_count];
        const float pixel_pos_x = start_pos_x + static_cast<float>(x);
        const float pixel_pos_y = start_pos_y + static_cast<float>(y);
        bilinear_splat_pixel(red, green, blue, pixel_pos_x, pixel_pos_y, result,
                             target_width, target_height);
      }
    }
  }
}
