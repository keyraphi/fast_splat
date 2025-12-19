#pragma once

#include <cstddef>

void fast_splat_2d_cpu_impl(const float *__restrict__ patch_list,
                            const float *__restrict__ position_list,
                            const size_t patch_count, const size_t patch_width,
                            const size_t patch_height,
                            float *__restrict__ result,
                            const size_t target_width, size_t target_height);
