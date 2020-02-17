/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_UTILS_THREADING_H_
#define LIGHTGBM_UTILS_THREADING_H_

#include <LightGBM/utils/openmp_wrapper.h>

#include <functional>
#include <vector>

namespace LightGBM {

class Threading {
 public:
  template<typename INDEX_T>
  static inline INDEX_T For(
      INDEX_T start, INDEX_T end,
      INDEX_T min_block_size, const std::function<void(int, INDEX_T, INDEX_T)>& inner_fun) {
    int num_threads = 1;
    #pragma omp parallel
    #pragma omp master
    {
      num_threads = omp_get_num_threads();
    }
    int n_block = std::min<int>(
        num_threads, (end - start + min_block_size - 1) / min_block_size);
    INDEX_T num_inner = SIZE_ALIGNED((end - start + n_block - 1) / n_block);
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < n_block; ++i) {
      OMP_LOOP_EX_BEGIN();
      INDEX_T inner_start = start + num_inner * i;
      INDEX_T inner_end = std::min(end, inner_start + num_inner);
      inner_fun(i, inner_start, inner_end);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    return n_block;
  }
};

}   // namespace LightGBM

#endif   // LightGBM_UTILS_THREADING_H_
