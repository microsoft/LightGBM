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
  static inline void For(INDEX_T start, INDEX_T end, const std::function<void(int, INDEX_T, INDEX_T)>& inner_fun) {
    int num_threads = 1;
    #pragma omp parallel
    #pragma omp master
    {
      num_threads = omp_get_num_threads();
    }
    INDEX_T num_inner = (end - start + num_threads - 1) / num_threads;
    if (num_inner <= 0) { num_inner = 1; }
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < num_threads; ++i) {
      OMP_LOOP_EX_BEGIN();
      INDEX_T inner_start = start + num_inner * i;
      INDEX_T inner_end = inner_start + num_inner;
      if (inner_end > end) { inner_end = end; }
      if (inner_start < end) {
        inner_fun(i, inner_start, inner_end);
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }
};

}   // namespace LightGBM

#endif   // LightGBM_UTILS_THREADING_H_
