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
  template <typename INDEX_T>
  static inline void BlockInfo(INDEX_T cnt, INDEX_T min_cnt_per_block,
                               int* out_nblock, INDEX_T* block_size) {
    int num_threads = 1;
#pragma omp parallel
#pragma omp master
    { num_threads = omp_get_num_threads(); }
    *out_nblock = std::min<int>(
        num_threads,
        static_cast<int>((cnt + min_cnt_per_block - 1) / min_cnt_per_block));
    if (*out_nblock > 1) {
      *block_size = SIZE_ALIGNED((cnt + (*out_nblock) - 1) / (*out_nblock));
    } else {
      *block_size = cnt;
    }
  }
  template <typename INDEX_T>
  static inline void BlockInfo(int num_threads, INDEX_T cnt,
                               INDEX_T min_cnt_per_block, int* out_nblock,
                               INDEX_T* block_size) {
    *out_nblock = std::min<int>(
        num_threads,
        static_cast<int>((cnt + min_cnt_per_block - 1) / min_cnt_per_block));
    if (*out_nblock > 1) {
      *block_size = SIZE_ALIGNED((cnt + (*out_nblock) - 1) / (*out_nblock));
    } else {
      *block_size = cnt;
    }
  }
  template <typename INDEX_T>
  static inline int For(
      INDEX_T start, INDEX_T end, INDEX_T min_block_size,
      const std::function<void(int, INDEX_T, INDEX_T)>& inner_fun) {
    int n_block = 1;
    INDEX_T num_inner = end - start;
    BlockInfo<INDEX_T>(end - start, min_block_size, &n_block, &num_inner);
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
  template <typename INDEX_T, bool use_index>
  static inline void BalancedFor(int n, INDEX_T total_size, const int* indices,
                                 const std::vector<INDEX_T>& size,
                                 const std::function<void(int, int)>& inner_fun) {
    int num_threads = 1;
#pragma omp parallel
#pragma omp master
    { num_threads = omp_get_num_threads(); }
    std::vector<std::vector<int>> groups(1);
    groups.back().push_back(0);
    std::vector<INDEX_T> group_sizes(1, size[0]);
    INDEX_T avg_size = total_size / num_threads;
    INDEX_T rest_size = total_size - size[0];
    INDEX_T rest_group = num_threads;
    for (int i = 1; i < n; ++i) {
      if (group_sizes.back() + size[i] > avg_size) {
        groups.emplace_back();
        group_sizes.emplace_back(0);
        --rest_group;
        avg_size = rest_size / rest_group;
      }
      group_sizes.back() += size[i];
      groups.back().push_back(i);
      rest_size -= size[i];
    }
    int n_block = static_cast<int>(groups.size());
    OMP_INIT_EX();
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < n_block; ++i) {
      OMP_LOOP_EX_BEGIN();
      for (auto j : groups[i]) {
        if (use_index) {
          inner_fun(i, indices[j]);
        } else {
          inner_fun(i, j);
        }
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }
};

}   // namespace LightGBM

#endif   // LightGBM_UTILS_THREADING_H_
