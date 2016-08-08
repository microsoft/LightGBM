#ifndef LIGHTGBM_UTILS_THREADING_H_
#define LIGHTGBM_UTILS_THREADING_H_

#include <omp.h>

#include <vector>
#include <functional>

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
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < num_threads; ++i) {
      INDEX_T inner_start = start + num_inner * i;
      INDEX_T inner_end = inner_start + num_inner;
      if (inner_end > end) { inner_end = end; }
      if (inner_start < end) {
        inner_fun(i, inner_start, inner_end);
      }
    }
  }
};

}   // namespace LightGBM

#endif   // LightGBM_UTILS_THREADING_H_
