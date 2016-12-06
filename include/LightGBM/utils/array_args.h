#ifndef LIGHTGBM_UTILS_ARRAY_AGRS_H_
#define LIGHTGBM_UTILS_ARRAY_AGRS_H_

#include <vector>
#include <algorithm>

namespace LightGBM {

/*!
* \brief Contains some operation for a array, e.g. ArgMax, TopK.
*/
template<typename VAL_T>
class ArrayArgs {
public:
  inline static size_t ArgMax(const std::vector<VAL_T>& array) {
    if (array.empty()) {
      return 0;
    }
    size_t argMax = 0;
    for (size_t i = 1; i < array.size(); ++i) {
      if (array[i] > array[argMax]) {
        argMax = i;
      }
    }
    return argMax;
  }

  inline static size_t ArgMin(const std::vector<VAL_T>& array) {
    if (array.empty()) {
      return 0;
    }
    size_t argMin = 0;
    for (size_t i = 1; i < array.size(); ++i) {
      if (array[i] < array[argMin]) {
        argMin = i;
      }
    }
    return argMin;
  }

  inline static size_t ArgMax(const VAL_T* array, size_t n) {
    if (n <= 0) {
      return 0;
    }
    size_t argMax = 0;
    for (size_t i = 1; i < n; ++i) {
      if (array[i] > array[argMax]) {
        argMax = i;
      }
    }
    return argMax;
  }

  inline static size_t ArgMin(const VAL_T* array, size_t n) {
    if (n <= 0) {
      return 0;
    }
    size_t argMin = 0;
    for (size_t i = 1; i < n; ++i) {
      if (array[i] < array[argMin]) {
        argMin = i;
      }
    }
    return argMin;
  }

  inline static size_t Partition(std::vector<VAL_T>* array, size_t start, size_t end) {
    VAL_T& pivot = (*array)[end - 1];
    size_t p_idx = start;
    for (size_t i = start; i < end - 1; ++i) {
      if ((*array)[i] > pivot) {
        std::swap((*array)[p_idx], (*array)[i]);
        ++p_idx;
      }
    }
    std::swap((*array)[p_idx], (*array)[end - 1]);
    return p_idx;
  };

  inline static size_t ArgMaxAtK(std::vector<VAL_T>* array, size_t start, size_t end, size_t k) {
    if (start == end - 1) {
      return start;
    }
    size_t p_idx = Partition(array, start, end);
    if (p_idx == k) {
      return p_idx;
    }
    else if (k < p_idx) {
      return ArgMaxAtK(array, start, p_idx, k);
    }
    else {
      return ArgMaxAtK(array, p_idx + 1, end, k);
    }
  }

  inline static void MaxK(const std::vector<VAL_T>& array, size_t k, std::vector<VAL_T>* out) {
    out->clear();
    if (k <= 0) {
      return;
    }
    for (auto val : array) {
      out->push_back(val);
    }
    if (k >= array.size()) {
      return;
    }
    ArgMaxAtK(out, 0, out->size(), k - 1);
    out->erase(out->begin() + k, out->end());
  }

};

}  // namespace LightGBM

#endif   // LightGBM_UTILS_ARRAY_AGRS_H_

