#ifndef LIGHTGBM_META_H_
#define LIGHTGBM_META_H_

#include <cstdint>

#include <limits>
#include <vector>
#include <functional>

namespace LightGBM {

/*! \brief Type of data size, it is better to use signed type*/
typedef int32_t data_size_t;
/*! \brief Type of score, and gradients */
typedef double score_t;

const score_t kMinScore = -std::numeric_limits<score_t>::infinity();

const score_t kEpsilon = 1e-15f;

template<typename T>
std::vector<const T*> ConstPtrInVectorWarpper(std::vector<T*> input) {
  return std::vector<const T*>(input.begin(), input.end());
}

using ReduceFunction = std::function<void(const char*, char*, int)>;

}  // namespace LightGBM

#endif   // LightGBM_META_H_
