#ifndef LIGHTGBM_META_H_
#define LIGHTGBM_META_H_

#include <cstdint>

#include <limits>
#include <vector>
#include <functional>
#include <memory>

namespace LightGBM {

/*! \brief Type of data size, it is better to use signed type*/
typedef int32_t data_size_t;
/*! \brief Type of score, and gradients */
typedef float score_t;

const score_t kMinScore = -std::numeric_limits<score_t>::infinity();

const score_t kEpsilon = 1e-15f;

const double kMissingValueRange = 1e-20f;

using ReduceFunction = std::function<void(const char*, char*, int)>;

using PredictFunction =
std::function<void(const std::vector<std::pair<int, double>>&, double* output)>;

#define NO_SPECIFIC (-1)

}  // namespace LightGBM

#endif   // LightGBM_META_H_
