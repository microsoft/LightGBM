#ifndef LIGHTGBM_META_H_
#define LIGHTGBM_META_H_

#include <cstdint>

#include <limits>
#include <vector>
#include <functional>
#include <memory>
#include <cstdlib>

#if defined(_WIN32)

#include <malloc.h>

#else

#include <mm_malloc.h>

#endif // (_WIN32)



namespace LightGBM {

/*! \brief Type of data size, it is better to use signed type*/
typedef int32_t data_size_t;

const float kMinScore = -std::numeric_limits<float>::infinity();

const float kEpsilon = 1e-15f;

using ReduceFunction = std::function<void(const char*, char*, int)>;

using PredictFunction =
std::function<void(const std::vector<std::pair<int, double>>&, double* output)>;

#define NO_SPECIFIC (-1)

}  // namespace LightGBM

#endif   // LightGBM_META_H_
