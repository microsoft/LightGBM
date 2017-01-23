#ifndef LIGHTGBM_UTILS_RANDOM_H_
#define LIGHTGBM_UTILS_RANDOM_H_

#include <cstdint>

#include <random>
#include <vector>

namespace LightGBM {

/*!
* \brief A wrapper for random generator
*/
class Random {
public:
  /*!
  * \brief Constructor, with random seed
  */
  Random() {
    std::random_device rd;
    auto genrator = std::mt19937(rd());
    std::uniform_int_distribution<int> distribution(0, x);
    x = static_cast<unsigned int>(distribution(genrator));
  }
  /*!
  * \brief Constructor, with specific seed
  */
  Random(int seed) {
    x = static_cast<unsigned int>(seed);
  }
  /*!
  * \brief Generate random integer
  * \param lower_bound lower bound
  * \param upper_bound upper bound
  * \return The random integer between [lower_bound, upper_bound)
  */
  inline int NextInt(int lower_bound, int upper_bound) {
    return (fastrand()) % (upper_bound - lower_bound) + lower_bound;
  }
  /*!
  * \brief Generate random float data
  * \return The random float between [0.0, 1.0)
  */
  inline float NextFloat() {
    // get random float in [0,1)
    return static_cast<float>(fastrand()) / (32768.0f);
  }
  /*!
  * \brief Sample K data from {0,1,...,N-1}
  * \param N
  * \param K
  * \return K Ordered sampled data from {0,1,...,N-1}
  */
  inline std::vector<int> Sample(int N, int K) {
    std::vector<int> ret;
    if (K > N || K < 0) {
      return ret;
    }
    for (int i = 0; i < N; ++i) {
      double prob = (K - ret.size()) / static_cast<double>(N - i);
      if (NextFloat() < prob) {
        ret.push_back(i);
      }
    }
    return ret;
  }
private:
  inline int fastrand() {
    x = (214013 * x + 2531011);
    return (x >> 16) & 0x7FFF;
  }
  int x = 123456789;
};


}  // namespace LightGBM

#endif   // LightGBM_UTILS_RANDOM_H_
