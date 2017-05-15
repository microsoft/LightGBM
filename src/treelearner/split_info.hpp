#ifndef LIGHTGBM_TREELEARNER_SPLIT_INFO_HPP_
#define LIGHTGBM_TREELEARNER_SPLIT_INFO_HPP_

#include <LightGBM/meta.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <functional>

namespace LightGBM {

/*!
* \brief Used to store some information for gain split point
*/
struct SplitInfo {
public:
  /*! \brief Feature index */
  int feature;
  /*! \brief Split threshold */
  uint32_t threshold;

  uint32_t default_bin_for_zero;
  /*! \brief Left output after split */
  double left_output;
  /*! \brief Right output after split */
  double right_output;
  /*! \brief Split gain */
  double gain;
  /*! \brief Left number of data after split */
  data_size_t left_count;
  /*! \brief Right number of data after split */
  data_size_t right_count;
  /*! \brief Left sum gradient after split */
  double left_sum_gradient;
  /*! \brief Left sum hessian after split */
  double left_sum_hessian;
  /*! \brief Right sum gradient after split */
  double right_sum_gradient;
  /*! \brief Right sum hessian after split */
  double right_sum_hessian;

  SplitInfo() {
    // initialize with -1 and -inf gain
    feature = -1;
    gain = kMinScore;
  }

  inline void Reset() {
    // initialize with -1 and -inf gain
    feature = -1;
    gain = kMinScore;
  }

  inline bool operator > (const SplitInfo &si) const;

  inline bool operator == (const SplitInfo &si) const;

  inline static void MaxReducer(const char* src, char* dst, int len) {
    const int type_size = sizeof(SplitInfo);
    int used_size = 0;
    const SplitInfo* p1;
    SplitInfo* p2;
    while (used_size < len) {
      p1 = reinterpret_cast<const SplitInfo*>(src);
      p2 = reinterpret_cast<SplitInfo*>(dst);
      if (*p1 > *p2) {
        // copy
        std::memcpy(dst, src, type_size);
      }
      src += type_size;
      dst += type_size;
      used_size += type_size;
    }
  }
};



inline bool SplitInfo::operator > (const SplitInfo& si) const {
  double local_gain = this->gain;
  double other_gain = si.gain;
  // replace nan with -inf
  if (local_gain == NAN) {
    local_gain = kMinScore;
  }
  // replace nan with -inf
  if (other_gain == NAN) {
    other_gain = kMinScore;
  }
  int local_feature = this->feature;
  int other_feature = si.feature;
  // replace -1 with max int
  if (local_feature == -1) {
    local_feature = INT32_MAX;
  }
  // replace -1 with max int
  if (other_feature == -1) {
    other_feature = INT32_MAX;
  }
  if (local_gain != other_gain) {
    return local_gain > other_gain;
  } else {
    // if same gain, use smaller feature
    return local_feature < other_feature;
  }
}

inline bool SplitInfo::operator == (const SplitInfo& si) const {
  double local_gain = this->gain;
  double other_gain = si.gain;
  // replace nan with -inf
  if (local_gain == NAN) {
    local_gain = kMinScore;
  }
  // replace nan with -inf
  if (other_gain == NAN) {
    other_gain = kMinScore;
  }
  int local_feature = this->feature;
  int other_feature = si.feature;
  // replace -1 with max int
  if (local_feature == -1) {
    local_feature = INT32_MAX;
  }
  // replace -1 with max int
  if (other_feature == -1) {
    other_feature = INT32_MAX;
  }
  if (local_gain != other_gain) {
    return local_gain == other_gain;
  } else {
    // if same gain, use smaller feature
    return local_feature == other_feature;
  }
}

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_SPLIT_INFO_HPP_
