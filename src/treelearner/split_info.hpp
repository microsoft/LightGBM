/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_SPLIT_INFO_HPP_
#define LIGHTGBM_TREELEARNER_SPLIT_INFO_HPP_

#include <LightGBM/meta.h>

#include <limits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

namespace LightGBM {

/*!
* \brief Used to store some information for gain split point
*/
struct SplitInfo {
 public:
  /*! \brief Feature index */
  int feature = -1;
  /*! \brief Split threshold */
  uint32_t threshold = 0;
  /*! \brief Left number of data after split */
  data_size_t left_count = 0;
  /*! \brief Right number of data after split */
  data_size_t right_count = 0;
  int num_cat_threshold = 0;
  /*! \brief Left output after split */
  double left_output = 0.0;
  /*! \brief Right output after split */
  double right_output = 0.0;
  /*! \brief Split gain */
  double gain = kMinScore;
  /*! \brief Left sum gradient after split */
  double left_sum_gradient = 0;
  /*! \brief Left sum hessian after split */
  double left_sum_hessian = 0;
  /*! \brief Right sum gradient after split */
  double right_sum_gradient = 0;
  /*! \brief Right sum hessian after split */
  double right_sum_hessian = 0;
  std::vector<uint32_t> cat_threshold;
  /*! \brief True if default split is left */
  bool default_left = true;
  int8_t monotone_type = 0;
  double min_constraint = -std::numeric_limits<double>::max();
  double max_constraint = std::numeric_limits<double>::max();
  inline static int Size(int max_cat_threshold) {
    return 2 * sizeof(int) + sizeof(uint32_t) + sizeof(bool) + sizeof(double) * 9 + sizeof(data_size_t) * 2 + max_cat_threshold * sizeof(uint32_t) + sizeof(int8_t);
  }

  inline void CopyTo(char* buffer) const {
    std::memcpy(buffer, &feature, sizeof(feature));
    buffer += sizeof(feature);
    std::memcpy(buffer, &left_count, sizeof(left_count));
    buffer += sizeof(left_count);
    std::memcpy(buffer, &right_count, sizeof(right_count));
    buffer += sizeof(right_count);
    std::memcpy(buffer, &gain, sizeof(gain));
    buffer += sizeof(gain);
    std::memcpy(buffer, &threshold, sizeof(threshold));
    buffer += sizeof(threshold);
    std::memcpy(buffer, &left_output, sizeof(left_output));
    buffer += sizeof(left_output);
    std::memcpy(buffer, &right_output, sizeof(right_output));
    buffer += sizeof(right_output);
    std::memcpy(buffer, &left_sum_gradient, sizeof(left_sum_gradient));
    buffer += sizeof(left_sum_gradient);
    std::memcpy(buffer, &left_sum_hessian, sizeof(left_sum_hessian));
    buffer += sizeof(left_sum_hessian);
    std::memcpy(buffer, &right_sum_gradient, sizeof(right_sum_gradient));
    buffer += sizeof(right_sum_gradient);
    std::memcpy(buffer, &right_sum_hessian, sizeof(right_sum_hessian));
    buffer += sizeof(right_sum_hessian);
    std::memcpy(buffer, &default_left, sizeof(default_left));
    buffer += sizeof(default_left);
    std::memcpy(buffer, &monotone_type, sizeof(monotone_type));
    buffer += sizeof(monotone_type);
    std::memcpy(buffer, &min_constraint, sizeof(min_constraint));
    buffer += sizeof(min_constraint);
    std::memcpy(buffer, &max_constraint, sizeof(max_constraint));
    buffer += sizeof(max_constraint);
    std::memcpy(buffer, &num_cat_threshold, sizeof(num_cat_threshold));
    buffer += sizeof(num_cat_threshold);
    std::memcpy(buffer, cat_threshold.data(), sizeof(uint32_t) * num_cat_threshold);
  }

  void CopyFrom(const char* buffer) {
    std::memcpy(&feature, buffer, sizeof(feature));
    buffer += sizeof(feature);
    std::memcpy(&left_count, buffer, sizeof(left_count));
    buffer += sizeof(left_count);
    std::memcpy(&right_count, buffer, sizeof(right_count));
    buffer += sizeof(right_count);
    std::memcpy(&gain, buffer, sizeof(gain));
    buffer += sizeof(gain);
    std::memcpy(&threshold, buffer, sizeof(threshold));
    buffer += sizeof(threshold);
    std::memcpy(&left_output, buffer, sizeof(left_output));
    buffer += sizeof(left_output);
    std::memcpy(&right_output, buffer, sizeof(right_output));
    buffer += sizeof(right_output);
    std::memcpy(&left_sum_gradient, buffer, sizeof(left_sum_gradient));
    buffer += sizeof(left_sum_gradient);
    std::memcpy(&left_sum_hessian, buffer, sizeof(left_sum_hessian));
    buffer += sizeof(left_sum_hessian);
    std::memcpy(&right_sum_gradient, buffer, sizeof(right_sum_gradient));
    buffer += sizeof(right_sum_gradient);
    std::memcpy(&right_sum_hessian, buffer, sizeof(right_sum_hessian));
    buffer += sizeof(right_sum_hessian);
    std::memcpy(&default_left, buffer, sizeof(default_left));
    buffer += sizeof(default_left);
    std::memcpy(&monotone_type, buffer, sizeof(monotone_type));
    buffer += sizeof(monotone_type);
    std::memcpy(&min_constraint, buffer, sizeof(min_constraint));
    buffer += sizeof(min_constraint);
    std::memcpy(&max_constraint, buffer, sizeof(max_constraint));
    buffer += sizeof(max_constraint);
    std::memcpy(&num_cat_threshold, buffer, sizeof(num_cat_threshold));
    buffer += sizeof(num_cat_threshold);
    cat_threshold.resize(num_cat_threshold);
    std::memcpy(cat_threshold.data(), buffer, sizeof(uint32_t) * num_cat_threshold);
  }

  inline void Reset() {
    // initialize with -1 and -inf gain
    feature = -1;
    gain = kMinScore;
  }

  inline bool operator > (const SplitInfo& si) const {
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

  inline bool operator == (const SplitInfo& si) const {
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
};

struct LightSplitInfo {
 public:
  /*! \brief Feature index */
  int feature = -1;
  /*! \brief Split gain */
  double gain = kMinScore;
  /*! \brief Left number of data after split */
  data_size_t left_count = 0;
  /*! \brief Right number of data after split */
  data_size_t right_count = 0;

  inline void Reset() {
    // initialize with -1 and -inf gain
    feature = -1;
    gain = kMinScore;
  }

  void CopyFrom(const SplitInfo& other) {
    feature = other.feature;
    gain = other.gain;
    left_count = other.left_count;
    right_count = other.right_count;
  }

  void CopyFrom(const char* buffer) {
    std::memcpy(&feature, buffer, sizeof(feature));
    buffer += sizeof(feature);
    std::memcpy(&left_count, buffer, sizeof(left_count));
    buffer += sizeof(left_count);
    std::memcpy(&right_count, buffer, sizeof(right_count));
    buffer += sizeof(right_count);
    std::memcpy(&gain, buffer, sizeof(gain));
    buffer += sizeof(gain);
  }

  inline bool operator > (const LightSplitInfo& si) const {
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

  inline bool operator == (const LightSplitInfo& si) const {
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
};

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_SPLIT_INFO_HPP_
