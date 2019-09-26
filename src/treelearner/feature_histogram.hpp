/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_FEATURE_HISTOGRAM_HPP_
#define LIGHTGBM_TREELEARNER_FEATURE_HISTOGRAM_HPP_

#include <LightGBM/dataset.h>
#include <LightGBM/utils/array_args.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "split_info.hpp"

namespace LightGBM {

class FeatureMetainfo {
 public:
  int num_bin;
  MissingType missing_type;
  int8_t bias = 0;
  uint32_t default_bin;
  int8_t monotone_type;
  double penalty;
  /*! \brief pointer of tree config */
  const Config* config;
  BinType bin_type;
};
/*!
* \brief FeatureHistogram is used to construct and store a histogram for a feature.
*/
class FeatureHistogram {
 public:
  FeatureHistogram() {
    data_ = nullptr;
  }

  ~FeatureHistogram() {
  }

  /*! \brief Disable copy */
  FeatureHistogram& operator=(const FeatureHistogram&) = delete;
  /*! \brief Disable copy */
  FeatureHistogram(const FeatureHistogram&) = delete;

  /*!
  * \brief Init the feature histogram
  * \param feature the feature data for this histogram
  * \param min_num_data_one_leaf minimal number of data in one leaf
  */
  void Init(HistogramBinEntry* data, const FeatureMetainfo* meta) {
    meta_ = meta;
    data_ = data;
    if (meta_->bin_type == BinType::NumericalBin) {
      find_best_threshold_fun_ = std::bind(&FeatureHistogram::FindBestThresholdNumerical, this, std::placeholders::_1
                                           , std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
    } else {
      find_best_threshold_fun_ = std::bind(&FeatureHistogram::FindBestThresholdCategorical, this, std::placeholders::_1
                                           , std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
    }
  }

  HistogramBinEntry* RawData() {
    return data_;
  }
  /*!
  * \brief Subtract current histograms with other
  * \param other The histogram that want to subtract
  */
  void Subtract(const FeatureHistogram& other) {
    for (int i = 0; i < meta_->num_bin - meta_->bias; ++i) {
      data_[i].cnt -= other.data_[i].cnt;
      data_[i].sum_gradients -= other.data_[i].sum_gradients;
      data_[i].sum_hessians -= other.data_[i].sum_hessians;
    }
  }

  void FindBestThreshold(double sum_gradient, double sum_hessian, data_size_t num_data, double min_constraint, double max_constraint,
                         SplitInfo* output) {
    output->default_left = true;
    output->gain = kMinScore;
    find_best_threshold_fun_(sum_gradient, sum_hessian + 2 * kEpsilon, num_data, min_constraint, max_constraint, output);
    output->gain *= meta_->penalty;
  }

  void FindBestThresholdNumerical(double sum_gradient, double sum_hessian, data_size_t num_data, double min_constraint, double max_constraint,
                                  SplitInfo* output) {
    is_splittable_ = false;
    double gain_shift = GetLeafSplitGain(sum_gradient, sum_hessian,
                                         meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step);
    double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;
    if (meta_->num_bin > 2 && meta_->missing_type != MissingType::None) {
      if (meta_->missing_type == MissingType::Zero) {
        FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_constraint, max_constraint, min_gain_shift, output, -1, true, false);
        FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_constraint, max_constraint, min_gain_shift, output, 1, true, false);
      } else {
        FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_constraint, max_constraint, min_gain_shift, output, -1, false, true);
        FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_constraint, max_constraint, min_gain_shift, output, 1, false, true);
      }
    } else {
      FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_constraint, max_constraint, min_gain_shift, output, -1, false, false);
      // fix the direction error when only have 2 bins
      if (meta_->missing_type == MissingType::NaN) {
        output->default_left = false;
      }
    }
    output->gain -= min_gain_shift;
    output->monotone_type = meta_->monotone_type;
    output->min_constraint = min_constraint;
    output->max_constraint = max_constraint;
  }

  void FindBestThresholdCategorical(double sum_gradient, double sum_hessian, data_size_t num_data,
                                    double min_constraint, double max_constraint,
                                    SplitInfo* output) {
    output->default_left = false;
    double best_gain = kMinScore;
    data_size_t best_left_count = 0;
    double best_sum_left_gradient = 0;
    double best_sum_left_hessian = 0;
    double gain_shift = GetLeafSplitGain(sum_gradient, sum_hessian, meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step);

    double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;
    bool is_full_categorical = meta_->missing_type == MissingType::None;
    int used_bin = meta_->num_bin - 1 + is_full_categorical;

    std::vector<int> sorted_idx;
    double l2 = meta_->config->lambda_l2;
    bool use_onehot = meta_->num_bin <= meta_->config->max_cat_to_onehot;
    int best_threshold = -1;
    int best_dir = 1;

    if (use_onehot) {
      for (int t = 0; t < used_bin; ++t) {
        // if data not enough, or sum hessian too small
        if (data_[t].cnt < meta_->config->min_data_in_leaf
            || data_[t].sum_hessians < meta_->config->min_sum_hessian_in_leaf) continue;
        data_size_t other_count = num_data - data_[t].cnt;
        // if data not enough
        if (other_count < meta_->config->min_data_in_leaf) continue;

        double sum_other_hessian = sum_hessian - data_[t].sum_hessians - kEpsilon;
        // if sum hessian too small
        if (sum_other_hessian < meta_->config->min_sum_hessian_in_leaf) continue;

        double sum_other_gradient = sum_gradient - data_[t].sum_gradients;
        // current split gain
        double current_gain = GetSplitGains(sum_other_gradient, sum_other_hessian, data_[t].sum_gradients, data_[t].sum_hessians + kEpsilon,
                                            meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
                                            min_constraint, max_constraint, 0);
        // gain with split is worse than without split
        if (current_gain <= min_gain_shift) continue;

        // mark to is splittable
        is_splittable_ = true;
        // better split point
        if (current_gain > best_gain) {
          best_threshold = t;
          best_sum_left_gradient = data_[t].sum_gradients;
          best_sum_left_hessian = data_[t].sum_hessians + kEpsilon;
          best_left_count = data_[t].cnt;
          best_gain = current_gain;
        }
      }
    } else {
      for (int i = 0; i < used_bin; ++i) {
        if (data_[i].cnt >= meta_->config->cat_smooth) {
          sorted_idx.push_back(i);
        }
      }
      used_bin = static_cast<int>(sorted_idx.size());

      l2 += meta_->config->cat_l2;

      auto ctr_fun = [this](double sum_grad, double sum_hess) {
        return (sum_grad) / (sum_hess + meta_->config->cat_smooth);
      };
      std::sort(sorted_idx.begin(), sorted_idx.end(),
                [this, &ctr_fun](int i, int j) {
        return ctr_fun(data_[i].sum_gradients, data_[i].sum_hessians) < ctr_fun(data_[j].sum_gradients, data_[j].sum_hessians);
      });

      std::vector<int> find_direction(1, 1);
      std::vector<int> start_position(1, 0);
      find_direction.push_back(-1);
      start_position.push_back(used_bin - 1);
      const int max_num_cat = std::min(meta_->config->max_cat_threshold, (used_bin + 1) / 2);

      is_splittable_ = false;
      for (size_t out_i = 0; out_i < find_direction.size(); ++out_i) {
        auto dir = find_direction[out_i];
        auto start_pos = start_position[out_i];
        data_size_t min_data_per_group = meta_->config->min_data_per_group;
        data_size_t cnt_cur_group = 0;
        double sum_left_gradient = 0.0f;
        double sum_left_hessian = kEpsilon;
        data_size_t left_count = 0;
        for (int i = 0; i < used_bin && i < max_num_cat; ++i) {
          auto t = sorted_idx[start_pos];
          start_pos += dir;

          sum_left_gradient += data_[t].sum_gradients;
          sum_left_hessian += data_[t].sum_hessians;
          left_count += data_[t].cnt;
          cnt_cur_group += data_[t].cnt;

          if (left_count < meta_->config->min_data_in_leaf
              || sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) continue;
          data_size_t right_count = num_data - left_count;
          if (right_count < meta_->config->min_data_in_leaf || right_count < min_data_per_group) break;

          double sum_right_hessian = sum_hessian - sum_left_hessian;
          if (sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) break;

          if (cnt_cur_group < min_data_per_group) continue;

          cnt_cur_group = 0;

          double sum_right_gradient = sum_gradient - sum_left_gradient;
          double current_gain = GetSplitGains(sum_left_gradient, sum_left_hessian, sum_right_gradient, sum_right_hessian,
                                              meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
                                              min_constraint, max_constraint, 0);
          if (current_gain <= min_gain_shift) continue;
          is_splittable_ = true;
          if (current_gain > best_gain) {
            best_left_count = left_count;
            best_sum_left_gradient = sum_left_gradient;
            best_sum_left_hessian = sum_left_hessian;
            best_threshold = i;
            best_gain = current_gain;
            best_dir = dir;
          }
        }
      }
    }

    if (is_splittable_) {
      output->left_output = CalculateSplittedLeafOutput(best_sum_left_gradient, best_sum_left_hessian,
                                                        meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
                                                        min_constraint, max_constraint);
      output->left_count = best_left_count;
      output->left_sum_gradient = best_sum_left_gradient;
      output->left_sum_hessian = best_sum_left_hessian - kEpsilon;
      output->right_output = CalculateSplittedLeafOutput(sum_gradient - best_sum_left_gradient,
                                                         sum_hessian - best_sum_left_hessian,
                                                         meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
                                                         min_constraint, max_constraint);
      output->right_count = num_data - best_left_count;
      output->right_sum_gradient = sum_gradient - best_sum_left_gradient;
      output->right_sum_hessian = sum_hessian - best_sum_left_hessian - kEpsilon;
      output->gain = best_gain - min_gain_shift;
      if (use_onehot) {
        output->num_cat_threshold = 1;
        output->cat_threshold = std::vector<uint32_t>(1, static_cast<uint32_t>(best_threshold));
      } else {
        output->num_cat_threshold = best_threshold + 1;
        output->cat_threshold = std::vector<uint32_t>(output->num_cat_threshold);
        if (best_dir == 1) {
          for (int i = 0; i < output->num_cat_threshold; ++i) {
            auto t = sorted_idx[i];
            output->cat_threshold[i] = t;
          }
        } else {
          for (int i = 0; i < output->num_cat_threshold; ++i) {
            auto t = sorted_idx[used_bin - 1 - i];
            output->cat_threshold[i] = t;
          }
        }
      }
      output->monotone_type = 0;
      output->min_constraint = min_constraint;
      output->max_constraint = max_constraint;
    }
  }

  void GatherInfoForThreshold(double sum_gradient, double sum_hessian,
                              uint32_t threshold, data_size_t num_data, SplitInfo *output) {
    if (meta_->bin_type == BinType::NumericalBin) {
      GatherInfoForThresholdNumerical(sum_gradient, sum_hessian, threshold,
                                      num_data, output);
    } else {
      GatherInfoForThresholdCategorical(sum_gradient, sum_hessian, threshold,
                                        num_data, output);
    }
  }

  void GatherInfoForThresholdNumerical(double sum_gradient, double sum_hessian,
                                       uint32_t threshold, data_size_t num_data,
                                       SplitInfo *output) {
    double gain_shift = GetLeafSplitGain(sum_gradient, sum_hessian,
                                         meta_->config->lambda_l1, meta_->config->lambda_l2,
                                         meta_->config->max_delta_step);
    double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;

    // do stuff here
    const int8_t bias = meta_->bias;

    double sum_right_gradient = 0.0f;
    double sum_right_hessian = kEpsilon;
    data_size_t right_count = 0;

    // set values
    bool use_na_as_missing = false;
    bool skip_default_bin = false;
    if (meta_->missing_type == MissingType::Zero) {
      skip_default_bin = true;
    } else if (meta_->missing_type == MissingType::NaN) {
      use_na_as_missing = true;
    }

    int t = meta_->num_bin - 1 - bias - use_na_as_missing;
    const int t_end = 1 - bias;

    // from right to left, and we don't need data in bin0
    for (; t >= t_end; --t) {
      if (static_cast<uint32_t>(t + bias) < threshold) { break; }

      // need to skip default bin
      if (skip_default_bin && (t + bias) == static_cast<int>(meta_->default_bin)) { continue; }

      sum_right_gradient += data_[t].sum_gradients;
      sum_right_hessian += data_[t].sum_hessians;
      right_count += data_[t].cnt;
    }
    double sum_left_gradient = sum_gradient - sum_right_gradient;
    double sum_left_hessian = sum_hessian - sum_right_hessian;
    data_size_t left_count = num_data - right_count;
    double current_gain = GetLeafSplitGain(sum_left_gradient, sum_left_hessian,
                                           meta_->config->lambda_l1, meta_->config->lambda_l2,
                                           meta_->config->max_delta_step)
          + GetLeafSplitGain(sum_right_gradient, sum_right_hessian,
                             meta_->config->lambda_l1, meta_->config->lambda_l2,
                             meta_->config->max_delta_step);

    // gain with split is worse than without split
    if (std::isnan(current_gain) || current_gain <= min_gain_shift) {
      output->gain = kMinScore;
      Log::Warning("'Forced Split' will be ignored since the gain getting worse. ");
      return;
    }

    // update split information
    output->threshold = threshold;
    output->left_output = CalculateSplittedLeafOutput(sum_left_gradient, sum_left_hessian,
                                                      meta_->config->lambda_l1, meta_->config->lambda_l2,
                                                      meta_->config->max_delta_step);
    output->left_count = left_count;
    output->left_sum_gradient = sum_left_gradient;
    output->left_sum_hessian = sum_left_hessian - kEpsilon;
    output->right_output = CalculateSplittedLeafOutput(sum_gradient - sum_left_gradient,
                                                       sum_hessian - sum_left_hessian,
                                                       meta_->config->lambda_l1, meta_->config->lambda_l2,
                                                       meta_->config->max_delta_step);
    output->right_count = num_data - left_count;
    output->right_sum_gradient = sum_gradient - sum_left_gradient;
    output->right_sum_hessian = sum_hessian - sum_left_hessian - kEpsilon;
    output->gain = current_gain;
    output->gain -= min_gain_shift;
    output->default_left = true;
  }

  void GatherInfoForThresholdCategorical(double sum_gradient, double sum_hessian,
                                         uint32_t threshold, data_size_t num_data, SplitInfo *output) {
    // get SplitInfo for a given one-hot categorical split.
    output->default_left = false;
    double gain_shift = GetLeafSplitGain(
            sum_gradient, sum_hessian,
            meta_->config->lambda_l1, meta_->config->lambda_l2,
            meta_->config->max_delta_step);
    double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;
    bool is_full_categorical = meta_->missing_type == MissingType::None;
    int used_bin = meta_->num_bin - 1 + is_full_categorical;
    if (threshold >= static_cast<uint32_t>(used_bin)) {
      output->gain = kMinScore;
      Log::Warning("Invalid categorical threshold split");
      return;
    }

    double l2 = meta_->config->lambda_l2;
    data_size_t left_count = data_[threshold].cnt;
    data_size_t right_count = num_data - left_count;
    double sum_left_hessian = data_[threshold].sum_hessians + kEpsilon;
    double sum_right_hessian = sum_hessian - sum_left_hessian;
    double sum_left_gradient = data_[threshold].sum_gradients;
    double sum_right_gradient = sum_gradient - sum_left_gradient;
    // current split gain
    double current_gain = GetLeafSplitGain(sum_right_gradient, sum_right_hessian,
                                           meta_->config->lambda_l1, l2,
                                           meta_->config->max_delta_step)
        + GetLeafSplitGain(sum_left_gradient, sum_left_hessian,
                           meta_->config->lambda_l1, l2,
                           meta_->config->max_delta_step);
    if (std::isnan(current_gain) || current_gain <= min_gain_shift) {
      output->gain = kMinScore;
      Log::Warning("'Forced Split' will be ignored since the gain getting worse. ");
      return;
    }

    output->left_output = CalculateSplittedLeafOutput(sum_left_gradient, sum_left_hessian,
                                                      meta_->config->lambda_l1, l2,
                                                      meta_->config->max_delta_step);
    output->left_count = left_count;
    output->left_sum_gradient = sum_left_gradient;
    output->left_sum_hessian = sum_left_hessian - kEpsilon;
    output->right_output = CalculateSplittedLeafOutput(sum_right_gradient, sum_right_hessian,
                                                       meta_->config->lambda_l1, l2,
                                                       meta_->config->max_delta_step);
    output->right_count = right_count;
    output->right_sum_gradient = sum_gradient - sum_left_gradient;
    output->right_sum_hessian = sum_right_hessian - kEpsilon;
    output->gain = current_gain - min_gain_shift;
    output->num_cat_threshold = 1;
    output->cat_threshold = std::vector<uint32_t>(1, threshold);
  }


  /*!
  * \brief Binary size of this histogram
  */
  int SizeOfHistgram() const {
    return (meta_->num_bin - meta_->bias) * sizeof(HistogramBinEntry);
  }

  /*!
  * \brief Restore histogram from memory
  */
  void FromMemory(char* memory_data) {
    std::memcpy(data_, memory_data, (meta_->num_bin - meta_->bias) * sizeof(HistogramBinEntry));
  }

  /*!
  * \brief True if this histogram can be splitted
  */
  bool is_splittable() { return is_splittable_; }

  /*!
  * \brief Set splittable to this histogram
  */
  void set_is_splittable(bool val) { is_splittable_ = val; }

  static double ThresholdL1(double s, double l1) {
    const double reg_s = std::max(0.0, std::fabs(s) - l1);
    return Common::Sign(s) * reg_s;
  }

  static double CalculateSplittedLeafOutput(double sum_gradients, double sum_hessians, double l1, double l2, double max_delta_step) {
    double ret = -ThresholdL1(sum_gradients, l1) / (sum_hessians + l2);
    if (max_delta_step <= 0.0f || std::fabs(ret) <= max_delta_step) {
      return ret;
    } else {
      return Common::Sign(ret) * max_delta_step;
    }
  }

 private:
  static double GetSplitGains(double sum_left_gradients, double sum_left_hessians,
                              double sum_right_gradients, double sum_right_hessians,
                              double l1, double l2, double max_delta_step,
                              double min_constraint, double max_constraint, int8_t monotone_constraint) {
    double left_output = CalculateSplittedLeafOutput(sum_left_gradients, sum_left_hessians, l1, l2, max_delta_step, min_constraint, max_constraint);
    double right_output = CalculateSplittedLeafOutput(sum_right_gradients, sum_right_hessians, l1, l2, max_delta_step, min_constraint, max_constraint);
    if (((monotone_constraint > 0) && (left_output > right_output)) ||
      ((monotone_constraint < 0) && (left_output < right_output))) {
      return 0;
    }
    return GetLeafSplitGainGivenOutput(sum_left_gradients, sum_left_hessians, l1, l2, left_output)
      + GetLeafSplitGainGivenOutput(sum_right_gradients, sum_right_hessians, l1, l2, right_output);
  }

  /*!
  * \brief Calculate the output of a leaf based on regularized sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  * \return leaf output
  */
  static double CalculateSplittedLeafOutput(double sum_gradients, double sum_hessians, double l1, double l2, double max_delta_step,
                                            double min_constraint, double max_constraint) {
    double ret = CalculateSplittedLeafOutput(sum_gradients, sum_hessians, l1, l2, max_delta_step);
    if (ret < min_constraint) {
      ret = min_constraint;
    } else if (ret > max_constraint) {
      ret = max_constraint;
    }
    return ret;
  }

  /*!
  * \brief Calculate the split gain based on regularized sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  * \return split gain
  */
  static double GetLeafSplitGain(double sum_gradients, double sum_hessians, double l1, double l2, double max_delta_step) {
    double output = CalculateSplittedLeafOutput(sum_gradients, sum_hessians, l1, l2, max_delta_step);
    return GetLeafSplitGainGivenOutput(sum_gradients, sum_hessians, l1, l2, output);
  }

  static double GetLeafSplitGainGivenOutput(double sum_gradients, double sum_hessians, double l1, double l2, double output) {
    const double sg_l1 = ThresholdL1(sum_gradients, l1);
    return -(2.0 * sg_l1 * output + (sum_hessians + l2) * output * output);
  }

  void FindBestThresholdSequence(double sum_gradient, double sum_hessian, data_size_t num_data, double min_constraint, double max_constraint,
                                 double min_gain_shift, SplitInfo* output, int dir, bool skip_default_bin, bool use_na_as_missing) {
    const int8_t bias = meta_->bias;

    double best_sum_left_gradient = NAN;
    double best_sum_left_hessian = NAN;
    double best_gain = kMinScore;
    data_size_t best_left_count = 0;
    uint32_t best_threshold = static_cast<uint32_t>(meta_->num_bin);

    if (dir == -1) {
      double sum_right_gradient = 0.0f;
      double sum_right_hessian = kEpsilon;
      data_size_t right_count = 0;

      int t = meta_->num_bin - 1 - bias - use_na_as_missing;
      const int t_end = 1 - bias;

      // from right to left, and we don't need data in bin0
      for (; t >= t_end; --t) {
        // need to skip default bin
        if (skip_default_bin && (t + bias) == static_cast<int>(meta_->default_bin)) { continue; }

        sum_right_gradient += data_[t].sum_gradients;
        sum_right_hessian += data_[t].sum_hessians;
        right_count += data_[t].cnt;
        // if data not enough, or sum hessian too small
        if (right_count < meta_->config->min_data_in_leaf
            || sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) continue;
        data_size_t left_count = num_data - right_count;
        // if data not enough
        if (left_count < meta_->config->min_data_in_leaf) break;

        double sum_left_hessian = sum_hessian - sum_right_hessian;
        // if sum hessian too small
        if (sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) break;

        double sum_left_gradient = sum_gradient - sum_right_gradient;
        // current split gain
        double current_gain = GetSplitGains(sum_left_gradient, sum_left_hessian, sum_right_gradient, sum_right_hessian,
                                            meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step,
                                            min_constraint, max_constraint, meta_->monotone_type);
        // gain with split is worse than without split
        if (current_gain <= min_gain_shift) continue;

        // mark to is splittable
        is_splittable_ = true;
        // better split point
        if (current_gain > best_gain) {
          best_left_count = left_count;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          // left is <= threshold, right is > threshold.  so this is t-1
          best_threshold = static_cast<uint32_t>(t - 1 + bias);
          best_gain = current_gain;
        }
      }
    } else {
      double sum_left_gradient = 0.0f;
      double sum_left_hessian = kEpsilon;
      data_size_t left_count = 0;

      int t = 0;
      const int t_end = meta_->num_bin - 2 - bias;

      if (use_na_as_missing && bias == 1) {
        sum_left_gradient = sum_gradient;
        sum_left_hessian = sum_hessian - kEpsilon;
        left_count = num_data;
        for (int i = 0; i < meta_->num_bin - bias; ++i) {
          sum_left_gradient -= data_[i].sum_gradients;
          sum_left_hessian -= data_[i].sum_hessians;
          left_count -= data_[i].cnt;
        }
        t = -1;
      }

      for (; t <= t_end; ++t) {
        // need to skip default bin
        if (skip_default_bin && (t + bias) == static_cast<int>(meta_->default_bin)) { continue; }
        if (t >= 0) {
          sum_left_gradient += data_[t].sum_gradients;
          sum_left_hessian += data_[t].sum_hessians;
          left_count += data_[t].cnt;
        }
        // if data not enough, or sum hessian too small
        if (left_count < meta_->config->min_data_in_leaf
            || sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) continue;
        data_size_t right_count = num_data - left_count;
        // if data not enough
        if (right_count < meta_->config->min_data_in_leaf) break;

        double sum_right_hessian = sum_hessian - sum_left_hessian;
        // if sum hessian too small
        if (sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) break;

        double sum_right_gradient = sum_gradient - sum_left_gradient;
        // current split gain
        double current_gain = GetSplitGains(sum_left_gradient, sum_left_hessian, sum_right_gradient, sum_right_hessian,
                                            meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step,
                                            min_constraint, max_constraint, meta_->monotone_type);
        // gain with split is worse than without split
        if (current_gain <= min_gain_shift) continue;

        // mark to is splittable
        is_splittable_ = true;
        // better split point
        if (current_gain > best_gain) {
          best_left_count = left_count;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          best_threshold = static_cast<uint32_t>(t + bias);
          best_gain = current_gain;
        }
      }
    }

    if (is_splittable_ && best_gain > output->gain) {
      // update split information
      output->threshold = best_threshold;
      output->left_output = CalculateSplittedLeafOutput(best_sum_left_gradient, best_sum_left_hessian,
                                                        meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step,
                                                        min_constraint, max_constraint);
      output->left_count = best_left_count;
      output->left_sum_gradient = best_sum_left_gradient;
      output->left_sum_hessian = best_sum_left_hessian - kEpsilon;
      output->right_output = CalculateSplittedLeafOutput(sum_gradient - best_sum_left_gradient,
                                                         sum_hessian - best_sum_left_hessian,
                                                         meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step,
                                                         min_constraint, max_constraint);
      output->right_count = num_data - best_left_count;
      output->right_sum_gradient = sum_gradient - best_sum_left_gradient;
      output->right_sum_hessian = sum_hessian - best_sum_left_hessian - kEpsilon;
      output->gain = best_gain;
      output->default_left = dir == -1;
    }
  }

  const FeatureMetainfo* meta_;
  /*! \brief sum of gradient of each bin */
  HistogramBinEntry* data_;
  // std::vector<HistogramBinEntry> data_;
  bool is_splittable_ = true;

  std::function<void(double, double, data_size_t, double, double, SplitInfo*)> find_best_threshold_fun_;
};
class HistogramPool {
 public:
  /*!
  * \brief Constructor
  */
  HistogramPool() {
    cache_size_ = 0;
    total_size_ = 0;
  }
  /*!
  * \brief Destructor
  */
  ~HistogramPool() {
  }
  /*!
  * \brief Reset pool size
  * \param cache_size Max cache size
  * \param total_size Total size will be used
  */
  void Reset(int cache_size, int total_size) {
    cache_size_ = cache_size;
    // at least need 2 bucket to store smaller leaf and larger leaf
    CHECK(cache_size_ >= 2);
    total_size_ = total_size;
    if (cache_size_ > total_size_) {
      cache_size_ = total_size_;
    }
    is_enough_ = (cache_size_ == total_size_);
    if (!is_enough_) {
      mapper_.resize(total_size_);
      inverse_mapper_.resize(cache_size_);
      last_used_time_.resize(cache_size_);
      ResetMap();
    }
  }
  /*!
  * \brief Reset mapper
  */
  void ResetMap() {
    if (!is_enough_) {
      cur_time_ = 0;
      std::fill(mapper_.begin(), mapper_.end(), -1);
      std::fill(inverse_mapper_.begin(), inverse_mapper_.end(), -1);
      std::fill(last_used_time_.begin(), last_used_time_.end(), 0);
    }
  }

  void DynamicChangeSize(const Dataset* train_data, const Config* config, int cache_size, int total_size) {
    if (feature_metas_.empty()) {
      int num_feature = train_data->num_features();
      feature_metas_.resize(num_feature);
      #pragma omp parallel for schedule(static, 512) if (num_feature >= 1024)
      for (int i = 0; i < num_feature; ++i) {
        feature_metas_[i].num_bin = train_data->FeatureNumBin(i);
        feature_metas_[i].default_bin = train_data->FeatureBinMapper(i)->GetDefaultBin();
        feature_metas_[i].missing_type = train_data->FeatureBinMapper(i)->missing_type();
        feature_metas_[i].monotone_type = train_data->FeatureMonotone(i);
        feature_metas_[i].penalty = train_data->FeaturePenalte(i);
        if (train_data->FeatureBinMapper(i)->GetDefaultBin() == 0) {
          feature_metas_[i].bias = 1;
        } else {
          feature_metas_[i].bias = 0;
        }
        feature_metas_[i].config = config;
        feature_metas_[i].bin_type = train_data->FeatureBinMapper(i)->bin_type();
      }
    }
    uint64_t num_total_bin = train_data->NumTotalBin();
    Log::Info("Total Bins %d", num_total_bin);
    int old_cache_size = static_cast<int>(pool_.size());
    Reset(cache_size, total_size);

    if (cache_size > old_cache_size) {
      pool_.resize(cache_size);
      data_.resize(cache_size);
    }

    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int i = old_cache_size; i < cache_size; ++i) {
      OMP_LOOP_EX_BEGIN();
      pool_[i].reset(new FeatureHistogram[train_data->num_features()]);
      data_[i].resize(num_total_bin);
      uint64_t offset = 0;
      for (int j = 0; j < train_data->num_features(); ++j) {
        offset += static_cast<uint64_t>(train_data->SubFeatureBinOffset(j));
        pool_[i][j].Init(data_[i].data() + offset, &feature_metas_[j]);
        auto num_bin = train_data->FeatureNumBin(j);
        if (train_data->FeatureBinMapper(j)->GetDefaultBin() == 0) {
          num_bin -= 1;
        }
        offset += static_cast<uint64_t>(num_bin);
      }
      CHECK(offset == num_total_bin);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    train_data_ = train_data;
  }

  void ResetConfig(const Config* config) {
    int size = static_cast<int>(feature_metas_.size());
    #pragma omp parallel for schedule(static, 512) if (size >= 1024)
    for (int i = 0; i < size; ++i) {
      feature_metas_[i].config = config;
      feature_metas_[i].monotone_type = train_data_->FeatureMonotone(i);
      feature_metas_[i].penalty = train_data_->FeaturePenalte(i);
    }
  }
  /*!
  * \brief Get data for the specific index
  * \param idx which index want to get
  * \param out output data will store into this
  * \return True if this index is in the pool, False if this index is not in the pool
  */
  bool Get(int idx, FeatureHistogram** out) {
    if (is_enough_) {
      *out = pool_[idx].get();
      return true;
    } else if (mapper_[idx] >= 0) {
      int slot = mapper_[idx];
      *out = pool_[slot].get();
      last_used_time_[slot] = ++cur_time_;
      return true;
    } else {
      // choose the least used slot
      int slot = static_cast<int>(ArrayArgs<int>::ArgMin(last_used_time_));
      *out = pool_[slot].get();
      last_used_time_[slot] = ++cur_time_;

      // reset previous mapper
      if (inverse_mapper_[slot] >= 0) mapper_[inverse_mapper_[slot]] = -1;

      // update current mapper
      mapper_[idx] = slot;
      inverse_mapper_[slot] = idx;
      return false;
    }
  }

  /*!
  * \brief Move data from one index to another index
  * \param src_idx
  * \param dst_idx
  */
  void Move(int src_idx, int dst_idx) {
    if (is_enough_) {
      std::swap(pool_[src_idx], pool_[dst_idx]);
      return;
    }
    if (mapper_[src_idx] < 0) {
      return;
    }
    // get slot of src idx
    int slot = mapper_[src_idx];
    // reset src_idx
    mapper_[src_idx] = -1;

    // move to dst idx
    mapper_[dst_idx] = slot;
    last_used_time_[slot] = ++cur_time_;
    inverse_mapper_[slot] = dst_idx;
  }

 private:
  std::vector<std::unique_ptr<FeatureHistogram[]>> pool_;
  std::vector<std::vector<HistogramBinEntry>> data_;
  std::vector<FeatureMetainfo> feature_metas_;
  int cache_size_;
  int total_size_;
  bool is_enough_ = false;
  std::vector<int> mapper_;
  std::vector<int> inverse_mapper_;
  std::vector<int> last_used_time_;
  int cur_time_ = 0;
  const Dataset* train_data_;
};

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_FEATURE_HISTOGRAM_HPP_
