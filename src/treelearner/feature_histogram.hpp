#ifndef LIGHTGBM_TREELEARNER_FEATURE_HISTOGRAM_HPP_
#define LIGHTGBM_TREELEARNER_FEATURE_HISTOGRAM_HPP_

#include "split_info.hpp"
#include <LightGBM/feature.h>

#include <cstring>

namespace LightGBM {

/*!
* \brief FeatureHistogram is used to construct and store a histogram for a feature.
*/
class FeatureHistogram {
public:
  FeatureHistogram()
    :data_(nullptr) {
  }
  ~FeatureHistogram() {
    if (data_ != nullptr) { delete[] data_; }
  }

  /*!
  * \brief Init the feature histogram
  * \param feature the feature data for this histogram
  * \param min_num_data_one_leaf minimal number of data in one leaf
  */
  void Init(const Feature* feature, int feature_idx, data_size_t min_num_data_one_leaf,
    score_t min_sum_hessian_one_leaf) {
    feature_idx_ = feature_idx;
    min_num_data_one_leaf_ = min_num_data_one_leaf;
    min_sum_hessian_one_leaf_ = min_sum_hessian_one_leaf;
    bin_data_ = feature->bin_data();
    num_bins_ = feature->num_bin();
    data_ = new HistogramBinEntry[num_bins_];
  }


  /*!
  * \brief Construct a histogram
  * \param num_data number of data in current leaf
  * \param sum_gradients sum of gradients of current leaf
  * \param sum_hessians sum of hissians of current leaf
  * \param ordered_gradients Orederd gradients
  * \param ordered_hessians  Ordered hessians
  * \param data_indices data indices of current leaf
  */
  void Construct(data_size_t* data_indices, data_size_t num_data, score_t sum_gradients,
                        score_t sum_hessians, const score_t* ordered_gradients, const score_t* ordered_hessians) {
    std::memset(data_, 0, sizeof(HistogramBinEntry)* num_bins_);
    num_data_ = num_data;
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians + 2 * kEpsilon;
    bin_data_->ConstructHistogram(data_indices, num_data, ordered_gradients, ordered_hessians, data_);
  }

  /*!
  * \brief Construct a histogram by ordered bin
  * \param leaf current leaf
  * \param num_data number of data in current leaf
  * \param sum_gradients sum of gradients of current leaf
  * \param sum_hessians sum of hissians of current leaf
  * \param gradients
  * \param hessian
  */
  void Construct(const OrderedBin* ordered_bin, int leaf, data_size_t num_data, score_t sum_gradients,
                        score_t sum_hessians, const score_t* gradients, const score_t* hessians) {
    std::memset(data_, 0, sizeof(HistogramBinEntry)* num_bins_);
    num_data_ = num_data;
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians + 2 * kEpsilon;
    ordered_bin->ConstructHistogram(leaf, gradients, hessians, data_);
  }

  /*!
  * \brief Set sumup information for current histogram
  * \param num_data number of data in current leaf
  * \param sum_gradients sum of gradients of current leaf
  * \param sum_hessians sum of hissians of current leaf
  */
  void SetSumup(data_size_t num_data, score_t sum_gradients, score_t sum_hessians) {
    num_data_ = num_data;
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians + 2 * kEpsilon;
  }

  /*!
  * \brief Subtract current histograms with other
  * \param other The histogram that want to subtract
  */
  void Subtract(const FeatureHistogram& other) {
    num_data_ -= other.num_data_;
    sum_gradients_ -= other.sum_gradients_;
    sum_hessians_ -= other.sum_hessians_;
    for (unsigned int i = 0; i < num_bins_; ++i) {
      data_[i].cnt -= other.data_[i].cnt;
      data_[i].sum_gradients -= other.data_[i].sum_gradients;
      data_[i].sum_hessians -= other.data_[i].sum_hessians;
    }
  }

  /*!
  * \brief Find best threshold for this histogram
  * \param output The best split result
  */
  void FindBestThreshold(SplitInfo* output) {
    score_t best_sum_left_gradient = NAN;
    score_t best_sum_left_hessian = NAN;
    score_t best_gain = kMinScore;
    data_size_t best_left_count = 0;
    unsigned int best_threshold = static_cast<unsigned int>(num_bins_);
    score_t sum_right_gradient = 0.0f;
    score_t sum_right_hessian = kEpsilon;
    data_size_t right_count = 0;
    score_t gain_shift = GetLeafSplitGain(sum_gradients_, sum_hessians_);
    is_splittable_ = false;
    // from right to left, and we don't need data in bin0
    for (unsigned int t = num_bins_ - 1; t > 0; --t) {
      sum_right_gradient += data_[t].sum_gradients;
      sum_right_hessian += data_[t].sum_hessians;
      right_count += data_[t].cnt;
      // if data not enough, or sum hessian too small
      if (right_count < min_num_data_one_leaf_ || sum_right_hessian < min_sum_hessian_one_leaf_) continue;
      data_size_t left_count = num_data_ - right_count;
      // if data not enough
      if (left_count < min_num_data_one_leaf_) break;

      score_t sum_left_hessian = sum_hessians_ - sum_right_hessian;
      // if sum hessian too small
      if (sum_left_hessian < min_sum_hessian_one_leaf_) {
        break;
      }
      score_t sum_left_gradient = sum_gradients_ - sum_right_gradient;
      // current split gain
      score_t current_gain = GetLeafSplitGain(sum_left_gradient, sum_left_hessian) + GetLeafSplitGain(sum_right_gradient, sum_right_hessian);
      // gain is worst than no perform split
      if (current_gain < gain_shift) {
        continue;
      }
      // mark to is splittable
      is_splittable_ = true;
      // better split point
      if (current_gain > best_gain) {
        best_left_count = left_count;
        best_sum_left_gradient = sum_left_gradient;
        best_sum_left_hessian = sum_left_hessian;
        // left is <= threshold, right is > threshold.  so this is t-1
        best_threshold = t - 1;
        best_gain = current_gain;
      }
    }
    // update split information
    output->feature = feature_idx_;
    output->threshold = best_threshold;
    output->left_output = CalculateSplittedLeafOutput(best_sum_left_gradient, best_sum_left_hessian);
    output->left_count = best_left_count;
    output->left_sum_gradient = best_sum_left_gradient;
    output->left_sum_hessian = best_sum_left_hessian;
    output->right_output = CalculateSplittedLeafOutput(sum_gradients_ - best_sum_left_gradient,
      sum_hessians_ - best_sum_left_hessian);
    output->right_count = num_data_ - best_left_count;
    output->right_sum_gradient = sum_gradients_ - best_sum_left_gradient;
    output->right_sum_hessian = sum_hessians_ - best_sum_left_hessian;
    output->gain = best_gain - gain_shift;
  }

  /*!
  * \brief Binary size of this histogram
  */
  int SizeOfHistgram() const {
    return num_bins_ * sizeof(HistogramBinEntry);
  }

  /*!
  * \brief Memory pointer to histogram data
  */
  const HistogramBinEntry* HistogramData() const {
    return data_;
  }

  /*!
  * \brief Restore histogram from memory
  */
  void FromMemory(char* memory_data)  {
    std::memcpy(data_, memory_data, num_bins_ * sizeof(HistogramBinEntry));
  }

  /*!
  * \brief Set min number data in one leaf
  */
  void SetMinNumDataOneLeaf(data_size_t new_val) {
    min_num_data_one_leaf_ = new_val;
  }

  /*!
  * \brief Set min sum hessian in one leaf
  */
  void SetMinSumHessianOneLeaf(score_t new_val) {
    min_sum_hessian_one_leaf_ = new_val;
  }

  /*!
  * \brief True if this histogram can be splitted
  */
  bool is_splittable() { return is_splittable_; }

  /*!
  * \brief Set splittable to this histogram
  */
  void set_is_splittable(bool val) { is_splittable_ = val; }

private:
  /*!
  * \brief Calculate the split gain based on sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  * \return split gain
  */
  score_t GetLeafSplitGain(score_t sum_gradients, score_t sum_hessians) const {
    return (sum_gradients * sum_gradients) / (sum_hessians);
  }

  /*!
  * \brief Calculate the output of a leaf based on sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  * \return leaf output
  */
  score_t CalculateSplittedLeafOutput(score_t sum_gradients, score_t sum_hessians) const {
    return -(sum_gradients) / (sum_hessians);
  }

  int feature_idx_;
  /*! \brief minimal number of data in one leaf */
  data_size_t min_num_data_one_leaf_;
  /*! \brief minimal sum hessian of data in one leaf */
  score_t min_sum_hessian_one_leaf_;
  /*! \brief the bin data of current feature */
  const Bin* bin_data_;
  /*! \brief number of bin of histogram */
  unsigned int num_bins_;
  /*! \brief sum of gradient of each bin */
  HistogramBinEntry* data_;
  /*! \brief number of all data */
  data_size_t num_data_;
  /*! \brief sum of gradient of current leaf */
  score_t sum_gradients_;
  /*! \brief sum of hessians of current leaf */
  score_t sum_hessians_;
  /*! \brief False if this histogram cannot split */
  bool is_splittable_ = true;
};

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_FEATURE_HISTOGRAM_HPP_
