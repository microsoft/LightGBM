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
  FeatureHistogram() {
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
  void Init(const Feature* feature, int feature_idx, const TreeConfig* tree_config) {
    feature_idx_ = feature_idx;
    tree_config_ = tree_config;
    bin_data_ = feature->bin_data();
    num_bins_ = feature->num_bin();
    data_.resize(num_bins_);
    if (feature->bin_type() == BinType::NumericalBin) {
      find_best_threshold_fun_ = std::bind(&FeatureHistogram::FindBestThresholdForNumerical, this, std::placeholders::_1);
    } else {
      find_best_threshold_fun_ = std::bind(&FeatureHistogram::FindBestThresholdForCategorical, this, std::placeholders::_1);
    }
  }


  /*!
  * \brief Construct a histogram
  * \param num_data number of data in current leaf
  * \param sum_gradients sum of gradients of current leaf
  * \param sum_hessians sum of hessians of current leaf
  * \param ordered_gradients Orederd gradients
  * \param ordered_hessians  Ordered hessians
  * \param data_indices data indices of current leaf
  */
  void Construct(const data_size_t* data_indices, data_size_t num_data, double sum_gradients,
    double sum_hessians, const score_t* ordered_gradients, const score_t* ordered_hessians) {
    std::memset(data_.data(), 0, sizeof(HistogramBinEntry)* num_bins_);
    num_data_ = num_data;
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians + 2 * kEpsilon;
    bin_data_->ConstructHistogram(data_indices, num_data, ordered_gradients, ordered_hessians, data_.data());
  }

  /*!
  * \brief Construct a histogram by ordered bin
  * \param leaf current leaf
  * \param num_data number of data in current leaf
  * \param sum_gradients sum of gradients of current leaf
  * \param sum_hessians sum of hessians of current leaf
  * \param gradients
  * \param hessian
  */
  void Construct(const OrderedBin* ordered_bin, int leaf, data_size_t num_data, double sum_gradients,
    double sum_hessians, const score_t* gradients, const score_t* hessians) {
    std::memset(data_.data(), 0, sizeof(HistogramBinEntry)* num_bins_);
    num_data_ = num_data;
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians + 2 * kEpsilon;
    ordered_bin->ConstructHistogram(leaf, gradients, hessians, data_.data());
  }

  /*!
  * \brief Set sumup information for current histogram
  * \param num_data number of data in current leaf
  * \param sum_gradients sum of gradients of current leaf
  * \param sum_hessians sum of hessians of current leaf
  */
  void SetSumup(data_size_t num_data, double sum_gradients, double sum_hessians) {
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
    find_best_threshold_fun_(output);
  }

  void FindBestThresholdForNumerical(SplitInfo* output) {
    double best_sum_left_gradient = NAN;
    double best_sum_left_hessian = NAN;
    double best_gain = kMinScore;
    data_size_t best_left_count = 0;
    unsigned int best_threshold = static_cast<unsigned int>(num_bins_);
    double sum_right_gradient = 0.0f;
    double sum_right_hessian = kEpsilon;
    data_size_t right_count = 0;
    double gain_shift = GetLeafSplitGain(sum_gradients_, sum_hessians_);
    double min_gain_shift = gain_shift + tree_config_->min_gain_to_split;
    is_splittable_ = false;
    // from right to left, and we don't need data in bin0
    for (unsigned int t = num_bins_ - 1; t > 0; --t) {
      sum_right_gradient += data_[t].sum_gradients;
      sum_right_hessian += data_[t].sum_hessians;
      right_count += data_[t].cnt;
      // if data not enough, or sum hessian too small
      if (right_count < tree_config_->min_data_in_leaf 
          || sum_right_hessian < tree_config_->min_sum_hessian_in_leaf) continue;
      data_size_t left_count = num_data_ - right_count;
      // if data not enough
      if (left_count < tree_config_->min_data_in_leaf) break;

      double sum_left_hessian = sum_hessians_ - sum_right_hessian;
      // if sum hessian too small
      if (sum_left_hessian < tree_config_->min_sum_hessian_in_leaf) break;

      double sum_left_gradient = sum_gradients_ - sum_right_gradient;
      // current split gain
      double current_gain = GetLeafSplitGain(sum_left_gradient, sum_left_hessian)
        + GetLeafSplitGain(sum_right_gradient, sum_right_hessian);
      // gain with split is worse than without split
      if (current_gain < min_gain_shift) continue;

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
    if (is_splittable_) {
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
    } else {
      output->feature = feature_idx_;
      output->gain = kMinScore;
    }
  }

  /*!
  * \brief Find best threshold for this histogram
  * \param output The best split result
  */
  void FindBestThresholdForCategorical(SplitInfo* output) {
    double best_gain = kMinScore;
    unsigned int best_threshold = static_cast<unsigned int>(num_bins_);

    double gain_shift = GetLeafSplitGain(sum_gradients_, sum_hessians_);
    double min_gain_shift = gain_shift + tree_config_->min_gain_to_split;
    is_splittable_ = false;

    for (int t = num_bins_ - 1; t >= 0; --t) {
      double sum_current_gradient = data_[t].sum_gradients;
      double sum_current_hessian = data_[t].sum_hessians;
      data_size_t current_count = data_[t].cnt;
      // if data not enough, or sum hessian too small
      if (current_count < tree_config_->min_data_in_leaf
          || sum_current_hessian < tree_config_->min_sum_hessian_in_leaf) continue;
      data_size_t other_count = num_data_ - current_count;
      // if data not enough
      if (other_count < tree_config_->min_data_in_leaf) continue;

      double sum_other_hessian = sum_hessians_ - sum_current_hessian;
      // if sum hessian too small
      if (sum_other_hessian < tree_config_->min_sum_hessian_in_leaf) continue;

      double sum_other_gradient = sum_gradients_ - sum_current_gradient;
      // current split gain
      double current_gain = GetLeafSplitGain(sum_other_gradient, sum_other_hessian) 
        + GetLeafSplitGain(sum_current_gradient, sum_current_hessian);
      // gain with split is worse than without split
      if (current_gain < min_gain_shift) continue;

      // mark to is splittable
      is_splittable_ = true;
      // better split point
      if (current_gain > best_gain) {
        best_threshold = static_cast<unsigned int>(t);
        best_gain = current_gain;
      }
    }
    // update split information
    if (is_splittable_) {
      output->feature = feature_idx_;
      output->threshold = best_threshold;
      output->left_output = CalculateSplittedLeafOutput(data_[best_threshold].sum_gradients,
        data_[best_threshold].sum_hessians);
      output->left_count = data_[best_threshold].cnt;
      output->left_sum_gradient = data_[best_threshold].sum_gradients;
      output->left_sum_hessian = data_[best_threshold].sum_hessians;

      output->right_output = CalculateSplittedLeafOutput(sum_gradients_ - data_[best_threshold].sum_gradients,
        sum_hessians_ - data_[best_threshold].sum_hessians);
      output->right_count = num_data_ - data_[best_threshold].cnt;
      output->right_sum_gradient = sum_gradients_ - data_[best_threshold].sum_gradients;
      output->right_sum_hessian = sum_hessians_ - data_[best_threshold].sum_hessians;

      output->gain = best_gain - gain_shift;
    } else {
      output->feature = feature_idx_;
      output->gain = kMinScore;
    }
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
    return data_.data();
  }

  /*!
  * \brief Restore histogram from memory
  */
  void FromMemory(char* memory_data)  {
    std::memcpy(data_.data(), memory_data, num_bins_ * sizeof(HistogramBinEntry));
  }

  /*!
  * \brief True if this histogram can be splitted
  */
  bool is_splittable() { return is_splittable_; }

  /*!
  * \brief Set splittable to this histogram
  */
  void set_is_splittable(bool val) { is_splittable_ = val; }

  void ResetConfig(const TreeConfig* tree_config) {
    tree_config_ = tree_config;
  }

private:
  /*!
  * \brief Calculate the split gain based on regularized sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  * \return split gain
  */
  double GetLeafSplitGain(double sum_gradients, double sum_hessians) const {
    double abs_sum_gradients = std::fabs(sum_gradients);
    if (abs_sum_gradients > tree_config_->lambda_l1) {
      double reg_abs_sum_gradients = abs_sum_gradients - tree_config_->lambda_l1;
      return (reg_abs_sum_gradients * reg_abs_sum_gradients) 
             / (sum_hessians + tree_config_->lambda_l2);
    }
    return 0.0f;
  }

  /*!
  * \brief Calculate the output of a leaf based on regularized sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  * \return leaf output
  */
  double CalculateSplittedLeafOutput(double sum_gradients, double sum_hessians) const {
    double abs_sum_gradients = std::fabs(sum_gradients);
    if (abs_sum_gradients > tree_config_->lambda_l1) {
      return -std::copysign(abs_sum_gradients - tree_config_->lambda_l1, sum_gradients) 
                            / (sum_hessians + tree_config_->lambda_l2);
    }
    return 0.0f;
  }

  int feature_idx_;
  /*! \brief pointer of tree config */
  const TreeConfig* tree_config_;
  /*! \brief the bin data of current feature */
  const Bin* bin_data_;
  /*! \brief number of bin of histogram */
  unsigned int num_bins_;
  /*! \brief sum of gradient of each bin */
  std::vector<HistogramBinEntry> data_;
  /*! \brief number of all data */
  data_size_t num_data_;
  /*! \brief sum of gradient of current leaf */
  double sum_gradients_;
  /*! \brief sum of hessians of current leaf */
  double sum_hessians_;
  /*! \brief False if this histogram cannot split */
  bool is_splittable_ = true;
  /*! \brief function that used to find best threshold */
  std::function<void(SplitInfo*)> find_best_threshold_fun_;
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

  /*!
  * \brief Fill the pool
  * \param obj_create_fun that used to generate object
  */
  void Fill(std::function<FeatureHistogram*()> obj_create_fun) {
    fill_func_ = obj_create_fun;
    pool_.clear();
    pool_.resize(cache_size_);
    for (int i = 0; i < cache_size_; ++i) {
      pool_[i].reset(obj_create_fun());
    }
  }

  void DynamicChangeSize(int cache_size, int total_size) {
    int old_cache_size = cache_size_;
    Reset(cache_size, total_size);
    pool_.resize(cache_size_);
    for (int i = old_cache_size; i < cache_size_; ++i) {
      pool_[i].reset(fill_func_());
    }
  }

  void ResetConfig(const TreeConfig* tree_config, int array_size) {
    for (int i = 0; i < cache_size_; ++i) {
      auto data_ptr = pool_[i].get();
      for (int j = 0; j < array_size; ++j) {
        data_ptr[j].ResetConfig(tree_config);
      }
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
  std::function<FeatureHistogram*()> fill_func_;
  int cache_size_;
  int total_size_;
  bool is_enough_ = false;
  std::vector<int> mapper_;
  std::vector<int> inverse_mapper_;
  std::vector<int> last_used_time_;
  int cur_time_ = 0;
};



}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_FEATURE_HISTOGRAM_HPP_
