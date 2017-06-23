#ifndef LIGHTGBM_TREELEARNER_FEATURE_HISTOGRAM_HPP_
#define LIGHTGBM_TREELEARNER_FEATURE_HISTOGRAM_HPP_

#include "split_info.hpp"

#include <LightGBM/utils/array_args.h>
#include <LightGBM/dataset.h>

#include <cstring>

namespace LightGBM
{

class FeatureMetainfo {
public:
  int num_bin;
  int bias = 0;
  uint32_t default_bin;
  /*! \brief pointer of tree config */
  const TreeConfig* tree_config;
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
  void Init(HistogramBinEntry* data, const FeatureMetainfo* meta, BinType bin_type) {
    meta_ = meta;
    data_ = data;
    if (bin_type == BinType::NumericalBin) {
      find_best_threshold_fun_ = std::bind(&FeatureHistogram::FindBestThresholdNumerical, this, std::placeholders::_1
                                           , std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
    } else {
      find_best_threshold_fun_ = std::bind(&FeatureHistogram::FindBestThresholdCategorical, this, std::placeholders::_1
                                           , std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
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

  void FindBestThreshold(double sum_gradient, double sum_hessian, data_size_t num_data,
                         SplitInfo* output) {
    output->default_bin_for_zero = meta_->default_bin;
    output->gain = kMinScore;
    find_best_threshold_fun_(sum_gradient, sum_hessian + 2 * kEpsilon, num_data, output);
  }

  void FindBestThresholdNumerical(double sum_gradient, double sum_hessian, data_size_t num_data,
                                  SplitInfo* output) {

    is_splittable_ = false;
    double gain_shift = GetLeafSplitGain(sum_gradient, sum_hessian,
                                         meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2);
    double min_gain_shift = gain_shift + meta_->tree_config->min_gain_to_split;
    if (meta_->tree_config->use_missing && meta_->num_bin > 2) {
      FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_gain_shift, output, 0);
      FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_gain_shift, output, meta_->num_bin - 1);
    } else {
      FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_gain_shift, output, meta_->default_bin);
    }
    output->gain -= min_gain_shift;
  }

  void FindBestThresholdCategorical(double sum_gradient, double sum_hessian, data_size_t num_data,
                                    SplitInfo* output) {
    double best_gain = kMinScore;
    uint32_t best_threshold = static_cast<uint32_t>(meta_->num_bin);
    data_size_t best_left_count = 0;
    double best_sum_left_gradient = 0.0f;
    double best_sum_left_hessian = 0.0f;
    double gain_shift = GetLeafSplitGain(sum_gradient, sum_hessian,
                                         meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2);
    double min_gain_shift = gain_shift + meta_->tree_config->min_gain_to_split;
    is_splittable_ = false;
    const int bias = meta_->bias;
    int t = meta_->num_bin - 1 - bias;
    const int t_end = 0;
    // from right to left, and we don't need data in bin0
    for (; t >= t_end; --t) {
      // if data not enough, or sum hessian too small
      if (data_[t].cnt < meta_->tree_config->min_data_in_leaf
          || data_[t].sum_hessians < meta_->tree_config->min_sum_hessian_in_leaf) continue;
      data_size_t other_count = num_data - data_[t].cnt;
      // if data not enough
      if (other_count < meta_->tree_config->min_data_in_leaf) continue;

      double sum_other_hessian = sum_hessian - data_[t].sum_hessians - kEpsilon;
      // if sum hessian too small
      if (sum_other_hessian < meta_->tree_config->min_sum_hessian_in_leaf) continue;

      double sum_other_gradient = sum_gradient - data_[t].sum_gradients;
      // current split gain
      double current_gain = GetLeafSplitGain(sum_other_gradient, sum_other_hessian,
                                             meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2)
        + GetLeafSplitGain(data_[t].sum_gradients, data_[t].sum_hessians + kEpsilon,
                           meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2);
      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) continue;

      // mark to is splittable
      is_splittable_ = true;
      // better split point
      if (current_gain > best_gain) {
        best_threshold = static_cast<uint32_t>(t + bias);
        best_sum_left_gradient = data_[t].sum_gradients;
        best_sum_left_hessian = data_[t].sum_hessians + kEpsilon;
        best_left_count = data_[t].cnt;
        best_gain = current_gain;
      }
    }
    // need restore zero bin
    if (bias == 1) {
      t = meta_->num_bin - 1 - bias;
      double sum_bin0_gradient = sum_gradient;
      double sum_bin0_hessian = sum_hessian - 2 * kEpsilon;
      data_size_t cnt_bin0 = num_data;
      for (; t >= 0; --t) {
        sum_bin0_gradient -= data_[t].sum_gradients;
        sum_bin0_hessian -= data_[t].sum_hessians;
        cnt_bin0 -= data_[t].cnt;
      }
      data_size_t other_count = num_data - cnt_bin0;
      double sum_other_hessian = sum_hessian - sum_bin0_hessian - kEpsilon;
      if (cnt_bin0 >= meta_->tree_config->min_data_in_leaf
          && sum_bin0_hessian >= meta_->tree_config->min_sum_hessian_in_leaf
          && other_count >= meta_->tree_config->min_data_in_leaf
          && sum_other_hessian >= meta_->tree_config->min_sum_hessian_in_leaf) {
        double sum_other_gradient = sum_gradient - sum_bin0_gradient;
        double current_gain = GetLeafSplitGain(sum_other_gradient, sum_other_hessian,
                                               meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2)
          + GetLeafSplitGain(sum_bin0_gradient, sum_bin0_hessian + kEpsilon,
                             meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2);
        if (current_gain > min_gain_shift) {
          is_splittable_ = true;
          // better split point
          if (current_gain > best_gain) {
            best_threshold = static_cast<uint32_t>(0);
            best_sum_left_gradient = sum_bin0_gradient;
            best_sum_left_hessian = sum_bin0_hessian + kEpsilon;
            best_left_count = cnt_bin0;
            best_gain = current_gain;
          }
        }
      }
    }
    if (is_splittable_) {
      // update split information
      output->threshold = best_threshold;
      output->left_output = CalculateSplittedLeafOutput(best_sum_left_gradient, best_sum_left_hessian,
                                                        meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2);
      output->left_count = best_left_count;
      output->left_sum_gradient = best_sum_left_gradient;
      output->left_sum_hessian = best_sum_left_hessian - kEpsilon;
      output->right_output = CalculateSplittedLeafOutput(sum_gradient - best_sum_left_gradient,
                                                         sum_hessian - best_sum_left_hessian,
                                                         meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2);
      output->right_count = num_data - best_left_count;
      output->right_sum_gradient = sum_gradient - best_sum_left_gradient;
      output->right_sum_hessian = sum_hessian - best_sum_left_hessian - kEpsilon;
      output->gain = best_gain - min_gain_shift;
    } 
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

  /*!
  * \brief Calculate the split gain based on regularized sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  * \return split gain
  */
  static double GetLeafSplitGain(double sum_gradients, double sum_hessians, double l1, double l2) {
    double abs_sum_gradients = std::fabs(sum_gradients);
    double reg_abs_sum_gradients = std::max(0.0, abs_sum_gradients - l1);
    return (reg_abs_sum_gradients * reg_abs_sum_gradients)
      / (sum_hessians + l2);

  }

  /*!
  * \brief Calculate the output of a leaf based on regularized sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  * \return leaf output
  */
  static double CalculateSplittedLeafOutput(double sum_gradients, double sum_hessians, double l1, double l2) {
    double abs_sum_gradients = std::fabs(sum_gradients);
    double reg_abs_sum_gradients = std::max(0.0, abs_sum_gradients - l1);
    return -std::copysign(reg_abs_sum_gradients, sum_gradients)
      / (sum_hessians + l2);
  }

private:

  void FindBestThresholdSequence(double sum_gradient, double sum_hessian, data_size_t num_data, double min_gain_shift,
                                  SplitInfo* output, uint32_t default_bin_for_zero) {
    int dir = -1;
    if (static_cast<int>(default_bin_for_zero) == meta_->num_bin - 1) { dir = 1; };

    bool skip_default_bin = true;
    if (static_cast<int>(default_bin_for_zero) > 0 && static_cast<int>(default_bin_for_zero) < meta_->num_bin - 1) {
      skip_default_bin = false;
    }
    const int bias = meta_->bias;

    double best_sum_left_gradient = NAN;
    double best_sum_left_hessian = NAN;
    double best_gain = kMinScore;
    data_size_t best_left_count = 0;
    uint32_t best_threshold = static_cast<uint32_t>(meta_->num_bin);

    if (dir == -1) {

      double sum_right_gradient = 0.0f;
      double sum_right_hessian = kEpsilon;
      data_size_t right_count = 0;

      int t = meta_->num_bin - 1 - bias;
      const int t_end = 1 - bias;

      // from right to left, and we don't need data in bin0
      for (; t >= t_end; --t) {

        // need to skip default bin
        if (skip_default_bin && (t + bias) == static_cast<int>(meta_->default_bin)) { continue; }

        sum_right_gradient += data_[t].sum_gradients;
        sum_right_hessian += data_[t].sum_hessians;
        right_count += data_[t].cnt;
        // if data not enough, or sum hessian too small
        if (right_count < meta_->tree_config->min_data_in_leaf
            || sum_right_hessian < meta_->tree_config->min_sum_hessian_in_leaf) continue;
        data_size_t left_count = num_data - right_count;
        // if data not enough
        if (left_count < meta_->tree_config->min_data_in_leaf) break;

        double sum_left_hessian = sum_hessian - sum_right_hessian;
        // if sum hessian too small
        if (sum_left_hessian < meta_->tree_config->min_sum_hessian_in_leaf) break;

        double sum_left_gradient = sum_gradient - sum_right_gradient;
        // current split gain
        double current_gain = GetLeafSplitGain(sum_left_gradient, sum_left_hessian,
                                               meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2)
          + GetLeafSplitGain(sum_right_gradient, sum_right_hessian,
                             meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2);
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
    } else{
      double sum_left_gradient = 0.0f;
      double sum_left_hessian = kEpsilon;
      data_size_t left_count = 0;

      int t = 0;
      const int t_end = meta_->num_bin - 2 - bias;

      // from right to left, and we don't need data in bin0
      for (; t <= t_end; ++t) {

        // need to skip default bin
        if (skip_default_bin && (t + bias) == static_cast<int>(meta_->default_bin)) { continue; }

        sum_left_gradient += data_[t].sum_gradients;
        sum_left_hessian += data_[t].sum_hessians;
        left_count += data_[t].cnt;
        // if data not enough, or sum hessian too small
        if (left_count < meta_->tree_config->min_data_in_leaf
            || sum_left_hessian < meta_->tree_config->min_sum_hessian_in_leaf) continue;
        data_size_t right_count = num_data - left_count;
        // if data not enough
        if (right_count < meta_->tree_config->min_data_in_leaf) break;

        double sum_right_hessian = sum_hessian - sum_left_hessian;
        // if sum hessian too small
        if (sum_right_hessian < meta_->tree_config->min_sum_hessian_in_leaf) break;

        double sum_right_gradient = sum_gradient - sum_left_gradient;
        // current split gain
        double current_gain = GetLeafSplitGain(sum_left_gradient, sum_left_hessian,
                                               meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2)
          + GetLeafSplitGain(sum_right_gradient, sum_right_hessian,
                             meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2);
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
                                                        meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2);
      output->left_count = best_left_count;
      output->left_sum_gradient = best_sum_left_gradient;
      output->left_sum_hessian = best_sum_left_hessian - kEpsilon;
      output->right_output = CalculateSplittedLeafOutput(sum_gradient - best_sum_left_gradient,
                                                         sum_hessian - best_sum_left_hessian,
                                                         meta_->tree_config->lambda_l1, meta_->tree_config->lambda_l2);
      output->right_count = num_data - best_left_count;
      output->right_sum_gradient = sum_gradient - best_sum_left_gradient;
      output->right_sum_hessian = sum_hessian - best_sum_left_hessian - kEpsilon;
      output->gain = best_gain;
      output->default_bin_for_zero = default_bin_for_zero;
    }
  }

  const FeatureMetainfo* meta_;
  /*! \brief sum of gradient of each bin */
  HistogramBinEntry* data_;
  //std::vector<HistogramBinEntry> data_;
  /*! \brief False if this histogram cannot split */
  bool is_splittable_ = true;

  std::function<void(double, double, data_size_t, SplitInfo*)> find_best_threshold_fun_;
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

  void DynamicChangeSize(const Dataset* train_data, const TreeConfig* tree_config, int cache_size, int total_size) {
    if (feature_metas_.empty()) {
      int num_feature = train_data->num_features();
      feature_metas_.resize(num_feature);
      #pragma omp parallel for schedule(static, 512) if(num_feature >= 1024)
      for (int i = 0; i < num_feature; ++i) {
        feature_metas_[i].num_bin = train_data->FeatureNumBin(i);
        feature_metas_[i].default_bin = train_data->FeatureBinMapper(i)->GetDefaultBin();
        if (train_data->FeatureBinMapper(i)->GetDefaultBin() == 0) {
          feature_metas_[i].bias = 1;
        } else {
          feature_metas_[i].bias = 0;
        }
        feature_metas_[i].tree_config = tree_config;
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
        pool_[i][j].Init(data_[i].data() + offset, &feature_metas_[j], train_data->FeatureBinMapper(j)->bin_type());
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
  }

  void ResetConfig(const TreeConfig* tree_config) {
    int size = static_cast<int>(feature_metas_.size());
    #pragma omp parallel for schedule(static, 512) if(size >= 1024)
    for (int i = 0; i < size; ++i) {
      feature_metas_[i].tree_config = tree_config;
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
};

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_FEATURE_HISTOGRAM_HPP_
