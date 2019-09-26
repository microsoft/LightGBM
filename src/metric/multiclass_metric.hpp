/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_METRIC_MULTICLASS_METRIC_HPP_
#define LIGHTGBM_METRIC_MULTICLASS_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <cmath>
#include <vector>

namespace LightGBM {
/*!
* \brief Metric for multiclass task.
* Use static class "PointWiseLossCalculator" to calculate loss point-wise
*/
template<typename PointWiseLossCalculator>
class MulticlassMetric: public Metric {
 public:
  explicit MulticlassMetric(const Config& config) :config_(config) {
    num_class_ = config.num_class;
  }

  virtual ~MulticlassMetric() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back(PointWiseLossCalculator::Name(config_));
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();
    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0;
    int num_tree_per_iteration = num_class_;
    int num_pred_per_row = num_class_;
    if (objective != nullptr) {
      num_tree_per_iteration = objective->NumModelPerIteration();
      num_pred_per_row = objective->NumPredictOneRow();
    }
    if (objective != nullptr) {
      if (weights_ == nullptr) {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> raw_score(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            raw_score[k] = static_cast<double>(score[idx]);
          }
          std::vector<double> rec(num_pred_per_row);
          objective->ConvertOutput(raw_score.data(), rec.data());
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], &rec, config_);
        }
      } else {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> raw_score(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            raw_score[k] = static_cast<double>(score[idx]);
          }
          std::vector<double> rec(num_pred_per_row);
          objective->ConvertOutput(raw_score.data(), rec.data());
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], &rec, config_) * weights_[i];
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> rec(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            rec[k] = static_cast<double>(score[idx]);
          }
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], &rec, config_);
        }
      } else {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> rec(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            rec[k] = static_cast<double>(score[idx]);
          }
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], &rec, config_) * weights_[i];
        }
      }
    }
    double loss = sum_loss / sum_weights_;
    return std::vector<double>(1, loss);
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weighs */
  const label_t* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of this test set */
  std::vector<std::string> name_;
  int num_class_;
  /*! \brief config parameters*/
  Config config_;
};

/*! \brief top-k error for multiclass task; if k=1 (default) this is the usual multi-error */
class MultiErrorMetric: public MulticlassMetric<MultiErrorMetric> {
 public:
  explicit MultiErrorMetric(const Config& config) :MulticlassMetric<MultiErrorMetric>(config) {}

  inline static double LossOnPoint(label_t label, std::vector<double>* score, const Config& config) {
    size_t k = static_cast<size_t>(label);
    int num_larger = 0;
    for (size_t i = 0; i < score->size(); ++i) {
      if (score->at(i) >= score->at(k)) ++num_larger;
      if (num_larger > config.multi_error_top_k) return 1.0f;
    }
    return 0.0f;
  }

  inline static const std::string Name(const Config& config) {
    if (config.multi_error_top_k == 1) {
      return "multi_error";
    } else {
      return "multi_error@" + std::to_string(config.multi_error_top_k);
    }
  }
};

/*! \brief Logloss for multiclass task */
class MultiSoftmaxLoglossMetric: public MulticlassMetric<MultiSoftmaxLoglossMetric> {
 public:
  explicit MultiSoftmaxLoglossMetric(const Config& config) :MulticlassMetric<MultiSoftmaxLoglossMetric>(config) {}

  inline static double LossOnPoint(label_t label, std::vector<double>* score, const Config&) {
    size_t k = static_cast<size_t>(label);
    if (score->at(k) > kEpsilon) {
      return static_cast<double>(-std::log(score->at(k)));
    } else {
      return -std::log(kEpsilon);
    }
  }

  inline static const std::string Name(const Config&) {
    return "multi_logloss";
  }
};

}  // namespace LightGBM
#endif   // LightGBM_METRIC_MULTICLASS_METRIC_HPP_
