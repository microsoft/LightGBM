/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_BOOSTING_MVS_H_
#define LIGHTGBM_BOOSTING_MVS_H_

#include <LightGBM/utils/array_args.h>
#include <LightGBM/sample_strategy.h>

#include <algorithm>
#include <string>
#include <memory>
#include <vector>

namespace LightGBM {

class MVS : public SampleStrategy {
 public:
  MVS(const Config* config, const Dataset* train_data, int num_tree_per_iteration) {
    config_ = config;
    train_data_ = train_data;
    num_tree_per_iteration_ = num_tree_per_iteration;
    num_data_ = train_data->num_data();
  }

  ~MVS() {}

  void Bagging(int iter, TreeLearner* tree_learner, score_t* gradients, score_t* hessians, const std::vector<std::unique_ptr<Tree>>& models) override;

  void ResetSampleConfig(const Config* config, bool is_change_dataset) override;

  bool IsHessianChange() const override {
    return true;
  }

 protected:
  data_size_t BaggingHelper(data_size_t start, data_size_t cnt, score_t* gradients, score_t* hessians, data_size_t *buffer);

  double GetThreshold(data_size_t begin, data_size_t end);

  double GetLambda(int iter, const score_t* gradients, const score_t* hessians, const std::vector<std::unique_ptr<Tree>>& models);

  double mvs_lambda_;
  double threshold_{0.0};
  std::vector<score_t> tmp_derivatives_;
  bool mvs_adaptive_;

  static constexpr double kMVSEps = 1e-20;
};
}  // namespace LightGBM
#endif   // LIGHTGBM_BOOSTING_MVS_H_
