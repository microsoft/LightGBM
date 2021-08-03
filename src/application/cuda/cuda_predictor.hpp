/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_APPLICATION_CUDA_CUDA_PREDICTOR_HPP_
#define LIGHTGBM_APPLICATION_CUDA_CUDA_PREDICTOR_HPP_

#include <LightGBM/cuda/cuda_tree.hpp>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/prediction_early_stop.h>

#include "../predictor.hpp"

#define CUDA_PREDICTOR_MAX_TREE_SIZE (1024)
#define CUAA_PREDICTOR_PREDICT_BLOCK_SIZE (1024)

namespace LightGBM {

class CUDAPredictor : public Predictor {
 public:
  CUDAPredictor(Boosting* boosting, int start_iteration, int num_iteration, bool is_raw_score,
            bool predict_leaf_index, bool predict_contrib, bool early_stop,
            int early_stop_freq, double early_stop_margin);

  ~CUDAPredictor();

  virtual void Predict(const char* data_filename, const char* result_filename, bool header, bool disable_shape_check) override;
 private:
  void InitCUDAModel();

  data_size_t ReadDataToCUDADevice(const char* data_filename, const bool header, const bool diable_shape_check);

  void LaunchPredictKernel(const data_size_t num_data);

  void GetPredictRowPtr();

  std::vector<int> predict_feature_index_;
  std::vector<double> predict_feature_value_;
  std::vector<data_size_t> predict_row_ptr_;
  std::vector<double> result_buffer_;
  int num_trees_;

  int* cuda_predict_feature_index_;
  double* cuda_predict_feature_value_;
  data_size_t* cuda_predict_row_ptr_;
  double* cuda_result_buffer_;
  double* cuda_data_;

  int* cuda_tree_num_leaves_;
  const int** cuda_left_child_;
  const int** cuda_right_child_;
  const double** cuda_threshold_;
  const int8_t** cuda_decision_type_;
  const double** cuda_leaf_value_;
  const int** cuda_split_feature_index_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_APPLICATION_CUDA_CUDA_PREDICTOR_HPP_
