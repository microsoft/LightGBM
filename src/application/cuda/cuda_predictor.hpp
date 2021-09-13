/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_APPLICATION_CUDA_CUDA_PREDICTOR_HPP_
#define LIGHTGBM_APPLICATION_CUDA_CUDA_PREDICTOR_HPP_

#include <LightGBM/cuda/cuda_objective_function.hpp>
#include <LightGBM/cuda/cuda_tree.hpp>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/objective_function.h>
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

  virtual void Predict(const char* data_filename, const char* result_filename, bool header, bool disable_shape_check, bool precise_float_parser) override;

  virtual void Predict(const data_size_t num_data,
                       const int64_t num_pred_in_one_row,
                       const std::function<std::vector<std::pair<int, double>>(int row_idx)>& get_row_fun,
                       double* out_result) override;

 private:
  void InitCUDAModel(const int start_iteration, const int num_iteration);

  void LaunchPredictKernelAsync(const data_size_t num_data, const bool is_csr);

  void PredictWithParserFun(std::function<void(const char*, std::vector<std::pair<int, double>>*)> parser_fun,
                            TextReader<data_size_t>* predict_data_reader,
                            VirtualFileWriter* writer);

  std::function<void(const char*, std::vector<std::pair<int, double>>*)> GetParserFun(const char* data_filename,
                                                                                      const bool header,
                                                                                      const bool disable_shape_check);

  double* cuda_result_buffer_;
  double* cuda_data_;

  int* cuda_tree_num_leaves_;
  const int** cuda_left_child_;
  const int** cuda_right_child_;
  const double** cuda_threshold_;
  const int8_t** cuda_decision_type_;
  const double** cuda_leaf_value_;
  const int** cuda_split_feature_index_;

  cudaStream_t cuda_stream_;

  int start_iteration_;
  int num_iteration_;
  int64_t num_pred_in_one_row_;
  const bool is_raw_score_;
  const bool predict_leaf_index_;
  const bool predict_contrib_;
  std::function<void(data_size_t, const double*, double*)> cuda_convert_output_function_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_APPLICATION_CUDA_CUDA_PREDICTOR_HPP_
