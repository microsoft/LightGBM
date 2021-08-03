/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_CUDA_META_DATA_HPP_
#define LIGHTGBM_CUDA_META_DATA_HPP_

#include "../meta.h"

#include <LightGBM/cuda/cuda_utils.h>

namespace LightGBM {

class CUDAMetadata {
 public:
  CUDAMetadata();

  ~CUDAMetadata();

  void Init(const std::vector<label_t>& label,
            const std::vector<label_t>& weight,
            const std::vector<data_size_t>& query_boundaries,
            const std::vector<label_t>& query_weights,
            const std::vector<double>& init_score,
            const std::vector<data_size_t>& queries);

  const label_t* cuda_label() const { return cuda_label_; }

  const label_t* cuda_weights() const { return cuda_weights_; }

  const data_size_t* cuda_query_boundaries() const { return cuda_query_boundaries_; }

 private:
  label_t* cuda_label_;
  label_t* cuda_weights_;
  data_size_t* cuda_query_boundaries_;
  label_t* cuda_query_weights_;
  double* cuda_init_score_;
  data_size_t* cuda_queries_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_CUDA_META_DATA_HPP_
