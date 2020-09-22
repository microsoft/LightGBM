/*!
 * Copyright (c) 2020 IBM Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_CUDA_KERNEL_LAUNCHER_H_
#define LIGHTGBM_TREELEARNER_CUDA_KERNEL_LAUNCHER_H_

#ifdef USE_CUDA
#include <chrono>
#include "kernels/histogram_16_64_256.hu"  // kernel, acc_type, data_size_t, uchar, score_t

namespace LightGBM {

struct ThreadData {
          // device id
          int             device_id;
          // parameters for cuda_histogram
          int             histogram_size;
          data_size_t     leaf_num_data;
          data_size_t     num_data;
          bool            use_all_features;
          bool            is_constant_hessian;
          int             num_workgroups;
          cudaStream_t    stream;
          uint8_t*        device_features;
          uint8_t*        device_feature_masks;
          data_size_t*    device_data_indices;
          score_t*        device_gradients;
          score_t*        device_hessians;
          score_t         hessians_const;
          char*           device_subhistograms;
          volatile int*   sync_counters;
          void*           device_histogram_outputs;
          size_t          exp_workgroups_per_feature;
          // cuda events
          cudaEvent_t*    kernel_start;
          cudaEvent_t*    kernel_wait_obj;
          std::chrono::duration<double, std::milli>* kernel_input_wait_time;
          // copy histogram
          size_t        output_size;
          char*                 host_histogram_output;
          cudaEvent_t*          histograms_wait_obj;
};


void cuda_histogram(
                int             histogram_size,
                data_size_t     leaf_num_data,
                data_size_t     num_data,
                bool            use_all_features,
                bool            is_constant_hessian,
                int             num_workgroups,
                cudaStream_t    stream,
                uint8_t*        arg0,
                uint8_t*        arg1,
                data_size_t     arg2,
                data_size_t*    arg3,
                data_size_t     arg4,
                score_t*        arg5,
                score_t*        arg6,
                score_t         arg6_const,
                char*           arg7,
                volatile int*   arg8,
                void*           arg9,
                size_t          exp_workgroups_per_feature);

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_TREELEARNER_CUDA_KERNEL_LAUNCHER_H_
