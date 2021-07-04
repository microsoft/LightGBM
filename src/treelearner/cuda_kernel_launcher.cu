/*!
 * Copyright (c) 2020 IBM Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_CUDA

#include "cuda_kernel_launcher.h"

#include <LightGBM/utils/log.h>

#include <cuda_runtime.h>

#include <cstdio>

namespace LightGBM {

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
                size_t          exp_workgroups_per_feature) {
  if (histogram_size == 16) {
    if (leaf_num_data == num_data) {
      if (use_all_features) {
        if (!is_constant_hessian)
          histogram16<<<num_workgroups, 16, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
           histogram16<<<num_workgroups, 16, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      } else {
        if (!is_constant_hessian)
           histogram16_fulldata<<<num_workgroups, 16, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
           histogram16_fulldata<<<num_workgroups, 16, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      }
    } else {
      if (use_all_features) {
        // seems all features is always enabled, so this should be the same as fulldata
        if (!is_constant_hessian)
          histogram16<<<num_workgroups, 16, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
          histogram16<<<num_workgroups, 16, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      } else {
        if (!is_constant_hessian)
          histogram16<<<num_workgroups, 16, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
          histogram16<<<num_workgroups, 16, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      }
    }
  } else if (histogram_size == 64) {
    if (leaf_num_data == num_data) {
      if (use_all_features) {
        if (!is_constant_hessian)
          histogram64<<<num_workgroups, 64, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
          histogram64<<<num_workgroups, 64, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      } else {
        if (!is_constant_hessian)
          histogram64_fulldata<<<num_workgroups, 64, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
          histogram64_fulldata<<<num_workgroups, 64, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      }
    } else {
      if (use_all_features) {
        // seems all features is always enabled, so this should be the same as fulldata
        if (!is_constant_hessian)
          histogram64<<<num_workgroups, 64, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
          histogram64<<<num_workgroups, 64, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      } else {
        if (!is_constant_hessian)
          histogram64<<<num_workgroups, 64, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
          histogram64<<<num_workgroups, 64, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      }
    }
  } else {
    if (leaf_num_data == num_data) {
      if (use_all_features) {
        if (!is_constant_hessian)
          histogram256<<<num_workgroups, 256, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
          histogram256<<<num_workgroups, 256, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      } else {
        if (!is_constant_hessian)
          histogram256_fulldata<<<num_workgroups, 256, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
          histogram256_fulldata<<<num_workgroups, 256, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      }
    } else {
      if (use_all_features) {
        // seems all features is always enabled, so this should be the same as fulldata
        if (!is_constant_hessian)
          histogram256<<<num_workgroups, 256, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
          histogram256<<<num_workgroups, 256, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      } else {
        if (!is_constant_hessian)
          histogram256<<<num_workgroups, 256, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
        else
          histogram256<<<num_workgroups, 256, 0, stream>>>(arg0, arg1, arg2,
                  arg3, arg4, arg5,
                  arg6_const, arg7, arg8, static_cast<acc_type*>(arg9), exp_workgroups_per_feature);
      }
    }
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
