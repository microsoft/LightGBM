/*!
 * Copyright (c) 2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef _OPENMP

#include <LightGBM/utils/openmp_wrapper.h>

int LGBM_MAX_NUM_THREADS = -1;

int LGBM_DEFAULT_NUM_THREADS = -1;

void OMP_SET_NUM_THREADS(int num_threads) {
  if (num_threads <= 0) {
    LGBM_DEFAULT_NUM_THREADS = -1;
  } else {
    LGBM_DEFAULT_NUM_THREADS = num_threads;
  }
}

#endif  // _OPENMP
