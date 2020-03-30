/*
 * ibmGBT: IBM CUDA Accelerated LightGBM
 *
 * IBM Confidential
 * (C) Copyright IBM Corp. 2019. All Rights Reserved.
 *
 * The source code for this program is not published or otherwise
 * divested of its trade secrets, irrespective of what has been
 * deposited with the U.S. Copyright Office.
 *
 * US Government Users Restricted Rights - Use, duplication or
 * disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef LGBM_CUDA_UTILS_H
#define LGBM_CUDA_UTILS_H

//LGBM_CUDA

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDASUCCESS_OR_FATAL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      LightGBM::Log::Fatal("CUDA_RUNTIME: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif /* USE_CUDA */

#endif
