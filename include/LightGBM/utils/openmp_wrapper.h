/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_OPENMP_WRAPPER_H_
#define LIGHTGBM_OPENMP_WRAPPER_H_

#ifdef _OPENMP

#include <LightGBM/utils/log.h>

#include <omp.h>

#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

inline int OMP_NUM_THREADS() {
  int ret = 1;
#pragma omp parallel
#pragma omp master
  { ret = omp_get_num_threads(); }
  return ret;
}

class ThreadExceptionHelper {
 public:
  ThreadExceptionHelper() {
    ex_ptr_ = nullptr;
  }

  ~ThreadExceptionHelper() {
    ReThrow();
  }
  void ReThrow() {
    if (ex_ptr_ != nullptr) {
      std::rethrow_exception(ex_ptr_);
    }
  }
  void CaptureException() {
    // only catch first exception.
    if (ex_ptr_ != nullptr) { return; }
    std::unique_lock<std::mutex> guard(lock_);
    if (ex_ptr_ != nullptr) { return; }
    ex_ptr_ = std::current_exception();
  }

 private:
  std::exception_ptr ex_ptr_;
  std::mutex lock_;
};

#define OMP_INIT_EX() ThreadExceptionHelper omp_except_helper
#define OMP_LOOP_EX_BEGIN() try {
#define OMP_LOOP_EX_END()                 \
  }                                       \
  catch (std::exception & ex) {           \
    Log::Warning(ex.what());              \
    omp_except_helper.CaptureException(); \
  }                                       \
  catch (...) {                           \
    omp_except_helper.CaptureException(); \
  }
#define OMP_THROW_EX() omp_except_helper.ReThrow()

#else

/*
 * To be compatible with OpenMP, define a nothrow macro which is used by gcc
 * openmp, but not by clang.
 * See also https://github.com/dmlc/dmlc-core/blob/3106c1cbdcc9fc9ef3a2c1d2196a7a6f6616c13d/include/dmlc/omp.h#L14
 */
#if defined(__clang__)
#undef __GOMP_NOTHROW
#define __GOMP_NOTHROW
#elif defined(__cplusplus)
#undef __GOMP_NOTHROW
#define __GOMP_NOTHROW throw()
#else
#undef __GOMP_NOTHROW
#define __GOMP_NOTHROW __attribute__((__nothrow__))
#endif

#ifdef _MSC_VER
  #pragma warning(disable : 4068)  // disable unknown pragma warning
#endif

#ifdef __cplusplus
  extern "C" {
#endif
  /** Fall here if no OPENMP support, so just
      simulate a single thread running.
      All #pragma omp should be ignored by the compiler **/
  inline void omp_set_num_threads(int) __GOMP_NOTHROW {}  // NOLINT (no cast done here)
  inline int omp_get_num_threads() __GOMP_NOTHROW {return 1;}
  inline int omp_get_max_threads() __GOMP_NOTHROW {return 1;}
  inline int omp_get_thread_num() __GOMP_NOTHROW {return 0;}
  inline int OMP_NUM_THREADS() __GOMP_NOTHROW { return 1; }
#ifdef __cplusplus
}  // extern "C"
#endif

#define OMP_INIT_EX()
#define OMP_LOOP_EX_BEGIN()
#define OMP_LOOP_EX_END()
#define OMP_THROW_EX()

#endif



#endif /* LIGHTGBM_OPENMP_WRAPPER_H_ */
