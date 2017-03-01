#ifndef LIGHTGBM_OPENMP_WRAPPER_H_
#define LIGHTGBM_OPENMP_WRAPPER_H_

#ifdef _OPENMP
  #include <omp.h>
#else
  #ifdef _MSC_VER
    #pragma warning( disable : 4068 ) // disable unknown pragma warning
  #endif

  #ifdef __cplusplus
    extern "C" {
  #endif
    /** Fall here if no OPENMP support, so just
        simulate a single thread running.
        All #pragma omp should be ignored by the compiler **/
    inline void omp_set_num_threads(int) {}
    inline int omp_get_num_threads() {return 1;}
    inline int omp_get_thread_num() {return 0;}
  #ifdef __cplusplus
  }; // extern "C"
  #endif
#endif



#endif /* LIGHTGBM_OPENMP_WRAPPER_H_ */
