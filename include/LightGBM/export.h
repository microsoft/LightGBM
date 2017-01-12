#ifndef LIGHTGBM_EXPORT_H_
#define LIGHTGBM_EXPORT_H_

/** Macros for exporting symbols in MSVC/GCC/CLANG **/

#ifdef __cplusplus
#define LIGHTGBM_EXTERN_C extern "C"
#else
#define LIGHTGBM_EXTERN_C
#endif


#ifdef _MSC_VER
#define LIGHTGBM_EXPORT __declspec(dllexport)
#define LIGHTGBM_C_EXPORT LIGHTGBM_EXTERN_C __declspec(dllexport)
#else
#define LIGHTGBM_EXPORT 
#define LIGHTGBM_C_EXPORT LIGHTGBM_EXTERN_C
#endif

#endif /** LIGHTGBM_EXPORT_H_ **/
