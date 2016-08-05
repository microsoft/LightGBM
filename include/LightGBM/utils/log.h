#ifndef LIGHTGBM_UTILS_LOG_H_
#define LIGHTGBM_UTILS_LOG_H_

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>

namespace LightGBM {

class Log {
public:

  inline static void Stderr(const char *format, ...) {
    va_list argptr;
    char fixed[512];
#ifdef _MSC_VER
    sprintf_s(fixed, "[LightGBM Error] %s \n", format);
#else
    sprintf(fixed, "[LightGBM Error] %s \n", format);
#endif
    va_start(argptr, format);
    vfprintf(stderr, fixed, argptr);
    va_end(argptr);
    fflush(stderr);
    std::exit(1);
  }

  inline static void Stdout(const char *format, ...) {
    va_list argptr;
    char fixed[512];
#ifdef _MSC_VER
    sprintf_s(fixed, "[LightGBM] %s\n", format);
#else
    sprintf(fixed, "[LightGBM] %s\n", format);
#endif
    va_start(argptr, format);
    vfprintf(stdout, fixed, argptr);
    va_end(argptr);
    fflush(stdout);
  }
};

#define CHECK(condition)                                   \
  if (!(condition)) Log::Stderr("Check failed: " #condition \
     " at %s, line %d .\n", __FILE__,  __LINE__);

}  // namespace LightGBM
#endif #endif  // LightGBM_UTILS_LOG_H_
