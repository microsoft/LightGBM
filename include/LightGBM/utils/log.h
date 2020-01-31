/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_UTILS_LOG_H_
#define LIGHTGBM_UTILS_LOG_H_

#include <string>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace LightGBM {

#if defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL thread_local
#endif

#ifndef CHECK
#define CHECK(condition)                                   \
  if (!(condition)) Log::Fatal("Check failed: " #condition \
     " at %s, line %d .\n", __FILE__,  __LINE__);
#endif

#ifndef CHECK_NOTNULL
#define CHECK_NOTNULL(pointer)                             \
  if ((pointer) == nullptr) LightGBM::Log::Fatal(#pointer " Can't be NULL at %s, line %d .\n", __FILE__,  __LINE__);
#endif


enum class LogLevel: int {
  Fatal = -1,
  Warning = 0,
  Info = 1,
  Debug = 2,
};


/*!
* \brief A static Log class
*/
class Log {
 public:
  /*!
  * \brief Resets the minimal log level. It is INFO by default.
  * \param level The new minimal log level.
  */
  static void ResetLogLevel(LogLevel level) {
    GetLevel() = level;
  }

  static void Debug(const char *format, ...) {
    va_list val;
    va_start(val, format);
    Write(LogLevel::Debug, "Debug", format, val);
    va_end(val);
  }
  static void Info(const char *format, ...) {
    va_list val;
    va_start(val, format);
    Write(LogLevel::Info, "Info", format, val);
    va_end(val);
  }
  static void Warning(const char *format, ...) {
    va_list val;
    va_start(val, format);
    Write(LogLevel::Warning, "Warning", format, val);
    va_end(val);
  }
  static void Fatal(const char *format, ...) {
    va_list val;
    char str_buf[1024];
    va_start(val, format);
#ifdef _MSC_VER
    vsprintf_s(str_buf, format, val);
#else
    vsprintf(str_buf, format, val);
#endif
    va_end(val);
    fprintf(stderr, "[LightGBM] [Fatal] %s\n", str_buf);
    fflush(stderr);
    throw std::runtime_error(std::string(str_buf));
  }

 private:
  static void Write(LogLevel level, const char* level_str, const char *format, va_list val) {
    if (level <= GetLevel()) {  // omit the message with low level
      // write to STDOUT
      printf("[LightGBM] [%s] ", level_str);
      vprintf(format, val);
      printf("\n");
      fflush(stdout);
    }
  }

  // a trick to use static variable in header file.
  // May be not good, but avoid to use an additional cpp file
  static LogLevel& GetLevel() { static THREAD_LOCAL LogLevel level = LogLevel::Info; return level; }
};

}  // namespace LightGBM
#endif   // LightGBM_UTILS_LOG_H_
