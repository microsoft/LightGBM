#ifndef LIGHTGBM_UTILS_LOG_H_
#define LIGHTGBM_UTILS_LOG_H_

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <fstream>

namespace LightGBM {


#ifndef CHECK
#define CHECK(condition)                                   \
  if (!(condition)) Log::Fatal("Check failed: " #condition \
     " at %s, line %d .\n", __FILE__,  __LINE__);
#endif

#ifndef CHECK_NOTNULL
#define CHECK_NOTNULL(pointer)                             \
  if ((pointer) == nullptr) LightGBM::Log::Fatal(#pointer " Can't be NULL\n");
#endif

// A enumeration type of log message levels. The values are ordered:
// Debug < Info < Error < Fatal.
enum class LogLevel : int {
  Debug = 0,
  Info = 1,
  Error = 2,
  Fatal = 3
};

/*!
* \brief The Logger class is responsible for writing log messages into
*        standard output or log file.
*/
class Logger {
  // Enable the static Log class to call the private method.
  friend class Log;

public:
  /*!
  * \brief Creates an instance of Logger class. By default, the log
  *        messages will be written to standard output with minimal
  *        level of INFO. Users are able to further set the log file or
  *        log level with corresponding methods.
  * \param level Minimal log level, Info by default.
  */
  explicit Logger(LogLevel level = LogLevel::Info);
  /*!
  * \brief Creates an instance of Logger class by specifying log file
  *        and log level. The log message will be written to both STDOUT
  *        and file (if created successfully).
  * \param filename Log file name
  * \param level Minimal log level
  */
  explicit Logger(std::string filename, LogLevel level = LogLevel::Info);
  ~Logger();

  /*!
  * \brief Resets the log file.
  * \param filename The new log filename. If it is empty, the Logger
  *        will close current log file (if it exists).
  * \return Returns -1 if the filename is not empty but failed on
  *         creating the log file, or 0 will be returned otherwise.
  */
  int ResetLogFile(std::string filename);
  /*!
  * \brief Resets the log level.
  * \param level The new log level.
  */
  void ResetLogLevel(LogLevel level) { level_ = level; }
  /*!
  * \brief Resets the option of whether kill the process when fatal
  *        error occurs. By defualt the option is false.
  */
  void ResetKillFatal(bool is_kill_fatal) { is_kill_fatal_ = is_kill_fatal; }

  /*!
  * \brief C style formatted method for writing log messages. A message
  *        is with the following format: [LEVEL] [TIME] message
  * \param level The log level of this message.
  * \param format The C format string.
  * \param ... Output items.
  */
  void Write(LogLevel level, const char *format, ...);
  void Debug(const char *format, ...);
  void Info(const char *format, ...);
  void Error(const char *format, ...);
  void Fatal(const char *format, ...);

private:
  void Write(LogLevel level, const char* format, va_list* val);
  void CloseLogFile();
  // Returns current system time as a string.
  std::string GetSystemTime();
  // Returns the string of a log level.
  std::string GetLevelStr(LogLevel level);

  std::FILE *file_;  // A file pointer to the log file.
  LogLevel level_;   // Only the message not less than level_ will be outputed.
  bool is_kill_fatal_;  // If kill the process when fatal error occurs.

  // No copying allowed
  Logger(const Logger&);
  void operator=(const Logger&);
};

/*!
* \brief The Log class is a static wrapper of a global Logger instance in
*        the scope of a process. Users can write logging messages easily
*        with the static methods.
*/
class Log {
public:
  /*!
  * \brief Resets the log file. The logger will write messages to the
  *        log file if it exists in addition to the STDOUT by default.
  * \param filename The log filename. If it is empty, the logger will
  *        close the current log file (if it exists) and only output to
  *        STDOUT.
  * \return -1 if fail on creating the log file, or 0 otherwise.
  */
  static int ResetLogFile(std::string filename);
  /*!
  * \brief Resets the minimal log level. It is INFO by default.
  * \param level The new minimal log level.
  */
  static void ResetLogLevel(LogLevel level);
  /*!
  * \brief Resets the option of whether kill the process when fatal
  *        error occurs. By defualt the option is false.
  */
  static void ResetKillFatal(bool is_kill_fatal);

  /*! \brief The C formatted methods of writing the messages. */
  static void Write(LogLevel level, const char* format, ...);
  static void Debug(const char *format, ...);
  static void Info(const char *format, ...);
  static void Error(const char *format, ...);
  static void Fatal(const char *format, ...);

private:
  static Logger logger_;
};

}  // namespace LightGBM
#endif   // LightGBM_UTILS_LOG_H_
