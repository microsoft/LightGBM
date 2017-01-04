#ifndef LIGHTGBM_UTILS_COMMON_FUN_H_
#define LIGHTGBM_UTILS_COMMON_FUN_H_

#include <LightGBM/utils/log.h>

#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <type_traits>
#include <iomanip>

namespace LightGBM {

namespace Common {

inline char tolower(char in) {
  if (in <= 'Z' && in >= 'A')
    return in - ('Z' - 'z');
  return in;
}

inline static std::string& Trim(std::string& str) {
  if (str.empty()) {
    return str;
  }
  str.erase(str.find_last_not_of(" \f\n\r\t\v") + 1);
  str.erase(0, str.find_first_not_of(" \f\n\r\t\v"));
  return str;
}

inline static std::string& RemoveQuotationSymbol(std::string& str) {
  if (str.empty()) {
    return str;
  }
  str.erase(str.find_last_not_of("'\"") + 1);
  str.erase(0, str.find_first_not_of("'\""));
  return str;
}
inline static bool StartsWith(const std::string& str, const std::string prefix) {
  if (str.substr(0, prefix.size()) == prefix) {
    return true;
  } else {
    return false;
  }
}
inline static std::vector<std::string> Split(const char* c_str, char delimiter) {
  std::vector<std::string> ret;
  std::string str(c_str);
  size_t i = 0;
  size_t pos = str.find(delimiter);
  while (pos != std::string::npos) {
    ret.push_back(str.substr(i, pos - i));
    i = ++pos;
    pos = str.find(delimiter, pos);
  }
  ret.push_back(str.substr(i));
  return ret;
}

inline static std::vector<std::string> Split(const char* c_str, const char* delimiters) {
  // will split when met any chars in delimiters
  std::vector<std::string> ret;
  std::string str(c_str);
  size_t i = 0;
  size_t pos = str.find_first_of(delimiters);
  while (pos != std::string::npos) {
    ret.push_back(str.substr(i, pos - i));
    i = ++pos;
    pos = str.find_first_of(delimiters, pos);
  }
  ret.push_back(str.substr(i));
  return ret;
}

inline static std::string FindFromLines(const std::vector<std::string>& lines, const char* key_word) {
  for (auto& line : lines) {
    size_t find_pos = line.find(key_word);
    if (find_pos != std::string::npos) {
      return line;
    }
  }
  return "";
}

inline static const char* Atoi(const char* p, int* out) {
  int sign, value;
  while (*p == ' ') {
    ++p;
  }
  sign = 1;
  if (*p == '-') {
    sign = -1;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  for (value = 0; *p >= '0' && *p <= '9'; ++p) {
    value = value * 10 + (*p - '0');
  }
  *out = sign * value;
  while (*p == ' ') {
    ++p;
  }
  return p;
}

inline static const char* Atof(const char* p, double* out) {
  int frac;
  double sign, value, scale;
  *out = 0;
  // Skip leading white space, if any.
  while (*p == ' ') {
    ++p;
  }

  // Get sign, if any.
  sign = 1.0;
  if (*p == '-') {
    sign = -1.0;
    ++p;
  } else if (*p == '+') {
    ++p;
  }

  // is a number
  if ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E') {
    // Get digits before decimal point or exponent, if any.
    for (value = 0.0; *p >= '0' && *p <= '9'; ++p) {
      value = value * 10.0 + (*p - '0');
    }

    // Get digits after decimal point, if any.
    if (*p == '.') {
      double pow10 = 10.0;
      ++p;
      while (*p >= '0' && *p <= '9') {
        value += (*p - '0') / pow10;
        pow10 *= 10.0;
        ++p;
      }
    }

    // Handle exponent, if any.
    frac = 0;
    scale = 1.0;
    if ((*p == 'e') || (*p == 'E')) {
      unsigned int expon;
      // Get sign of exponent, if any.
      ++p;
      if (*p == '-') {
        frac = 1;
        ++p;
      } else if (*p == '+') {
        ++p;
      }
      // Get digits of exponent, if any.
      for (expon = 0; *p >= '0' && *p <= '9'; ++p) {
        expon = expon * 10 + (*p - '0');
      }
      if (expon > 308) expon = 308;
      // Calculate scaling factor.
      while (expon >= 50) { scale *= 1E50; expon -= 50; }
      while (expon >= 8) { scale *= 1E8;  expon -= 8; }
      while (expon > 0) { scale *= 10.0; expon -= 1; }
    }
    // Return signed and scaled floating point result.
    *out = sign * (frac ? (value / scale) : (value * scale));
  } else {
    size_t cnt = 0;
    while (*(p + cnt) != '\0' && *(p + cnt) != ' '
      && *(p + cnt) != '\t' && *(p + cnt) != ','
      && *(p + cnt) != '\n' && *(p + cnt) != '\r'
      && *(p + cnt) != ':') {
      ++cnt;
    }
    if (cnt > 0) {
      std::string tmp_str(p, cnt);
      std::transform(tmp_str.begin(), tmp_str.end(), tmp_str.begin(), Common::tolower);
      if (tmp_str == std::string("na") || tmp_str == std::string("nan")) {
        *out = 0;
      } else if (tmp_str == std::string("inf") || tmp_str == std::string("infinity")) {
        *out = sign * 1e308;
      } else {
        Log::Fatal("Unknown token %s in data file", tmp_str.c_str());
      }
      p += cnt;
    }
  }

  while (*p == ' ') {
    ++p;
  }

  return p;
}



inline bool AtoiAndCheck(const char* p, int* out) {
  const char* after = Atoi(p, out);
  if (*after != '\0') {
    return false;
  }
  return true;
}

inline bool AtofAndCheck(const char* p, double* out) {
  const char* after = Atof(p, out);
  if (*after != '\0') {
    return false;
  }
  return true;
}

inline static const char* SkipSpaceAndTab(const char* p) {
  while (*p == ' ' || *p == '\t') {
    ++p;
  }
  return p;
}

inline static const char* SkipReturn(const char* p) {
  while (*p == '\n' || *p == '\r' || *p == ' ') {
    ++p;
  }
  return p;
}

template<typename T, typename T2>
inline static std::vector<T2> ArrayCast(const std::vector<T>& arr) {
  std::vector<T2> ret;
  for (size_t i = 0; i < arr.size(); ++i) {
    ret.push_back(static_cast<T2>(arr[i]));
  }
  return ret;
}

template<typename T>
inline static std::string ArrayToString(const std::vector<T>& arr, char delimiter) {
  if (arr.empty()) {
    return std::string("");
  }
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << arr[0];
  for (size_t i = 1; i < arr.size(); ++i) {
    str_buf << delimiter;
    str_buf << arr[i];
  }
  return str_buf.str();
}

template<typename T>
inline static std::string ArrayToString(const std::vector<T>& arr, size_t n, char delimiter) {
  if (arr.empty() || n == 0) {
    return std::string("");
  }
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << arr[0];
  for (size_t i = 1; i < std::min(n, arr.size()); ++i) {
    str_buf << delimiter;
    str_buf << arr[i];
  }
  return str_buf.str();
}

template<typename T>
inline static std::vector<T> StringToArray(const std::string& str, char delimiter, size_t n) {
  std::vector<std::string> strs = Split(str.c_str(), delimiter);
  if (strs.size() != n) {
    Log::Fatal("StringToArray error, size doesn't match.");
  }
  std::vector<T> ret(n);
  if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
    for (size_t i = 0; i < n; ++i) {
      ret[i] = static_cast<T>(std::stod(strs[i]));
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      ret[i] = static_cast<T>(std::stol(strs[i]));
    }
  }
  return ret;
}

template<typename T>
inline static std::vector<T> StringToArray(const std::string& str, char delimiter) {
  std::vector<std::string> strs = Split(str.c_str(), delimiter);
  std::vector<T> ret;
  if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
    for (const auto& s : strs) {
      ret.push_back(static_cast<T>(std::stod(s)));
    }
  } else {
    for (const auto& s : strs) {
      ret.push_back(static_cast<T>(std::stol(s)));
    }
  }
  return ret;
}

template<typename T>
inline static std::string Join(const std::vector<T>& strs, const char* delimiter) {
  if (strs.empty()) {
    return std::string("");
  }
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << strs[0];
  for (size_t i = 1; i < strs.size(); ++i) {
    str_buf << delimiter;
    str_buf << strs[i];
  }
  return str_buf.str();
}

template<typename T>
inline static std::string Join(const std::vector<T>& strs, size_t start, size_t end, const char* delimiter) {
  if (end - start <= 0) {
    return std::string("");
  }
  start = std::min(start, static_cast<size_t>(strs.size()) - 1);
  end = std::min(end, static_cast<size_t>(strs.size()));
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << strs[start];
  for (size_t i = start + 1; i < end; ++i) {
    str_buf << delimiter;
    str_buf << strs[i];
  }
  return str_buf.str();
}

static inline int64_t Pow2RoundUp(int64_t x) {
  int64_t t = 1;
  for (int i = 0; i < 64; ++i) {
    if (t >= x) {
      return t;
    }
    t <<= 1;
  }
  return 0;
}

/*!
 * \brief Do inplace softmax transformaton on p_rec
 * \param p_rec The input/output vector of the values.
 */
inline void Softmax(std::vector<double>* p_rec) {
  std::vector<double> &rec = *p_rec;
  double wmax = rec[0];
  for (size_t i = 1; i < rec.size(); ++i) {
    wmax = std::max(rec[i], wmax);
  }
  double wsum = 0.0f;
  for (size_t i = 0; i < rec.size(); ++i) {
    rec[i] = std::exp(rec[i] - wmax);
    wsum += rec[i];
  }
  for (size_t i = 0; i < rec.size(); ++i) {
    rec[i] /= static_cast<double>(wsum);
  }
}

template<typename T>
std::vector<const T*> ConstPtrInVectorWrapper(const std::vector<std::unique_ptr<T>>& input) {
  std::vector<const T*> ret;
  for (size_t i = 0; i < input.size(); ++i) {
    ret.push_back(input.at(i).get());
  }
  return ret;
}

template<typename T1, typename T2>
inline void SortForPair(std::vector<T1>& keys, std::vector<T2>& values, size_t start, bool is_reverse = false) {
  std::vector<std::pair<T1, T2>> arr;
  for (size_t i = start; i < keys.size(); ++i) {
    arr.emplace_back(keys[i], values[i]);
  }
  if (!is_reverse) {
    std::sort(arr.begin(), arr.end(), [](const std::pair<T1, T2>& a, const std::pair<T1, T2>& b) {
      return a.first < b.first;
    });
  } else {
    std::sort(arr.begin(), arr.end(), [](const std::pair<T1, T2>& a, const std::pair<T1, T2>& b) {
      return a.first > b.first;
    });
  }
  for (size_t i = start; i < arr.size(); ++i) {
    keys[i] = arr[i].first;
    values[i] = arr[i].second;
  }

}

}  // namespace Common

}  // namespace LightGBM

#endif   // LightGBM_UTILS_COMMON_FUN_H_
