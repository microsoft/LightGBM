#ifndef LIGHTGBM_UTILS_COMMON_FUN_H_
#define LIGHTGBM_UTILS_COMMON_FUN_H_

#include <LightGBM/utils/log.h>

#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <cstdint>
#include <algorithm>

namespace LightGBM {

namespace Common {

template<typename T>
inline static T Max(const T& a, const T& b) {
  return a > b ? a : b;
}

template<typename T>
inline static T Min(const T& a, const T& b) {
  return a < b ? a : b;
}



inline static std::string& Trim(std::string& str) {
  if (str.size() <= 0) {
    return str;
  }
  str.erase(str.find_last_not_of(" \f\n\r\t\v") + 1);
  str.erase(0, str.find_first_not_of(" \f\n\r\t\v"));
  return str;
}

inline static std::string& RemoveQuotationSymbol(std::string& str) {
  if (str.size() <= 0) {
    return str;
  }
  str.erase(str.find_last_not_of("'\"") + 1);
  str.erase(0, str.find_first_not_of("'\""));
  return str;
}

inline static std::vector<std::string> Split(const char* str, char delimiter) {
  std::stringstream ss(str);
  std::string tmp_str;
  std::vector<std::string> ret;
  while (std::getline(ss, tmp_str, delimiter)) {
    ret.push_back(tmp_str);
  }
  return ret;
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
  }
  else if (*p == '+') {
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

//ref to http://www.leapsecond.com/tools/fast_atof.c
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
  }
  else if (*p == '+') {
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
      && *(p + cnt) != ':')  {
      ++cnt;
    }
    if(cnt > 0){
      std::string tmp_str(p, cnt);
      std::transform(tmp_str.begin(), tmp_str.end(), tmp_str.begin(), ::tolower);
      if (tmp_str == std::string("na") || tmp_str == std::string("nan")) {
        *out = 0;
      } else if( tmp_str == std::string("inf") || tmp_str == std::string("infinity")) {
        *out = sign * 1e308;
      }
      else {
        Log::Error("Unknow token %s in data file", tmp_str.c_str());
      }
      p += cnt;
    }
  }

  while (*p == ' ') {
    ++p;
  }

  return p;
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

template<typename T>
inline static std::string ArrayToString(const T* arr, int n, char delimiter) {
  if (n <= 0) {
    return std::string("");
  }
  std::stringstream ss;
  ss << arr[0];
  for (int i = 1; i < n; ++i) {
    ss << delimiter;
    ss << arr[i];
  }
  return ss.str();
}

inline static void StringToIntArray(const std::string& str, char delimiter, size_t n, int* out) {
  std::vector<std::string> strs = Split(str.c_str(), delimiter);
  if (strs.size() != n) {
    Log::Error("StringToIntArray error, size don't equal.");
  }
  for (size_t i = 0; i < strs.size(); ++i) {
    strs[i] = Trim(strs[i]);
    Atoi(strs[i].c_str(), &out[i]);
  }
}

inline static void StringToDoubleArray(const std::string& str, char delimiter, size_t n, double* out) {
  std::vector<std::string> strs = Split(str.c_str(), delimiter);
  if (strs.size() != n) {
    Log::Error("StringToDoubleArray error, size don't equal");
  }
  for (size_t i = 0; i < strs.size(); ++i) {
    strs[i] = Trim(strs[i]);
    Atof(strs[i].c_str(), &out[i]);
  }
}

inline static void StringToDoubleArray(const std::string& str, char delimiter, size_t n, float* out) {
  std::vector<std::string> strs = Split(str.c_str(), delimiter);
  if (strs.size() != n) {
    Log::Error("StringToDoubleArray error, size don't equal");
  }
  double tmp;
  for (size_t i = 0; i < strs.size(); ++i) {
    strs[i] = Trim(strs[i]);
    Atof(strs[i].c_str(), &tmp);
    out[i] = static_cast<float>(tmp);
  }
}

inline static std::vector<double> StringToDoubleArray(const std::string& str, char delimiter) {
  std::vector<std::string> strs = Split(str.c_str(), delimiter);
  std::vector<double> ret;
  for (size_t i = 0; i < strs.size(); ++i) {
    strs[i] = Trim(strs[i]);
    double val = 0.0;
    Atof(strs[i].c_str(), &val);
    ret.push_back(val);
  }
  return ret;
}

inline static std::vector<int> StringToIntArray(const std::string& str, char delimiter) {
  std::vector<std::string> strs = Split(str.c_str(), delimiter);
  std::vector<int> ret;
  for (size_t i = 0; i < strs.size(); ++i) {
    strs[i] = Trim(strs[i]);
    int val = 0;
    Atoi(strs[i].c_str(), &val);
    ret.push_back(val);
  }
  return ret;
}

inline static std::string Join(const std::vector<std::string>& strs, char delimiter) {
  if (strs.size() <= 0) {
    return std::string("");
  }
  std::stringstream ss;
  ss << strs[0];
  for (size_t i = 1; i < strs.size(); ++i) {
    ss << delimiter;
    ss << strs[i];
  }
  return ss.str();
}

inline static std::string Join(const std::vector<std::string>& strs, size_t start, size_t end, char delimiter) {
  if (end - start <= 0) {
    return std::string("");
  }
  start = Min<size_t>(start, static_cast<size_t>(strs.size()) - 1);
  end = Min<size_t>(end, static_cast<size_t>(strs.size()));
  std::stringstream ss;
  ss << strs[start];
  for (size_t i = start + 1; i < end; ++i) {
    ss << delimiter;
    ss << strs[i];
  }
  return ss.str();
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

}  // namespace Common

}  // namespace LightGBM

#endif   // LightGBM_UTILS_COMMON_FUN_H_
