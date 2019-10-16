/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_UTILS_COMMON_FUN_H_
#define LIGHTGBM_UTILS_COMMON_FUN_H_

#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <limits>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#include "intrin.h"
#endif

namespace LightGBM {

namespace Common {

inline static char tolower(char in) {
  if (in <= 'Z' && in >= 'A')
    return in - ('Z' - 'z');
  return in;
}

inline static std::string Trim(std::string str) {
  if (str.empty()) {
    return str;
  }
  str.erase(str.find_last_not_of(" \f\n\r\t\v") + 1);
  str.erase(0, str.find_first_not_of(" \f\n\r\t\v"));
  return str;
}

inline static std::string RemoveQuotationSymbol(std::string str) {
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
  size_t pos = 0;
  while (pos < str.length()) {
    if (str[pos] == delimiter) {
      if (i < pos) {
        ret.push_back(str.substr(i, pos - i));
      }
      ++pos;
      i = pos;
    } else {
      ++pos;
    }
  }
  if (i < pos) {
    ret.push_back(str.substr(i));
  }
  return ret;
}

inline static std::vector<std::string> SplitLines(const char* c_str) {
  std::vector<std::string> ret;
  std::string str(c_str);
  size_t i = 0;
  size_t pos = 0;
  while (pos < str.length()) {
    if (str[pos] == '\n' || str[pos] == '\r') {
      if (i < pos) {
        ret.push_back(str.substr(i, pos - i));
      }
      // skip the line endings
      while (str[pos] == '\n' || str[pos] == '\r') ++pos;
      // new begin
      i = pos;
    } else {
      ++pos;
    }
  }
  if (i < pos) {
    ret.push_back(str.substr(i));
  }
  return ret;
}

inline static std::vector<std::string> Split(const char* c_str, const char* delimiters) {
  std::vector<std::string> ret;
  std::string str(c_str);
  size_t i = 0;
  size_t pos = 0;
  while (pos < str.length()) {
    bool met_delimiters = false;
    for (int j = 0; delimiters[j] != '\0'; ++j) {
      if (str[pos] == delimiters[j]) {
        met_delimiters = true;
        break;
      }
    }
    if (met_delimiters) {
      if (i < pos) {
        ret.push_back(str.substr(i, pos - i));
      }
      ++pos;
      i = pos;
    } else {
      ++pos;
    }
  }
  if (i < pos) {
    ret.push_back(str.substr(i));
  }
  return ret;
}

template<typename T>
inline static const char* Atoi(const char* p, T* out) {
  int sign;
  T value;
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
  *out = static_cast<T>(sign * value);
  while (*p == ' ') {
    ++p;
  }
  return p;
}

template<typename T>
inline static double Pow(T base, int power) {
  if (power < 0) {
    return 1.0 / Pow(base, -power);
  } else if (power == 0) {
    return 1;
  } else if (power % 2 == 0) {
    return Pow(base*base, power / 2);
  } else if (power % 3 == 0) {
    return Pow(base*base*base, power / 3);
  } else {
    return base * Pow(base, power - 1);
  }
}

inline static const char* Atof(const char* p, double* out) {
  int frac;
  double sign, value, scale;
  *out = NAN;
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
      double right = 0.0;
      int nn = 0;
      ++p;
      while (*p >= '0' && *p <= '9') {
        right = (*p - '0') + right * 10.0;
        ++nn;
        ++p;
      }
      value += right / Pow(10.0, nn);
    }

    // Handle exponent, if any.
    frac = 0;
    scale = 1.0;
    if ((*p == 'e') || (*p == 'E')) {
      uint32_t expon;
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
      if (tmp_str == std::string("na") || tmp_str == std::string("nan") ||
          tmp_str == std::string("null")) {
        *out = NAN;
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

inline static bool AtoiAndCheck(const char* p, int* out) {
  const char* after = Atoi(p, out);
  if (*after != '\0') {
    return false;
  }
  return true;
}

inline static bool AtofAndCheck(const char* p, double* out) {
  const char* after = Atof(p, out);
  if (*after != '\0') {
    return false;
  }
  return true;
}

inline static unsigned CountDecimalDigit32(uint32_t n) {
#if defined(_MSC_VER) || defined(__GNUC__)
  static const uint32_t powers_of_10[] = {
    0,
    10,
    100,
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000,
    1000000000
  };
#ifdef _MSC_VER
  unsigned long i = 0;
  _BitScanReverse(&i, n | 1);
  uint32_t t = (i + 1) * 1233 >> 12;
#elif __GNUC__
  uint32_t t = (32 - __builtin_clz(n | 1)) * 1233 >> 12;
#endif
  return t - (n < powers_of_10[t]) + 1;
#else
  if (n < 10) return 1;
  if (n < 100) return 2;
  if (n < 1000) return 3;
  if (n < 10000) return 4;
  if (n < 100000) return 5;
  if (n < 1000000) return 6;
  if (n < 10000000) return 7;
  if (n < 100000000) return 8;
  if (n < 1000000000) return 9;
  return 10;
#endif
}

inline static void Uint32ToStr(uint32_t value, char* buffer) {
  const char kDigitsLut[200] = {
    '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6', '0', '7', '0', '8', '0', '9',
    '1', '0', '1', '1', '1', '2', '1', '3', '1', '4', '1', '5', '1', '6', '1', '7', '1', '8', '1', '9',
    '2', '0', '2', '1', '2', '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7', '2', '8', '2', '9',
    '3', '0', '3', '1', '3', '2', '3', '3', '3', '4', '3', '5', '3', '6', '3', '7', '3', '8', '3', '9',
    '4', '0', '4', '1', '4', '2', '4', '3', '4', '4', '4', '5', '4', '6', '4', '7', '4', '8', '4', '9',
    '5', '0', '5', '1', '5', '2', '5', '3', '5', '4', '5', '5', '5', '6', '5', '7', '5', '8', '5', '9',
    '6', '0', '6', '1', '6', '2', '6', '3', '6', '4', '6', '5', '6', '6', '6', '7', '6', '8', '6', '9',
    '7', '0', '7', '1', '7', '2', '7', '3', '7', '4', '7', '5', '7', '6', '7', '7', '7', '8', '7', '9',
    '8', '0', '8', '1', '8', '2', '8', '3', '8', '4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9',
    '9', '0', '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9', '7', '9', '8', '9', '9'
  };
  unsigned digit = CountDecimalDigit32(value);
  buffer += digit;
  *buffer = '\0';

  while (value >= 100) {
    const unsigned i = (value % 100) << 1;
    value /= 100;
    *--buffer = kDigitsLut[i + 1];
    *--buffer = kDigitsLut[i];
  }

  if (value < 10) {
    *--buffer = static_cast<char>(value) + '0';
  } else {
    const unsigned i = value << 1;
    *--buffer = kDigitsLut[i + 1];
    *--buffer = kDigitsLut[i];
  }
}

inline static void Int32ToStr(int32_t value, char* buffer) {
  uint32_t u = static_cast<uint32_t>(value);
  if (value < 0) {
    *buffer++ = '-';
    u = ~u + 1;
  }
  Uint32ToStr(u, buffer);
}

inline static void DoubleToStr(double value, char* buffer, size_t
                               #ifdef _MSC_VER
                               buffer_len
                               #endif
) {
  #ifdef _MSC_VER
  sprintf_s(buffer, buffer_len, "%.17g", value);
  #else
  sprintf(buffer, "%.17g", value);
  #endif
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
  std::vector<T2> ret(arr.size());
  for (size_t i = 0; i < arr.size(); ++i) {
    ret[i] = static_cast<T2>(arr[i]);
  }
  return ret;
}

template<typename T, bool is_float, bool is_unsign>
struct __TToStringHelperFast {
  void operator()(T value, char* buffer, size_t) const {
    Int32ToStr(value, buffer);
  }
};

template<typename T>
struct __TToStringHelperFast<T, true, false> {
  void operator()(T value, char* buffer, size_t
                  #ifdef _MSC_VER
                  buf_len
                  #endif
                  ) const {
    #ifdef _MSC_VER
    sprintf_s(buffer, buf_len, "%g", value);
    #else
    sprintf(buffer, "%g", value);
    #endif
  }
};

template<typename T>
struct __TToStringHelperFast<T, false, true> {
  void operator()(T value, char* buffer, size_t) const {
    Uint32ToStr(value, buffer);
  }
};

template<typename T>
inline static std::string ArrayToStringFast(const std::vector<T>& arr, size_t n) {
  if (arr.empty() || n == 0) {
    return std::string("");
  }
  __TToStringHelperFast<T, std::is_floating_point<T>::value, std::is_unsigned<T>::value> helper;
  const size_t buf_len = 16;
  std::vector<char> buffer(buf_len);
  std::stringstream str_buf;
  helper(arr[0], buffer.data(), buf_len);
  str_buf << buffer.data();
  for (size_t i = 1; i < std::min(n, arr.size()); ++i) {
    helper(arr[i], buffer.data(), buf_len);
    str_buf << ' ' << buffer.data();
  }
  return str_buf.str();
}

inline static std::string ArrayToString(const std::vector<double>& arr, size_t n) {
  if (arr.empty() || n == 0) {
    return std::string("");
  }
  const size_t buf_len = 32;
  std::vector<char> buffer(buf_len);
  std::stringstream str_buf;
  DoubleToStr(arr[0], buffer.data(), buf_len);
  str_buf << buffer.data();
  for (size_t i = 1; i < std::min(n, arr.size()); ++i) {
    DoubleToStr(arr[i], buffer.data(), buf_len);
    str_buf << ' ' << buffer.data();
  }
  return str_buf.str();
}

template<typename T, bool is_float>
struct __StringToTHelper {
  T operator()(const std::string& str) const {
    T ret = 0;
    Atoi(str.c_str(), &ret);
    return ret;
  }
};

template<typename T>
struct __StringToTHelper<T, true> {
  T operator()(const std::string& str) const {
    return static_cast<T>(std::stod(str));
  }
};

template<typename T>
inline static std::vector<T> StringToArray(const std::string& str, char delimiter) {
  std::vector<std::string> strs = Split(str.c_str(), delimiter);
  std::vector<T> ret;
  ret.reserve(strs.size());
  __StringToTHelper<T, std::is_floating_point<T>::value> helper;
  for (const auto& s : strs) {
    ret.push_back(helper(s));
  }
  return ret;
}

template<typename T>
inline static std::vector<T> StringToArray(const std::string& str, int n) {
  if (n == 0) {
    return std::vector<T>();
  }
  std::vector<std::string> strs = Split(str.c_str(), ' ');
  CHECK(strs.size() == static_cast<size_t>(n));
  std::vector<T> ret;
  ret.reserve(strs.size());
  __StringToTHelper<T, std::is_floating_point<T>::value> helper;
  for (const auto& s : strs) {
    ret.push_back(helper(s));
  }
  return ret;
}

template<typename T, bool is_float>
struct __StringToTHelperFast {
  const char* operator()(const char*p, T* out) const {
    return Atoi(p, out);
  }
};

template<typename T>
struct __StringToTHelperFast<T, true> {
  const char* operator()(const char*p, T* out) const {
    double tmp = 0.0f;
    auto ret = Atof(p, &tmp);
    *out = static_cast<T>(tmp);
    return ret;
  }
};

template<typename T>
inline static std::vector<T> StringToArrayFast(const std::string& str, int n) {
  if (n == 0) {
    return std::vector<T>();
  }
  auto p_str = str.c_str();
  __StringToTHelperFast<T, std::is_floating_point<T>::value> helper;
  std::vector<T> ret(n);
  for (int i = 0; i < n; ++i) {
    p_str = helper(p_str, &ret[i]);
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

template<>
inline std::string Join<int8_t>(const std::vector<int8_t>& strs, const char* delimiter) {
  if (strs.empty()) {
    return std::string("");
  }
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << static_cast<int16_t>(strs[0]);
  for (size_t i = 1; i < strs.size(); ++i) {
    str_buf << delimiter;
    str_buf << static_cast<int16_t>(strs[i]);
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

inline static int64_t Pow2RoundUp(int64_t x) {
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
 * \brief Do inplace softmax transformation on p_rec
 * \param p_rec The input/output vector of the values.
 */
inline static void Softmax(std::vector<double>* p_rec) {
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

inline static void Softmax(const double* input, double* output, int len) {
  double wmax = input[0];
  for (int i = 1; i < len; ++i) {
    wmax = std::max(input[i], wmax);
  }
  double wsum = 0.0f;
  for (int i = 0; i < len; ++i) {
    output[i] = std::exp(input[i] - wmax);
    wsum += output[i];
  }
  for (int i = 0; i < len; ++i) {
    output[i] /= static_cast<double>(wsum);
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
inline static void SortForPair(std::vector<T1>* keys, std::vector<T2>* values, size_t start, bool is_reverse = false) {
  std::vector<std::pair<T1, T2>> arr;
  for (size_t i = start; i < keys->size(); ++i) {
    arr.emplace_back(keys->at(i), values->at(i));
  }
  if (!is_reverse) {
    std::stable_sort(arr.begin(), arr.end(), [](const std::pair<T1, T2>& a, const std::pair<T1, T2>& b) {
      return a.first < b.first;
    });
  } else {
    std::stable_sort(arr.begin(), arr.end(), [](const std::pair<T1, T2>& a, const std::pair<T1, T2>& b) {
      return a.first > b.first;
    });
  }
  for (size_t i = start; i < arr.size(); ++i) {
    keys->at(i) = arr[i].first;
    values->at(i) = arr[i].second;
  }
}

template <typename T>
inline static std::vector<T*> Vector2Ptr(std::vector<std::vector<T>>* data) {
  std::vector<T*> ptr(data->size());
  for (size_t i = 0; i < data->size(); ++i) {
    ptr[i] = data->at(i).data();
  }
  return ptr;
}

template <typename T>
inline static std::vector<int> VectorSize(const std::vector<std::vector<T>>& data) {
  std::vector<int> ret(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    ret[i] = static_cast<int>(data[i].size());
  }
  return ret;
}

inline static double AvoidInf(double x) {
  if (std::isnan(x)) {
    return 0.0;
  } else if (x >= 1e300) {
    return 1e300;
  } else if (x <= -1e300) {
    return -1e300;
  } else {
    return x;
  }
}

inline static float AvoidInf(float x) {
  if (std::isnan(x)) {
    return 0.0f;
  } else if (x >= 1e38) {
    return 1e38f;
  } else if (x <= -1e38) {
    return -1e38f;
  } else {
    return x;
  }
}

template<typename _Iter> inline
static typename std::iterator_traits<_Iter>::value_type* IteratorValType(_Iter) {
  return (0);
}

template<typename _RanIt, typename _Pr, typename _VTRanIt> inline
static void ParallelSort(_RanIt _First, _RanIt _Last, _Pr _Pred, _VTRanIt*) {
  size_t len = _Last - _First;
  const size_t kMinInnerLen = 1024;
  int num_threads = 1;
  #pragma omp parallel
  #pragma omp master
  {
    num_threads = omp_get_num_threads();
  }
  if (len <= kMinInnerLen || num_threads <= 1) {
    std::sort(_First, _Last, _Pred);
    return;
  }
  size_t inner_size = (len + num_threads - 1) / num_threads;
  inner_size = std::max(inner_size, kMinInnerLen);
  num_threads = static_cast<int>((len + inner_size - 1) / inner_size);
  #pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < num_threads; ++i) {
    size_t left = inner_size*i;
    size_t right = left + inner_size;
    right = std::min(right, len);
    if (right > left) {
      std::sort(_First + left, _First + right, _Pred);
    }
  }
  // Buffer for merge.
  std::vector<_VTRanIt> temp_buf(len);
  _RanIt buf = temp_buf.begin();
  size_t s = inner_size;
  // Recursive merge
  while (s < len) {
    int loop_size = static_cast<int>((len + s * 2 - 1) / (s * 2));
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < loop_size; ++i) {
      size_t left = i * 2 * s;
      size_t mid = left + s;
      size_t right = mid + s;
      right = std::min(len, right);
      if (mid >= right) { continue; }
      std::copy(_First + left, _First + mid, buf + left);
      std::merge(buf + left, buf + mid, _First + mid, _First + right, _First + left, _Pred);
    }
    s *= 2;
  }
}

template<typename _RanIt, typename _Pr> inline
static void ParallelSort(_RanIt _First, _RanIt _Last, _Pr _Pred) {
  return ParallelSort(_First, _Last, _Pred, IteratorValType(_First));
}

// Check that all y[] are in interval [ymin, ymax] (end points included); throws error if not
template <typename T>
inline static void CheckElementsIntervalClosed(const T *y, T ymin, T ymax, int ny, const char *callername) {
  auto fatal_msg = [&y, &ymin, &ymax, &callername](int i) {
    std::ostringstream os;
    os << "[%s]: does not tolerate element [#%i = " << y[i] << "] outside [" << ymin << ", " << ymax << "]";
    Log::Fatal(os.str().c_str(), callername, i);
  };
  for (int i = 1; i < ny; i += 2) {
    if (y[i - 1] < y[i]) {
      if (y[i - 1] < ymin) {
        fatal_msg(i - 1);
      } else if (y[i] > ymax) {
        fatal_msg(i);
      }
    } else {
      if (y[i - 1] > ymax) {
        fatal_msg(i - 1);
      } else if (y[i] < ymin) {
        fatal_msg(i);
      }
    }
  }
  if (ny & 1) {  // odd
    if (y[ny - 1] < ymin || y[ny - 1] > ymax) {
      fatal_msg(ny - 1);
    }
  }
}

// One-pass scan over array w with nw elements: find min, max and sum of elements;
// this is useful for checking weight requirements.
template <typename T1, typename T2>
inline static void ObtainMinMaxSum(const T1 *w, int nw, T1 *mi, T1 *ma, T2 *su) {
  T1 minw;
  T1 maxw;
  T1 sumw;
  int i;
  if (nw & 1) {  // odd
    minw = w[0];
    maxw = w[0];
    sumw = w[0];
    i = 2;
  } else {  // even
    if (w[0] < w[1]) {
      minw = w[0];
      maxw = w[1];
    } else {
      minw = w[1];
      maxw = w[0];
    }
    sumw = w[0] + w[1];
    i = 3;
  }
  for (; i < nw; i += 2) {
    if (w[i - 1] < w[i]) {
      minw = std::min(minw, w[i - 1]);
      maxw = std::max(maxw, w[i]);
    } else {
      minw = std::min(minw, w[i]);
      maxw = std::max(maxw, w[i - 1]);
    }
    sumw += w[i - 1] + w[i];
  }
  if (mi != nullptr) {
    *mi = minw;
  }
  if (ma != nullptr) {
    *ma = maxw;
  }
  if (su != nullptr) {
    *su = static_cast<T2>(sumw);
  }
}

inline static std::vector<uint32_t> EmptyBitset(int n) {
  int size = n / 32;
  if (n % 32 != 0) ++size;
  return std::vector<uint32_t>(size);
}

template<typename T>
inline static void InsertBitset(std::vector<uint32_t>* vec, const T val) {
    int i1 = val / 32;
    int i2 = val % 32;
    if (static_cast<int>(vec->size()) < i1 + 1) {
      vec->resize(i1 + 1, 0);
    }
    vec->at(i1) |= (1 << i2);
}

template<typename T>
inline static std::vector<uint32_t> ConstructBitset(const T* vals, int n) {
  std::vector<uint32_t> ret;
  for (int i = 0; i < n; ++i) {
    int i1 = vals[i] / 32;
    int i2 = vals[i] % 32;
    if (static_cast<int>(ret.size()) < i1 + 1) {
      ret.resize(i1 + 1, 0);
    }
    ret[i1] |= (1 << i2);
  }
  return ret;
}

template<typename T>
inline static bool FindInBitset(const uint32_t* bits, int n, T pos) {
  int i1 = pos / 32;
  if (i1 >= n) {
    return false;
  }
  int i2 = pos % 32;
  return (bits[i1] >> i2) & 1;
}

inline static bool CheckDoubleEqualOrdered(double a, double b) {
  double upper = std::nextafter(a, INFINITY);
  return b <= upper;
}

inline static double GetDoubleUpperBound(double a) {
  return std::nextafter(a, INFINITY);;
}

inline static size_t GetLine(const char* str) {
  auto start = str;
  while (*str != '\0' && *str != '\n' && *str != '\r') {
    ++str;
  }
  return str - start;
}

inline static const char* SkipNewLine(const char* str) {
  if (*str == '\r') {
    ++str;
  }
  if (*str == '\n') {
    ++str;
  }
  return str;
}

template <typename T>
static int Sign(T x) {
  return (x > T(0)) - (x < T(0));
}

template <typename T>
static T SafeLog(T x) {
  if (x > 0) {
    return std::log(x);
  } else {
    return -INFINITY;
  }
}

inline bool CheckASCII(const std::string& s) {
  for (auto c : s) {
    if (static_cast<unsigned char>(c) > 127) {
      return false;
    }
  }
  return true;
}

}  // namespace Common

}  // namespace LightGBM

#endif   // LightGBM_UTILS_COMMON_FUN_H_
