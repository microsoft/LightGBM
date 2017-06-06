#ifndef LIGHTGBM_UTILS_COMMON_FUN_H_
#define LIGHTGBM_UTILS_COMMON_FUN_H_

#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <iterator>
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

template<typename T, bool is_float>
struct __StringToTHelper {
  T operator()(const std::string& str) const {
    return static_cast<T>(std::stol(str));
  }
};

template<typename T>
struct __StringToTHelper<T, true> {
  T operator()(const std::string& str) const {
    return static_cast<T>(std::stod(str));
  }
};

template<typename T>
inline static std::vector<T> StringToArray(const std::string& str, char delimiter, size_t n) {
  if (n == 0) {
    return std::vector<T>();
  }
  std::vector<std::string> strs = Split(str.c_str(), delimiter);
  if (strs.size() != n) {
    Log::Fatal("StringToArray error, size doesn't match.");
  }
  std::vector<T> ret(n);
  __StringToTHelper<T, std::is_floating_point<T>::value> helper;
  for (size_t i = 0; i < n; ++i) {
    ret[i] = helper(strs[i]);
  }
  return ret;
}

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

inline void Softmax(const double* input, double* output, int len) {
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

/*
* approximate hessians of absolute loss with Gaussian function
* cf. https://en.wikipedia.org/wiki/Gaussian_function
*
* y is a prediction.
* t means true target.
* g means gradient.
* eta is a parameter to control the width of Gaussian function.
* w means weights.
*/
inline static double ApproximateHessianWithGaussian(const double y, const double t, const double g,
                                                    const double eta, const double w=1.0f) {
  const double diff = y - t;
  const double pi = 4.0 * std::atan(1.0);
  const double x = std::fabs(diff);
  const double a = 2.0 * std::fabs(g) * w;  // difference of two first derivatives, (zero to inf) and (zero to -inf).
  const double b = 0.0;
  const double c = std::max((std::fabs(y) + std::fabs(t)) * eta, 1.0e-10);
  return w * std::exp(-(x - b) * (x - b) / (2.0 * c * c)) * a / (c * std::sqrt(2 * pi));
}

template <typename T>
inline static std::vector<T*> Vector2Ptr(std::vector<std::vector<T>>& data) {
  std::vector<T*> ptr(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    ptr[i] = data[i].data();
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
  if (x >= std::numeric_limits<double>::max()) {
    return std::numeric_limits<double>::max();
  } else if(x <= std::numeric_limits<double>::lowest()) {
    return std::numeric_limits<double>::lowest();
  } else {
    return x;
  }
}

template<class _Iter> inline
static typename std::iterator_traits<_Iter>::value_type* IteratorValType(_Iter) {
  return (0);
}

template<class _RanIt, class _Pr, class _VTRanIt> inline
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
  #pragma omp parallel for schedule(static,1)
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
    #pragma omp parallel for schedule(static,1)
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

template<class _RanIt, class _Pr> inline
static void ParallelSort(_RanIt _First, _RanIt _Last, _Pr _Pred) {
  return ParallelSort(_First, _Last, _Pred, IteratorValType(_First));
}

}  // namespace Common

}  // namespace LightGBM

#endif   // LightGBM_UTILS_COMMON_FUN_H_
