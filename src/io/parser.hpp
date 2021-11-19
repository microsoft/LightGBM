/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_PARSER_HPP_
#define LIGHTGBM_IO_PARSER_HPP_

#include <LightGBM/dataset.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace LightGBM {

// define a str2num map class
class Str2Num {
public:

  Str2Num(){
    current = 0;
  }

  std::map<std::string, double> Str2NumMap;
  double current;
};
std::vector<Str2Num> maps;

const char* Atof_and_map(const char* p, double* out, int idx) {
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

  size_t flag = 0;
  size_t count = 0;
  while(*(p + count) != '\0' && *(p + count) != ' '
           && *(p + count) != '\t' && *(p + count) != ','
           && *(p + count) != '\n' && *(p + count) != '\r'
           && *(p + count) != ':'){
             flag = 1;
             if(*(p+count) >= '0' && *(p+count) <= '9')
              flag = 0;

              count++;
           }


  // is a number
  if (!flag && ( (*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' )  ) {
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
      value += right / Common::Pow(10.0, nn);
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


     if(maps.size() <= idx){
          Str2Num temp;
          maps.push_back(temp);
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
        // when is a string not a float, map it to float
        if(maps.size() <= idx){
          Str2Num temp;
          temp.Str2NumMap.insert(std::pair < std::string , double > (tmp_str, temp.current) );
          temp.current += 1.0;
          maps.push_back(temp);
        }else{
          if(maps[idx].Str2NumMap.count(tmp_str) == 1){
            // the key already exits
            *out = maps[idx].Str2NumMap[tmp_str];
          }else{
            maps[idx].Str2NumMap.insert(std::pair < std::string , double > (tmp_str, maps[idx].current) );
            maps[idx].current += 1.0;
          }
          
        }
        
        // Log::Fatal("Unknown token %s in data file", tmp_str.c_str());
      }
      p += cnt;
    }
  }
  return p;
}

class CSVParser: public Parser {
 public:
  explicit CSVParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0f;
    int offset = 0;
    *out_label = 0.0f;
    while (*str != '\0') {
      // map str to number
      str = Atof_and_map(str, &val, idx);
      // str = atof_(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        offset = -1;
      } else if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
        out_features->emplace_back(idx + offset, val);
      }
      ++idx;
      if (*str == ',') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("Input format error when parsing as CSV");
      }
    }
  }

  inline int NumFeatures() const override {
    return total_columns_ - (label_idx_ >= 0);
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

class TSVParser: public Parser {
 public:
  explicit TSVParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0f;
    int offset = 0;
    while (*str != '\0') {
      str = atof_(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        offset = -1;
      } else if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
        out_features->emplace_back(idx + offset, val);
      }
      ++idx;
      if (*str == '\t') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("Input format error when parsing as TSV");
      }
    }
  }

  inline int NumFeatures() const override {
    return total_columns_ - (label_idx_ >= 0);
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

class LibSVMParser: public Parser {
 public:
  explicit LibSVMParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
    if (label_idx > 0) {
      Log::Fatal("Label should be the first column in a LibSVM file");
    }
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0f;
    if (label_idx_ == 0) {
      str = atof_(str, &val);
      *out_label = val;
      str = Common::SkipSpaceAndTab(str);
    }
    while (*str != '\0') {
      str = Common::Atoi(str, &idx);
      str = Common::SkipSpaceAndTab(str);
      if (*str == ':') {
        ++str;
        str = Common::Atof(str, &val);
        out_features->emplace_back(idx, val);
      } else {
        Log::Fatal("Input format error when parsing as LibSVM");
      }
      str = Common::SkipSpaceAndTab(str);
    }
  }

  inline int NumFeatures() const override {
    return total_columns_;
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

}  // namespace LightGBM
#endif   // LightGBM_IO_PARSER_HPP_
