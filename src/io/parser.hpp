#ifndef LIGHTGBM_IO_PARSER_HPP_
#define LIGHTGBM_IO_PARSER_HPP_

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <LightGBM/dataset.h>

#include <unordered_map>
#include <vector>
#include <utility>

namespace LightGBM {

class CSVParser: public Parser {
public:
  explicit CSVParser(int label_idx)
    :label_idx_(label_idx) {
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0;
    int bias = 0;
    *out_label = 0.0f;
    while (*str != '\0') {
      str = Common::Atof(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        bias = -1;
      }
      else if (fabs(val) > 1e-10) {
        out_features->emplace_back(idx + bias, val);
      }
      ++idx;
      if (*str == ',') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("input format error, should be CSV");
      }
    }
  }
private:
  int label_idx_ = 0;
};

class TSVParser: public Parser {
public:
  explicit TSVParser(int label_idx)
    :label_idx_(label_idx) {
  }
  inline void ParseOneLine(const char* str, 
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0;
    int bias = 0;
    while (*str != '\0') {
      str = Common::Atof(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        bias = -1;
      } else if (fabs(val) > 1e-10) {
        out_features->emplace_back(idx + bias, val);
      }
      ++idx;
      if (*str == '\t') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("input format error, should be TSV");
      }
    }
  }
private:
  int label_idx_ = 0;
};

class LibSVMParser: public Parser {
public:
  explicit LibSVMParser(int label_idx)
    :label_idx_(label_idx) {
    if (label_idx > 0) {
      Log::Fatal("label should be the first column in Libsvm file");
    }
  }
  inline void ParseOneLine(const char* str, 
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0;
    if (label_idx_ == 0) {
      str = Common::Atof(str, &val);
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
        Log::Fatal("input format error, should be LibSVM");
      }
      str = Common::SkipSpaceAndTab(str);
    }
  }
private:
  int label_idx_ = 0;
};

}  // namespace LightGBM
#endif   // LightGBM_IO_PARSER_HPP_
