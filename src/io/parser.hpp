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
  inline void ParseOneLine(const char* str,
                           std::vector<std::pair<int, double>>* out_features) const override {
    int idx = 0;
    double val = 0.0;
    while (*str != '\0') {
      str = Common::Atof(str, &val);
      out_features->emplace_back(idx, val);
      ++idx;
      if (*str == ',') {
        ++str;
      } else if (*str != '\0') {
        Log::Stderr("input format error, should be CSV");
      }
    }
  }
  inline void ParseOneLine(const char* str, std::vector<std::pair<int, double>>* out_features,
                                                           double* out_label) const override {
    // first column is label
    str = Common::Atof(str, out_label);

	if (*str == ',') {
		++str;
	} else if (*str != '\0') {
		Log::Stderr("input format error, should be CSV");
	}

    return ParseOneLine(str, out_features);
  }
};

class TSVParser: public Parser {
public:
  inline void ParseOneLine(const char* str, std::vector<std::pair<int, double>>* out_features) const override {
    int idx = 0;
    double val = 0.0;
    while (*str != '\0') {
      str = Common::Atof(str, &val);
      out_features->emplace_back(idx, val);
      ++idx;
      if (*str == '\t') {
        ++str;
      } else if (*str != '\0') {
        Log::Stderr("input format error, should be TSV");
      }
    }
  }
  inline void ParseOneLine(const char* str, std::vector<std::pair<int, double>>* out_features,
                                                           double* out_label) const override{
    // first column is label
    str = Common::Atof(str, out_label);

	if (*str == '\t') {
		++str;
	} else if (*str != '\0') {
		Log::Stderr("input format error, should be TSV");
	}

    return ParseOneLine(str, out_features);
  }
};

class LibSVMParser: public Parser {
public:
  inline void ParseOneLine(const char* str, std::vector<std::pair<int, double>>* out_features) const override {
    int idx = 0;
    double val = 0.0;
    while (*str != '\0') {
      str = Common::Atoi(str, &idx);
      str = Common::SkipSpaceAndTab(str);
      if (*str == ':') {
        ++str;
        str = Common::Atof(str, &val);
        out_features->emplace_back(idx, val);
      } else {
        Log::Stderr("input format error, should be LibSVM");
      }
      str = Common::SkipSpaceAndTab(str);
    }
  }
  inline void ParseOneLine(const char* str, std::vector<std::pair<int, double>>* out_features,
                                                            double* out_label) const override{
    // first column is label
    str = Common::Atof(str, out_label);
    str = Common::SkipSpaceAndTab(str);
    return ParseOneLine(str, out_features);
  }
};
}  // namespace LightGBM
#endif  #endif  // LightGBM_IO_PARSER_HPP_
