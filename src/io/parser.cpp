#include "parser.hpp"

#include <iostream>
#include <fstream>
#include <functional>
#include <memory>

namespace LightGBM {

void GetStatistic(const char* str, int* comma_cnt, int* tab_cnt, int* colon_cnt) {
  *comma_cnt = 0;
  *tab_cnt = 0;
  *colon_cnt = 0;
  for (int i = 0; str[i] != '\0'; ++i) {
    if (str[i] == ',') {
      ++(*comma_cnt);
    } else if (str[i] == '\t') {
      ++(*tab_cnt);
    } else if (str[i] == ':') {
      ++(*colon_cnt);
    }
  }
}

int GetLabelIdxForLibsvm(std::string& str, int num_features, int label_idx) {
  if (num_features <= 0) {
    return label_idx;
  }
  str = Common::Trim(str);
  auto pos_space = str.find_first_of(" \f\n\r\t\v");
  auto pos_colon = str.find_first_of(":");
  if (pos_space == std::string::npos || pos_space < pos_colon) {
    return label_idx;
  } else {
    return -1;
  }
}

int GetLabelIdxForTSV(std::string& str, int num_features, int label_idx) {
  if (num_features <= 0) {
    return label_idx;
  }
  str = Common::Trim(str);
  auto tokens = Common::Split(str.c_str(), '\t');
  if (static_cast<int>(tokens.size()) == num_features) {
    return -1;
  } else {
    return label_idx;
  }
}

int GetLabelIdxForCSV(std::string& str, int num_features, int label_idx) {
  if (num_features <= 0) {
    return label_idx;
  }
  str = Common::Trim(str);
  auto tokens = Common::Split(str.c_str(), ',');
  if (static_cast<int>(tokens.size()) == num_features) {
    return -1;
  } else {
    return label_idx;
  }
}

enum DataType {
  INVALID,
  CSV,
  TSV,
  LIBSVM
};

Parser* Parser::CreateParser(const char* filename, bool has_header, int num_features, int label_idx) {
  std::ifstream tmp_file;
  tmp_file.open(filename);
  if (!tmp_file.is_open()) {
    Log::Fatal("Data file %s doesn't exist'", filename);
  }
  std::string line1, line2;
  if (has_header) {
    if (!tmp_file.eof()) {
      std::getline(tmp_file, line1);
    }
  }
  if (!tmp_file.eof()) {
    std::getline(tmp_file, line1);
  } else {
    Log::Fatal("Data file %s should have at least one line", filename);
  }
  if (!tmp_file.eof()) {
    std::getline(tmp_file, line2);
  } else {
    Log::Warning("Data file %s only has one line", filename);
  }
  tmp_file.close();
  int comma_cnt = 0, comma_cnt2 = 0;
  int tab_cnt = 0, tab_cnt2 = 0;
  int colon_cnt = 0, colon_cnt2 = 0;
  // Get some statistic from 2 line
  GetStatistic(line1.c_str(), &comma_cnt, &tab_cnt, &colon_cnt);
  GetStatistic(line2.c_str(), &comma_cnt2, &tab_cnt2, &colon_cnt2);



  DataType type = DataType::INVALID;
  if (line2.size() == 0) {
    // if only have one line on file
    if (colon_cnt > 0) {
      type = DataType::LIBSVM;
    } else if (tab_cnt > 0) {
      type = DataType::TSV;
    } else if (comma_cnt > 0) {
      type = DataType::CSV;
    }
  } else {
    if (colon_cnt > 0 || colon_cnt2 > 0) {
      type = DataType::LIBSVM;
    } else if (tab_cnt == tab_cnt2 && tab_cnt > 0) {
      type = DataType::TSV;
    } else if (comma_cnt == comma_cnt2 && comma_cnt > 0) {
      type = DataType::CSV;
    }
  }
  if (type == DataType::INVALID) {
    Log::Fatal("Unknown format of training data");
  }
  std::unique_ptr<Parser> ret;
  if (type == DataType::LIBSVM) {
    label_idx = GetLabelIdxForLibsvm(line1, num_features, label_idx);
    ret.reset(new LibSVMParser(label_idx));
  }
  else if (type == DataType::TSV) {
    label_idx = GetLabelIdxForTSV(line1, num_features, label_idx);
    ret.reset(new TSVParser(label_idx));
  }
  else if (type == DataType::CSV) {
    label_idx = GetLabelIdxForCSV(line1, num_features, label_idx);
    ret.reset(new CSVParser(label_idx));
  }

  if (label_idx < 0) {
    Log::Info("Data file %s doesn't contain a label column", filename);
  }
  return ret.release();
}

}  // namespace LightGBM
