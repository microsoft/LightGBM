#include "parser.hpp"

#include <iostream>
#include <fstream>

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

bool CheckHasLabelForLibsvm(std::string& str) {
  str = Common::Trim(str);
  auto pos_space = str.find_first_of(" \f\n\r\t\v");
  auto pos_colon = str.find_first_of(":");
  if (pos_colon == std::string::npos || pos_colon > pos_space) {
    return true;
  } else {
    return false;
  }
}

bool CheckHasLabelForTSV(std::string& str, int num_features) {
  str = Common::Trim(str);
  auto tokens = Common::Split(str.c_str(), '\t');
  if (static_cast<int>(tokens.size()) == num_features) {
    return false;
  } else {
    return true;
  }
}

bool CheckHasLabelForCSV(std::string& str, int num_features) {
  str = Common::Trim(str);
  auto tokens = Common::Split(str.c_str(), ',');
  if (static_cast<int>(tokens.size()) == num_features) {
    return false;
  } else {
    return true;
  }
}

Parser* Parser::CreateParser(const char* filename, bool has_header, int num_features, int label_idx) {
  std::ifstream tmp_file;
  tmp_file.open(filename);
  if (!tmp_file.is_open()) {
    Log::Fatal("Data file: %s doesn't exist", filename);
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
    Log::Fatal("Data file: %s at least should have one line", filename);
  }
  if (!tmp_file.eof()) {
    std::getline(tmp_file, line2);
  } else {
    Log::Error("Data file: %s only have one line", filename);
  }
  tmp_file.close();
  int comma_cnt = 0, comma_cnt2 = 0;
  int tab_cnt = 0, tab_cnt2 = 0;
  int colon_cnt = 0, colon_cnt2 = 0;
  // Get some statistic from 2 line
  GetStatistic(line1.c_str(), &comma_cnt, &tab_cnt, &colon_cnt);
  GetStatistic(line2.c_str(), &comma_cnt2, &tab_cnt2, &colon_cnt2);
  Parser* ret = nullptr;
  bool has_label = true;
  if (line2.size() == 0) {
    // if only have one line on file
    if (colon_cnt > 0) {
      if (num_features > 0) {
        has_label = CheckHasLabelForLibsvm(line1);
      }
      ret = new LibSVMParser(has_label ? label_idx : -1);
    } else if (tab_cnt > 0) {
      if (num_features > 0 ) {
        has_label = CheckHasLabelForTSV(line1, num_features);
      }
      ret = new TSVParser(has_label ? label_idx : -1);
    } else if (comma_cnt > 0) {
      if (num_features > 0) {
        has_label = CheckHasLabelForCSV(line1, num_features);
      }
      ret = new CSVParser(has_label ? label_idx : -1);
    } 
  } else {
    if (colon_cnt > 0 || colon_cnt2 > 0) {
      if (num_features > 0) {
        has_label = CheckHasLabelForLibsvm(line1);
      }
      ret = new LibSVMParser(has_label ? label_idx : -1);
    }
    else if (tab_cnt == tab_cnt2 && tab_cnt > 0) {
      if (num_features > 0) {
        has_label = CheckHasLabelForTSV(line1, num_features);
      }
      ret = new TSVParser(has_label ? label_idx : -1);
    } else if (comma_cnt == comma_cnt2 && comma_cnt > 0) {
      if (num_features > 0) {
        has_label = CheckHasLabelForCSV(line1, num_features);
      }
      ret = new CSVParser(has_label ? label_idx : -1);
    }
  }
  if (!has_label) {
    Log::Info("Data file: %s doesn't contain label column", filename);
  }
  return ret;
}

}  // namespace LightGBM
