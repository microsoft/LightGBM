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

Parser* Parser::CreateParser(const char* filename, int num_features, bool* has_label) {
  std::ifstream tmp_file;
  tmp_file.open(filename);
  if (!tmp_file.is_open()) {
    Log::Fatal("Data file: %s doesn't exist", filename);
  }
  std::string line1, line2;
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
  if (line2.size() == 0) {
    // if only have one line on file
    if (colon_cnt > 0) {
      ret =  new LibSVMParser();
      if (num_features > 0 && has_label != nullptr) {
        *has_label = CheckHasLabelForLibsvm(line1);
      }
    } else if (tab_cnt > 0) {
      ret = new TSVParser();
      if (num_features > 0 && has_label != nullptr) {
        *has_label = CheckHasLabelForTSV(line1, num_features);
      }
    } else if (comma_cnt > 0) {
      ret = new CSVParser();
      if (num_features > 0 && has_label != nullptr) {
        *has_label = CheckHasLabelForCSV(line1, num_features);
      }
    } 
  } else {
    if (colon_cnt > 0 || colon_cnt2 > 0) {
      ret = new LibSVMParser();
      if (num_features > 0 && has_label != nullptr) {
        *has_label = CheckHasLabelForLibsvm(line1);
      }
    }
    else if (tab_cnt == tab_cnt2 && tab_cnt > 0) {
      ret = new TSVParser();
      if (num_features > 0 && has_label != nullptr) {
        *has_label = CheckHasLabelForTSV(line1, num_features);
      }
    } else if (comma_cnt == comma_cnt2 && comma_cnt > 0) {
      ret = new CSVParser();
      if (num_features > 0 && has_label != nullptr) {
        *has_label = CheckHasLabelForCSV(line1, num_features);
      }
    }
  }
  return ret;
}

}  // namespace LightGBM
