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

Parser* Parser::CreateParser(const char* filename) {
  std::ifstream tmp_file;
  tmp_file.open(filename);
  if (!tmp_file.is_open()) {
    Log::Stderr("Data file: %s doesn't exist", filename);
  }
  std::string line1, line2;
  if (!tmp_file.eof()) {
    std::getline(tmp_file, line1);
  } else {
    Log::Stderr("Data file: %s at least should have one line", filename);
  }
  if (!tmp_file.eof()) {
    std::getline(tmp_file, line2);
  } else {
    Log::Stdout("Data file: %s only have one line", filename);
  }
  tmp_file.close();
  int comma_cnt = 0, comma_cnt2 = 0;
  int tab_cnt = 0, tab_cnt2 = 0;
  int colon_cnt = 0, colon_cnt2 = 0;
  // Get some statistic from 2 line
  GetStatistic(line1.c_str(), &comma_cnt, &tab_cnt, &colon_cnt);
  GetStatistic(line2.c_str(), &comma_cnt2, &tab_cnt2, &colon_cnt2);
  if (line2.size() == 0) {
    // if only have one line on file
    if (colon_cnt > 0) {
      return new LibSVMParser();
    } else if (tab_cnt > 0) {
      return new TSVParser();
    } else if (comma_cnt > 0) {
      return new CSVParser();
    } else {
      return nullptr;
    }
  } else {
    if (colon_cnt > 0 || colon_cnt2 > 0) {
      return new LibSVMParser();
    }
    else if (tab_cnt == tab_cnt2 && tab_cnt > 0) {
      return new TSVParser();
    } else if (comma_cnt == comma_cnt2 && comma_cnt > 0) {
      return new CSVParser();
    } else {
      return nullptr;
    }
  }
}

}  // namespace LightGBM
