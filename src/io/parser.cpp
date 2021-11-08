/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "parser.hpp"

#include <string>
#include <algorithm>
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

int GetLabelIdxForLibsvm(const std::string& str, int num_features, int label_idx) {
  if (num_features <= 0) {
    return label_idx;
  }
  auto str2 = Common::Trim(str);
  auto pos_space = str2.find_first_of(" \f\n\r\t\v");
  auto pos_colon = str2.find_first_of(":");
  if (pos_space == std::string::npos || pos_space < pos_colon) {
    return label_idx;
  } else {
    return -1;
  }
}

int GetLabelIdxForTSV(const std::string& str, int num_features, int label_idx) {
  if (num_features <= 0) {
    return label_idx;
  }
  auto str2 = Common::Trim(str);
  auto tokens = Common::Split(str2.c_str(), '\t');
  if (static_cast<int>(tokens.size()) == num_features) {
    return -1;
  } else {
    return label_idx;
  }
}

int GetLabelIdxForCSV(const std::string& str, int num_features, int label_idx) {
  if (num_features <= 0) {
    return label_idx;
  }
  auto str2 = Common::Trim(str);
  auto tokens = Common::Split(str2.c_str(), ',');
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

void GetLine(std::stringstream* ss, std::string* line, const VirtualFileReader* reader, std::vector<char>* buffer, size_t buffer_size) {
  std::getline(*ss, *line);
  while (ss->eof()) {
    size_t read_len = reader->Read(buffer->data(), buffer_size);
    if (read_len <= 0) {
      break;
    }
    ss->clear();
    ss->str(std::string(buffer->data(), read_len));
    std::string tmp;
    std::getline(*ss, tmp);
    *line += tmp;
  }
}

std::vector<std::string> ReadKLineFromFile(const char* filename, bool header, int k) {
  auto reader = VirtualFileReader::Make(filename);
  if (!reader->Init()) {
    Log::Fatal("Data file %s doesn't exist.", filename);
  }
  std::vector<std::string> ret;
  std::string cur_line;
  const size_t buffer_size = 1024 * 1024;
  auto buffer = std::vector<char>(buffer_size);
  size_t read_len = reader->Read(buffer.data(), buffer_size);
  if (read_len <= 0) {
    Log::Fatal("Data file %s couldn't be read.", filename);
  }
  std::string read_str = std::string(buffer.data(), read_len);
  std::stringstream tmp_file(read_str);
  if (header) {
    if (!tmp_file.eof()) {
      GetLine(&tmp_file, &cur_line, reader.get(), &buffer, buffer_size);
    }
  }
  for (int i = 0; i < k; ++i) {
    if (!tmp_file.eof()) {
      GetLine(&tmp_file, &cur_line, reader.get(), &buffer, buffer_size);
      cur_line = Common::Trim(cur_line);
      if (!cur_line.empty()) {
        ret.push_back(cur_line);
      }
    } else {
      break;
    }
  }
  if (ret.empty()) {
    Log::Fatal("Data file %s should have at least one line.", filename);
  } else if (ret.size() == 1) {
    Log::Warning("Data file %s only has one line.", filename);
  }
  return ret;
}

int GetNumColFromLIBSVMFile(const char* filename, bool header) {
  auto reader = VirtualFileReader::Make(filename);
  if (!reader->Init()) {
    Log::Fatal("Data file %s doesn't exist.", filename);
  }
  std::vector<std::string> ret;
  std::string cur_line;
  const size_t buffer_size = 1024 * 1024;
  auto buffer = std::vector<char>(buffer_size);
  size_t read_len = reader->Read(buffer.data(), buffer_size);
  if (read_len <= 0) {
    Log::Fatal("Data file %s couldn't be read.", filename);
  }
  std::string read_str = std::string(buffer.data(), read_len);
  std::stringstream tmp_file(read_str);
  if (header) {
    if (!tmp_file.eof()) {
      GetLine(&tmp_file, &cur_line, reader.get(), &buffer, buffer_size);
    }
  }
  int max_col_idx = 0;
  int max_line_idx = 0;
  const int stop_round = 1 << 7;
  const int max_line = 1 << 13;
  for (int i = 0; i < max_line; ++i) {
    if (!tmp_file.eof()) {
      GetLine(&tmp_file, &cur_line, reader.get(), &buffer, buffer_size);
      cur_line = Common::Trim(cur_line);
      auto colon_pos = cur_line.find_last_of(":");
      auto space_pos = cur_line.find_last_of(" \f\t\v");
      auto sub_str = cur_line.substr(space_pos + 1, space_pos - colon_pos - 1);
      int cur_idx = 0;
      Common::Atoi(sub_str.c_str(), &cur_idx);
      if (cur_idx > max_col_idx) {
        max_col_idx = cur_idx;
        max_line_idx = i;
      }
      if (i - max_line_idx >= stop_round) {
        break;
      }
    } else {
      break;
    }
  }
  CHECK_GT(max_col_idx, 0);
  return max_col_idx;
}

DataType GetDataType(const char* filename, bool header,
                     const std::vector<std::string>& lines, int* num_col) {
  DataType type = DataType::INVALID;
  if (lines.empty()) {
    return type;
  }
  int comma_cnt = 0;
  int tab_cnt = 0;
  int colon_cnt = 0;
  GetStatistic(lines[0].c_str(), &comma_cnt, &tab_cnt, &colon_cnt);
  size_t num_lines = lines.size();
  if (num_lines == 1) {
    if (colon_cnt > 0) {
      type = DataType::LIBSVM;
    } else if (tab_cnt > 0) {
      type = DataType::TSV;
    } else if (comma_cnt > 0) {
      type = DataType::CSV;
    }
  } else {
    int comma_cnt2 = 0;
    int tab_cnt2 = 0;
    int colon_cnt2 = 0;
    GetStatistic(lines[1].c_str(), &comma_cnt2, &tab_cnt2, &colon_cnt2);
    if (colon_cnt > 0 || colon_cnt2 > 0) {
      type = DataType::LIBSVM;
    } else if (tab_cnt == tab_cnt2 && tab_cnt > 0) {
      type = DataType::TSV;
    } else if (comma_cnt == comma_cnt2 && comma_cnt > 0) {
      type = DataType::CSV;
    }
    if (type == DataType::TSV || type == DataType::CSV) {
      // valid the type
      for (size_t i = 2; i < num_lines; ++i) {
        GetStatistic(lines[i].c_str(), &comma_cnt2, &tab_cnt2, &colon_cnt2);
        if (type == DataType::TSV && tab_cnt2 != tab_cnt) {
          type = DataType::INVALID;
          break;
        } else if (type == DataType::CSV && comma_cnt != comma_cnt2) {
          type = DataType::INVALID;
          break;
        }
      }
    }
  }
  if (type == DataType::LIBSVM) {
    int max_col_idx = GetNumColFromLIBSVMFile(filename, header);
    *num_col = max_col_idx + 1;
  } else if (type == DataType::CSV) {
    *num_col = comma_cnt + 1;
  } else if (type == DataType::TSV) {
    *num_col = tab_cnt + 1;
  }
  return type;
}

Parser* Parser::CreateParser(const char* filename, bool header, int num_features, int label_idx, bool precise_float_parser) {
  const int n_read_line = 32;
  auto lines = ReadKLineFromFile(filename, header, n_read_line);
  int num_col = 0;
  DataType type = GetDataType(filename, header, lines, &num_col);
  if (type == DataType::INVALID) {
    Log::Fatal("Unknown format of training data. Only CSV, TSV, and LibSVM (zero-based) formatted text files are supported.");
  }
  std::unique_ptr<Parser> ret;
  int output_label_index = -1;
  AtofFunc atof = precise_float_parser ? Common::AtofPrecise : Common::Atof;
  if (type == DataType::LIBSVM) {
    output_label_index = GetLabelIdxForLibsvm(lines[0], num_features, label_idx);
    ret.reset(new LibSVMParser(output_label_index, num_col, atof));
  } else if (type == DataType::TSV) {
    output_label_index = GetLabelIdxForTSV(lines[0], num_features, label_idx);
    ret.reset(new TSVParser(output_label_index, num_col, atof));
  } else if (type == DataType::CSV) {
    output_label_index = GetLabelIdxForCSV(lines[0], num_features, label_idx);
    ret.reset(new CSVParser(output_label_index, num_col, atof));
  }

  if (output_label_index < 0 && label_idx >= 0) {
    Log::Info("Data file %s doesn't contain a label column.", filename);
  }
  return ret.release();
}

}  // namespace LightGBM
