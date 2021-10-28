/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "TransformProcessor.h"

#include <LightGBM/meta.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <ctype.h>
#include <math.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>
#include <cctype>
#include <codecvt>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace LightGBM;

size_t UIntParseFast(string value) {
  size_t num = 0;
  for (int index = 0; index < value.length(); ++index) {
    num = (size_t)(10 * (int)num + ((int)value[index] - 48));
  }
  return num;
}

double TransformProcessor::ConvertStrToDouble(string val) {
  if (val.empty() || val == "0" || val == "") return 0;
  if (val.find("E") == string::npos) return std::stod(val);
  // case scientific notation.
  vector<string> str_array;
  boost::split(str_array, val, boost::is_any_of("E"), boost::token_compress_on);
  double num1 = std::stod(str_array[0]);
  double num2 = pow(10.0, std::stod((str_array[1])));
  return num1 * num2;
}

bool TransformProcessor::CloseToZero(double val) {
  // Set an EPS to deal with the situation of "-0.00000" and make the result
  // consistent with the previous version.
  if ((val < EPS && val > -EPS) || (val < 0.001 && val > 0.0)) return true;
  return false;
}

vector<FeatureEvaluatorExtendedInfo> GetEvaluators(
    IniFileParserInterface *input_ini_interface,
    std::map<string, int> column_index_mapping) {
  vector<FeatureEvaluatorExtendedInfo> evaluator_extended_info_list;
  for (int i = 0; i < input_ini_interface->GetInputCount(); i++) {
    FeatureEvaluator *evaluator = new FeatureEvaluator(input_ini_interface, i);
    if (evaluator->IsRawFeatureEvaluator()) {
      int num = -1;
      string transform_str = "linear.";
      string key =
          input_ini_interface->GetInputName(i).substr(transform_str.length());
      if (column_index_mapping.find(key) != column_index_mapping.end()) {
        num = column_index_mapping.find(key)->second;
      } else {
        Log::Warning(
            "Feature cannot be found!\n\tFeature: %s\n\tInput expression: %s",
            key.c_str(), input_ini_interface->GetInputName(i).c_str());
      }

      FeatureEvaluatorExtendedInfo evaluator_extended_info;
      evaluator_extended_info.feature_evaluator_ = evaluator;
      evaluator_extended_info.featuremap_ = NULL;
      evaluator_extended_info.required_feature_mapped_indexes_ = vector<int>();
      evaluator_extended_info.index_ = num;
      evaluator_extended_info.israw_ = true;
      evaluator_extended_info_list.push_back(evaluator_extended_info);
    } else {
      string transform_str = "freeform2.";
      string key =
          input_ini_interface->GetInputName(i).substr(transform_str.length());
      IniFileParserInterface *from_freeform2 =
          IniFileParserInterface::Createfromfreeform2(key);
      FeatureEvaluator *evaluator = new FeatureEvaluator(from_freeform2, 0);
      FeatureMap *featuremap = new FeatureMap(from_freeform2);
      vector<int> intList;
      for (int i = 0; i < evaluator->GetRequiredRawFeatureIndices().size();
           i++) {
        int requiredRawFeatureIndex =
            evaluator->GetRequiredRawFeatureIndices()[i];
        int num = -1;
        string rawFeatureName =
            featuremap->GetRawFeatureName(requiredRawFeatureIndex);
        if (column_index_mapping.find(rawFeatureName) !=
            column_index_mapping.end()) {
          num = column_index_mapping.find(rawFeatureName)->second;
        } else {
          Log::Warning(
              "Feature cannot be found!\n\tFeature: %s\n\tInput expression: %s",
              rawFeatureName.c_str(), key.c_str());
        }
        intList.push_back(num);
      }
      FeatureEvaluatorExtendedInfo evaluator_extended_info;
      evaluator_extended_info.feature_evaluator_ = evaluator;
      evaluator_extended_info.featuremap_ = featuremap;
      evaluator_extended_info.required_feature_mapped_indexes_ = intList;
      evaluator_extended_info.index_ = -1;
      evaluator_extended_info.israw_ = false;
      evaluator_extended_info_list.push_back(evaluator_extended_info);
    }
  }
  return evaluator_extended_info_list;
}

TransformProcessor::TransformProcessor(const std::string &transform_str,
                                       const std::string &header_str,
                                       int label_id) {
  auto t0 = std::chrono::high_resolution_clock::now();
  this->_label_id = label_id;
  if (label_id < 0) {
    Log::Warning("Label id %d, found no label.", label_id);
  }
  this->_from_input_str =
      IniFileParserInterface::CreateFromInputStr(transform_str);
  // column Index mapping
  header_str = Common::Trim(header_str);
  string token;
  size_t pos = 0;
  size_t index = 0;
  string delimiter = "\t";
  while ((pos = header_str.find(delimiter)) != string::npos) {
    token = header_str.substr(0, pos);
    this->_column_index_mapping.insert(
        std::pair<string, int>(header_str.substr(0, pos), index));
    index++;
    header_str.erase(0, pos + delimiter.length());
  }
  this->_column_index_mapping.insert(std::pair<string, int>(header_str, index));
  this->_feature_evaluator_list =
      GetEvaluators(this->_from_input_str, this->_column_index_mapping);
  // time
  auto t1 = std::chrono::high_resolution_clock::now();
  Log::Info("Initialize transform time: %.2f seconds",
            std::chrono::duration<double, std::milli>(t1 - t0) * 1e-3);
}

void TransformProcessor::Parse(const char *str,
                               vector<string> *out_feature_strs,
                               double *out_label, string delimiter) {
  string data_line = str;
  data_line = Common::Trim(data_line);
  boost::split(*out_feature_strs, data_line, boost::is_any_of(delimiter));
  if (this->_label_id >= 0 and this->_label_id < out_feature_strs->size()) {
    *out_label = ConvertStrToDouble((*out_feature_strs)[this->_label_id]);
  }
}

void TransformProcessor::Apply(vector<string> *input_row,
                               vector<pair<int, double> > *out_features) {
  int num1 = 0;
  out_features->clear();
  for (vector<FeatureEvaluatorExtendedInfo>::iterator feature_evaluator =
           this->_feature_evaluator_list.begin();
       feature_evaluator != this->_feature_evaluator_list.end();
       feature_evaluator++) {
    double result = 0;
    if (feature_evaluator->israw_) {
      if (feature_evaluator->index_ > -1 &&
          feature_evaluator->index_ < input_row->size()) {
        result = ConvertStrToDouble((*input_row)[feature_evaluator->index_]);
      }
    } else {
      size_t inputsize =
          feature_evaluator->required_feature_mapped_indexes_.size();
      uint32_t *input = new uint32_t[inputsize];
      int index = 0;
      string str2 = "";
      for (int i = 0;
           i < feature_evaluator->required_feature_mapped_indexes_.size();
           i++) {
        input[index] = 0u;
        int feature_mapped_index =
            feature_evaluator->required_feature_mapped_indexes_[i];
        if (feature_mapped_index != -1 &&
            feature_mapped_index < input_row->size()) {
          str2 = (*input_row)[feature_mapped_index];
          if (str2.empty()) {
            input[index] = 0;
          }
          if (!str2.empty()) {
            input[index] = UIntParseFast(str2);
          }
        }
        ++index;
      }

      result = feature_evaluator->feature_evaluator_->Evaluate(&input[0]);
      delete[] input;

      if (std::isinf(result) && result < 0) {
        result = -3.40282346638529E+38;
      } else if (std::isinf(result) && result > 0) {
        result = 3.40282346638529E+38;
      } else if (std::isnan(result)) {
        result = 0.0;
      }
    }
    if (!CloseToZero(result)) {
      out_features->push_back(std::make_pair(num1, result));
    }
    num1++;
  }
}

int TransformProcessor::GetFeatureCount() {
  return this->_feature_evaluator_list.size();
}

vector<TransformedData> TransformProcessor::ApplyForFile(string data_path) {
  ifstream data_fin(data_path.c_str());
  string data_line;
  vector<TransformedData> dataset;
  while (std::getline(data_fin, data_line)) {
    double label;
    vector<string> input_row;
    vector<pair<int, double> > sparse_features;
    this->Parse(data_line.c_str(), &input_row, &label);
    this->Apply(&input_row, &sparse_features);
    dataset.push_back(TransformedData(label, sparse_features));
  }
  return dataset;
}

// Default: precision=5
string TransformedData::ToString(int precision) {
  string output_str = "";
  output_str += to_string((int)this->Label()) + " ";
  for (auto f : this->Features()) {
    std::stringstream ss;
    ss << f.first << ":" << setiosflags(ios::fixed)
       << std::setprecision(precision) << f.second << " ";
    output_str += ss.str();
  }
  return output_str;
}

// Default: precision=5
vector<string> TransformedData::ToString(vector<TransformedData> dataset,
                                         int precision) {
  vector<string> output_vector;
  for (auto d : dataset) output_vector.push_back(d.ToString(precision));
  return output_vector;
}

const double TransformProcessor::EPS = 1E-9;
