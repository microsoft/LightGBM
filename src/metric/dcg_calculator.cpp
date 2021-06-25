/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/metric.h>
#include <LightGBM/utils/log.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <math.h>
#include <fstream>

namespace LightGBM {

/*! \brief Declaration for some static members */
std::vector<double> DCGCalculator::label_gain_;
std::vector<double> DCGCalculator::discount_;
const data_size_t DCGCalculator::kMaxPosition = 10000;
std::map<std::pair<data_size_t, data_size_t>, double> DCGCalculator::position_bias_lookup_;
int position_bias_lookup_max_ = -1;

void DCGCalculator::DefaultEvalAt(std::vector<int>* eval_at) {
  auto& ref_eval_at = *eval_at;
  if (ref_eval_at.empty()) {
    for (int i = 1; i <= 5; ++i) {
      ref_eval_at.push_back(i);
    }
  } else {
    for (size_t i = 0; i < eval_at->size(); ++i) {
      CHECK_GT(ref_eval_at[i], 0);
    }
  }
}

void DCGCalculator::DefaultLabelGain(std::vector<double>* label_gain) {
  if (!label_gain->empty()) { return; }
  // label_gain = 2^i - 1, may overflow, so we use 31 here
  const int max_label = 31;
  label_gain->push_back(0.0f);
  for (int i = 1; i < max_label; ++i) {
    label_gain->push_back(static_cast<double>((1 << i) - 1));
  }
}

void DCGCalculator::Init(const std::vector<double>& input_label_gain) {
  label_gain_.resize(input_label_gain.size());
  for (size_t i = 0; i < input_label_gain.size(); ++i) {
    label_gain_[i] = static_cast<double>(input_label_gain[i]);
  }
  discount_.resize(kMaxPosition);
  for (data_size_t i = 0; i < kMaxPosition; ++i) {
    discount_[i] = 1.0 / std::log2(2.0 + i);
  }
  // Load Position biases if any
  if (const char* position_bias_path = std::getenv("POS_BIAS_PATH")) {
      std::fstream pb_fin;
      pb_fin.open(position_bias_path, std::ios::in);

      if (!pb_fin.is_open()) {
        Log::Fatal("Couldn't open position bias file at path %s", position_bias_path);
      }


      std::string line;

      data_size_t i, j;
      double bias;
      while(std::getline(pb_fin,line)) {
        std::sscanf(line.c_str(),"%d,%d,%lf", &i, &j, &bias);
        position_bias_lookup_[std::make_pair(i, j)] = bias;
        position_bias_lookup_[std::make_pair(j, i)] = 1.0 / bias;
        //find the max index in the position bias lookup table
        if (((i > position_bias_lookup_max_) || (j > position_bias_lookup_max_)) && (bias > 0.00001)){
            position_bias_lookup_max_ = std::max(i,j);
        }
      }

  } else {
      Log::Warning("POS_BIAS_PATH environment variable not set, using biases of 1.0");
  }
}

double DCGCalculator::CalMaxDCGAtK(data_size_t k, const label_t* label, data_size_t num_data) {
  double ret = 0.0f;
  // counts for all labels
  std::vector<data_size_t> label_cnt(label_gain_.size(), 0);
  for (data_size_t i = 0; i < num_data; ++i) {
    ++label_cnt[static_cast<int>(label[i])];
  }
  int top_label = static_cast<int>(label_gain_.size()) - 1;

  if (k > num_data) { k = num_data; }
  //  start from top label, and accumulate DCG
  for (data_size_t j = 0; j < k; ++j) {
    while (top_label > 0 && label_cnt[top_label] <= 0) {
      top_label -= 1;
    }
    if (top_label < 0) {
      break;
    }
    ret += discount_[j] * label_gain_[top_label];
    label_cnt[top_label] -= 1;
  }
  return ret;
}

void DCGCalculator::CalMaxDCG(const std::vector<data_size_t>& ks,
                              const label_t* label,
                              data_size_t num_data,
                              bool weighted,
                              std::vector<double>* out) {
  if (!weighted) {
    std::vector<data_size_t> label_cnt(label_gain_.size(), 0);
    // counts for all labels
    for (data_size_t i = 0; i < num_data; ++i) {
    ++label_cnt[static_cast<int>(label[i])];
    }
    double cur_result = 0.0f;
    data_size_t cur_left = 0;
    int top_label = static_cast<int>(label_gain_.size()) - 1;
    // calculate k Max DCG by one pass
    for (size_t i = 0; i < ks.size(); ++i) {
    data_size_t cur_k = ks[i];
    if (cur_k > num_data) { cur_k = num_data; }
    for (data_size_t j = cur_left; j < cur_k; ++j) {
      while (top_label > 0 && label_cnt[top_label] <= 0) {
        top_label -= 1;
      }
      if (top_label < 0) {
        break;
      }
      cur_result += discount_[j] * label_gain_[top_label] * 1.0;
      label_cnt[top_label] -= 1;
    }
    (*out)[i] = cur_result;
    cur_left = cur_k;
    }
  }
  else {
    //vectors to help us calculate idcg
    std::vector<double> knowns(num_data);
    std::vector<double> unknowns(num_data);
    for (data_size_t i = 0; i < num_data; ++i) {
      knowns[i] = 0;
      unknowns[i] = 0;
    }

    data_size_t cur_left = 0;
    // calculate multi dcg by one pass
    for (size_t i = 0; i < ks.size(); ++i) {
      data_size_t cur_k = ks[i];
      if (cur_k > num_data) { cur_k = num_data; }
      
      for (data_size_t j = cur_left; j < num_data; ++j) {

        //default to 1.0
        double position_bias_ratio = 1.0;
        if (!position_bias_lookup_.empty()) {
            //try to find position bias
            auto it = position_bias_lookup_.find(std::make_pair(1, std::min(j+1, position_bias_lookup_max_)));
            if (it != position_bias_lookup_.end()){
                //if found, get that value
                position_bias_ratio = it->second;
            }
        }
        knowns[j] = label_gain_[static_cast<int>(label[j])] / position_bias_ratio;
        unknowns[j] = position_bias_ratio * discount_[j];
      }
      std::stable_sort(knowns.begin(), knowns.end(), std::greater <double>());
      std::stable_sort(unknowns.begin(), unknowns.end(), std::greater <double>());
        
      double cur_result = 0.0f;
      for (int j = 0; j < cur_k; ++j) {
          cur_result += knowns[j] * unknowns[j];
          //Log::Info("\nknowns[%d]=%f\nunknowns[%d]=%f",j,knowns[j],j,unknowns[j]);
      }
      (*out)[i] = cur_result;
      cur_left = cur_k;
    }
  }
}


double DCGCalculator::CalDCGAtK(data_size_t k, const label_t* label,
                                const double* score, data_size_t num_data) {
  // get sorted indices by score
  std::vector<data_size_t> sorted_idx(num_data);
  for (data_size_t i = 0; i < num_data; ++i) {
    sorted_idx[i] = i;
  }
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [score](data_size_t a, data_size_t b) {return score[a] > score[b]; });

  if (k > num_data) { k = num_data; }
  double dcg = 0.0f;
  // calculate dcg
  for (data_size_t i = 0; i < k; ++i) {
    data_size_t idx = sorted_idx[i];
    dcg += label_gain_[static_cast<int>(label[idx])] * discount_[i];
  }
  return dcg;
}

void DCGCalculator::CalDCG(const std::vector<data_size_t>& ks, const label_t* label,
                           const double * score, data_size_t num_data, bool weighted, std::vector<double>* out) {
  // get sorted indices by score
  std::vector<data_size_t> sorted_idx(num_data);
  for (data_size_t i = 0; i < num_data; ++i) {
    sorted_idx[i] = i;
  }
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [score](data_size_t a, data_size_t b) {return score[a] > score[b]; });

  double cur_result = 0.0f;
  data_size_t cur_left = 0;
  // calculate multi dcg by one pass
  for (size_t i = 0; i < ks.size(); ++i) {
    data_size_t cur_k = ks[i];
    if (cur_k > num_data) { cur_k = num_data; }
    for (data_size_t j = cur_left; j < cur_k; ++j) {
      data_size_t idx = sorted_idx[j];
      if (weighted) {
        
        //default to 1.0
        double position_bias_ratio = 1.0;
        
        if (!position_bias_lookup_.empty()) {
            //try to find position bias
            auto it = position_bias_lookup_.find(std::make_pair(std::min(idx+1, position_bias_lookup_max_), std::min(j+1, position_bias_lookup_max_)));
            //if found, get that bias
            if (it != position_bias_lookup_.end()){
                position_bias_ratio = it->second;
            }
        }
   
        cur_result += label_gain_[static_cast<int>(label[idx])] * discount_[j] * position_bias_ratio;
      }
      else {
        cur_result += label_gain_[static_cast<int>(label[idx])] * discount_[j];
      }

    }
    (*out)[i] = cur_result;
    cur_left = cur_k;
  }
}

void DCGCalculator::CheckMetadata(const Metadata& metadata, data_size_t num_queries) {
  const data_size_t* query_boundaries = metadata.query_boundaries();
  if (num_queries > 0 && query_boundaries != nullptr) {
    for (data_size_t i = 0; i < num_queries; i++) {
      data_size_t num_rows = query_boundaries[i + 1] - query_boundaries[i];
      if (num_rows > kMaxPosition) {
        Log::Fatal("Number of rows %i exceeds upper limit of %i for a query", static_cast<int>(num_rows), static_cast<int>(kMaxPosition));
      }
    }
  }
}


void DCGCalculator::CheckLabel(const label_t* label, data_size_t num_data) {
  for (data_size_t i = 0; i < num_data; ++i) {
    label_t delta = std::fabs(label[i] - static_cast<int>(label[i]));
    if (delta > kEpsilon) {
      Log::Fatal("label should be int type (met %f) for ranking task,\n"
                 "for the gain of label, please set the label_gain parameter", label[i]);
    }

    if (label[i] < 0) {
      Log::Fatal("Label should be non-negative (met %f) for ranking task", label[i]);
    }

    if (static_cast<size_t>(label[i]) >= label_gain_.size()) {
      Log::Fatal("Label %zu is not less than the number of label mappings (%zu)", static_cast<size_t>(label[i]), label_gain_.size());
    }
  }
}

}  // namespace LightGBM
