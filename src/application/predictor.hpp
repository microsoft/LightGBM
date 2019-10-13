/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_PREDICTOR_HPP_
#define LIGHTGBM_PREDICTOR_HPP_

#include <LightGBM/boosting.h>
#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/text_reader.h>

#include <string>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace LightGBM {

/*!
* \brief Used to predict data with input model
*/
class Predictor {
 public:
  /*!
  * \brief Constructor
  * \param boosting Input boosting model
  * \param num_iteration Number of boosting round
  * \param is_raw_score True if need to predict result with raw score
  * \param predict_leaf_index True to output leaf index instead of prediction score
  * \param predict_contrib True to output feature contributions instead of prediction score
  */
  Predictor(Boosting* boosting, int num_iteration,
            bool is_raw_score, bool predict_leaf_index, bool predict_contrib,
            bool early_stop, int early_stop_freq, double early_stop_margin) {
    early_stop_ = CreatePredictionEarlyStopInstance("none", LightGBM::PredictionEarlyStopConfig());
    if (early_stop && !boosting->NeedAccuratePrediction()) {
      PredictionEarlyStopConfig pred_early_stop_config;
      CHECK(early_stop_freq > 0);
      CHECK(early_stop_margin >= 0);
      pred_early_stop_config.margin_threshold = early_stop_margin;
      pred_early_stop_config.round_period = early_stop_freq;
      if (boosting->NumberOfClasses() == 1) {
        early_stop_ = CreatePredictionEarlyStopInstance("binary", pred_early_stop_config);
      } else {
        early_stop_ = CreatePredictionEarlyStopInstance("multiclass", pred_early_stop_config);
      }
    }

    #pragma omp parallel
    #pragma omp master
    {
      num_threads_ = omp_get_num_threads();
    }
    boosting->InitPredict(num_iteration, predict_contrib);
    boosting_ = boosting;
    num_pred_one_row_ = boosting_->NumPredictOneRow(num_iteration, predict_leaf_index, predict_contrib);
    num_feature_ = boosting_->MaxFeatureIdx() + 1;
    predict_buf_ = std::vector<std::vector<double>>(num_threads_, std::vector<double>(num_feature_, 0.0f));
    const int kFeatureThreshold = 100000;
    const size_t KSparseThreshold = static_cast<size_t>(0.01 * num_feature_);
    if (predict_leaf_index) {
      predict_fun_ = [=](const std::vector<std::pair<int, double>>& features, double* output) {
        int tid = omp_get_thread_num();
        if (num_feature_ > kFeatureThreshold && features.size() < KSparseThreshold) {
          auto buf = CopyToPredictMap(features);
          boosting_->PredictLeafIndexByMap(buf, output);
        } else {
          CopyToPredictBuffer(predict_buf_[tid].data(), features);
          // get result for leaf index
          boosting_->PredictLeafIndex(predict_buf_[tid].data(), output);
          ClearPredictBuffer(predict_buf_[tid].data(), predict_buf_[tid].size(), features);
        }
      };
    } else if (predict_contrib) {
        predict_fun_ = [=](const std::vector<std::pair<int, double>>& features, double* output) {
          int tid = omp_get_thread_num();
          CopyToPredictBuffer(predict_buf_[tid].data(), features);
          // get result for leaf index
          boosting_->PredictContrib(predict_buf_[tid].data(), output, &early_stop_);
          ClearPredictBuffer(predict_buf_[tid].data(), predict_buf_[tid].size(), features);
        };
    } else {
      if (is_raw_score) {
        predict_fun_ = [=](const std::vector<std::pair<int, double>>& features, double* output) {
          int tid = omp_get_thread_num();
          if (num_feature_ > kFeatureThreshold && features.size() < KSparseThreshold) {
            auto buf = CopyToPredictMap(features);
            boosting_->PredictRawByMap(buf, output, &early_stop_);
          } else {
            CopyToPredictBuffer(predict_buf_[tid].data(), features);
            boosting_->PredictRaw(predict_buf_[tid].data(), output, &early_stop_);
            ClearPredictBuffer(predict_buf_[tid].data(), predict_buf_[tid].size(), features);
          }
        };
      } else {
        predict_fun_ = [=](const std::vector<std::pair<int, double>>& features, double* output) {
          int tid = omp_get_thread_num();
          if (num_feature_ > kFeatureThreshold && features.size() < KSparseThreshold) {
            auto buf = CopyToPredictMap(features);
            boosting_->PredictByMap(buf, output, &early_stop_);
          } else {
            CopyToPredictBuffer(predict_buf_[tid].data(), features);
            boosting_->Predict(predict_buf_[tid].data(), output, &early_stop_);
            ClearPredictBuffer(predict_buf_[tid].data(), predict_buf_[tid].size(), features);
          }
        };
      }
    }
  }

  /*!
  * \brief Destructor
  */
  ~Predictor() {
  }

  inline const PredictFunction& GetPredictFunction() const {
    return predict_fun_;
  }

  /*!
  * \brief predicting on data, then saving result to disk
  * \param data_filename Filename of data
  * \param result_filename Filename of output result
  */
  void Predict(const char* data_filename, const char* result_filename, bool header) {
    auto writer = VirtualFileWriter::Make(result_filename);
    if (!writer->Init()) {
      Log::Fatal("Prediction results file %s cannot be found", result_filename);
    }
    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(data_filename, header, boosting_->MaxFeatureIdx() + 1, boosting_->LabelIdx()));

    if (parser == nullptr) {
      Log::Fatal("Could not recognize the data format of data file %s", data_filename);
    }
    if (parser->NumFeatures() != boosting_->MaxFeatureIdx() + 1) {
      Log::Fatal("The number of features in data (%d) is not the same as it was in training data (%d).", parser->NumFeatures(), boosting_->MaxFeatureIdx() + 1);
    }
    TextReader<data_size_t> predict_data_reader(data_filename, header);
    std::unordered_map<int, int> feature_names_map_;
    bool need_adjust = false;
    if (header) {
      std::string first_line = predict_data_reader.first_line();
      std::vector<std::string> header_words = Common::Split(first_line.c_str(), "\t,");
      header_words.erase(header_words.begin() + boosting_->LabelIdx());
      for (int i = 0; i < static_cast<int>(header_words.size()); ++i) {
        for (int j = 0; j < static_cast<int>(boosting_->FeatureNames().size()); ++j) {
          if (header_words[i] == boosting_->FeatureNames()[j]) {
            feature_names_map_[i] = j;
            break;
          }
        }
      }
      for (auto s : feature_names_map_) {
        if (s.first != s.second) {
          need_adjust = true;
          break;
        }
      }
    }
    // function for parse data
    std::function<void(const char*, std::vector<std::pair<int, double>>*)> parser_fun;
    double tmp_label;
    parser_fun = [&]
    (const char* buffer, std::vector<std::pair<int, double>>* feature) {
      parser->ParseOneLine(buffer, feature, &tmp_label);
      if (need_adjust) {
        int i = 0, j = static_cast<int>(feature->size());
        while (i < j) {
          if (feature_names_map_.find((*feature)[i].first) != feature_names_map_.end()) {
            (*feature)[i].first = feature_names_map_[(*feature)[i].first];
            ++i;
          } else {
            // move the non-used features to the end of the feature vector
            std::swap((*feature)[i], (*feature)[--j]);
          }
        }
        feature->resize(i);
      }
    };

    std::function<void(data_size_t, const std::vector<std::string>&)> process_fun = [&]
    (data_size_t, const std::vector<std::string>& lines) {
      std::vector<std::pair<int, double>> oneline_features;
      std::vector<std::string> result_to_write(lines.size());
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static) firstprivate(oneline_features)
      for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); ++i) {
        OMP_LOOP_EX_BEGIN();
        oneline_features.clear();
        // parser
        parser_fun(lines[i].c_str(), &oneline_features);
        // predict
        std::vector<double> result(num_pred_one_row_);
        predict_fun_(oneline_features, result.data());
        auto str_result = Common::Join<double>(result, "\t");
        result_to_write[i] = str_result;
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      for (data_size_t i = 0; i < static_cast<data_size_t>(result_to_write.size()); ++i) {
        writer->Write(result_to_write[i].c_str(), result_to_write[i].size());
        writer->Write("\n", 1);
      }
    };
    predict_data_reader.ReadAllAndProcessParallel(process_fun);
  }

 private:
  void CopyToPredictBuffer(double* pred_buf, const std::vector<std::pair<int, double>>& features) {
    int loop_size = static_cast<int>(features.size());
    for (int i = 0; i < loop_size; ++i) {
      if (features[i].first < num_feature_) {
        pred_buf[features[i].first] = features[i].second;
      }
    }
  }

  void ClearPredictBuffer(double* pred_buf, size_t buf_size, const std::vector<std::pair<int, double>>& features) {
    if (features.size() > static_cast<size_t>(buf_size / 2)) {
      std::memset(pred_buf, 0, sizeof(double)*(buf_size));
    } else {
      int loop_size = static_cast<int>(features.size());
      for (int i = 0; i < loop_size; ++i) {
        if (features[i].first < num_feature_) {
          pred_buf[features[i].first] = 0.0f;
        }
      }
    }
  }

  std::unordered_map<int, double> CopyToPredictMap(const std::vector<std::pair<int, double>>& features) {
    std::unordered_map<int, double> buf;
    int loop_size = static_cast<int>(features.size());
    for (int i = 0; i < loop_size; ++i) {
      if (features[i].first < num_feature_) {
        buf[features[i].first] = features[i].second;
      }
    }
    return buf;
  }

  /*! \brief Boosting model */
  const Boosting* boosting_;
  /*! \brief function for prediction */
  PredictFunction predict_fun_;
  PredictionEarlyStopInstance early_stop_;
  int num_feature_;
  int num_pred_one_row_;
  int num_threads_;
  std::vector<std::vector<double>> predict_buf_;
};

}  // namespace LightGBM

#endif   // LightGBM_PREDICTOR_HPP_
