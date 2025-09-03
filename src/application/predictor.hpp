/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_PREDICTOR_HPP_
#define LIGHTGBM_PREDICTOR_HPP_

#include <LightGBM/boosting.h>
#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/common.h>
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
  * \param start_iteration Start index of the iteration to predict
  * \param num_iteration Number of boosting round
  * \param is_raw_score True if need to predict result with raw score
  * \param predict_leaf_index True to output leaf index instead of prediction score
  * \param predict_contrib True to output feature contributions instead of prediction score
  */
  Predictor(Boosting* boosting, int start_iteration, int num_iteration, bool is_raw_score,
            bool predict_leaf_index, bool predict_contrib, bool early_stop,
            int early_stop_freq, double early_stop_margin) {
    early_stop_ = CreatePredictionEarlyStopInstance(
        "none", LightGBM::PredictionEarlyStopConfig());
    if (early_stop && !boosting->NeedAccuratePrediction()) {
      PredictionEarlyStopConfig pred_early_stop_config;
      CHECK_GT(early_stop_freq, 0);
      CHECK_GE(early_stop_margin, 0);
      pred_early_stop_config.margin_threshold = early_stop_margin;
      pred_early_stop_config.round_period = early_stop_freq;
      if (boosting->NumberOfClasses() == 1) {
        early_stop_ =
            CreatePredictionEarlyStopInstance("binary", pred_early_stop_config);
      } else {
        early_stop_ = CreatePredictionEarlyStopInstance("multiclass",
                                                        pred_early_stop_config);
      }
    }

    boosting->InitPredict(start_iteration, num_iteration, predict_contrib);
    boosting_ = boosting;
    num_pred_one_row_ = boosting_->NumPredictOneRow(start_iteration,
        num_iteration, predict_leaf_index, predict_contrib);
    num_feature_ = boosting_->MaxFeatureIdx() + 1;
    predict_buf_.resize(
        OMP_NUM_THREADS(),
        std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>(
            num_feature_, 0.0f));
    const int kFeatureThreshold = 100000;
    const size_t KSparseThreshold = static_cast<size_t>(0.01 * num_feature_);
    if (predict_leaf_index) {
      predict_fun_ = [=](const std::vector<std::pair<int, double>>& features,
                         double* output) {
        int tid = omp_get_thread_num();
        if (num_feature_ > kFeatureThreshold &&
            features.size() < KSparseThreshold) {
          auto buf = CopyToPredictMap(features);
          boosting_->PredictLeafIndexByMap(buf, output);
        } else {
          CopyToPredictBuffer(predict_buf_[tid].data(), features);
          // get result for leaf index
          boosting_->PredictLeafIndex(predict_buf_[tid].data(), output);
          ClearPredictBuffer(predict_buf_[tid].data(), predict_buf_[tid].size(),
                             features);
        }
      };
    } else if (predict_contrib) {
      if (boosting_->IsLinear()) {
        Log::Fatal("Predicting SHAP feature contributions is not implemented for linear trees.");
      }
      predict_fun_ = [=](const std::vector<std::pair<int, double>>& features,
                         double* output) {
        int tid = omp_get_thread_num();
        CopyToPredictBuffer(predict_buf_[tid].data(), features);
        // get feature importances
        boosting_->PredictContrib(predict_buf_[tid].data(), output);
        ClearPredictBuffer(predict_buf_[tid].data(), predict_buf_[tid].size(),
                           features);
      };
      predict_sparse_fun_ = [=](const std::vector<std::pair<int, double>>& features,
                                std::vector<std::unordered_map<int, double>>* output) {
        auto buf = CopyToPredictMap(features);
        // get sparse feature importances
        boosting_->PredictContribByMap(buf, output);
      };

    } else {
      if (is_raw_score) {
        predict_fun_ = [=](const std::vector<std::pair<int, double>>& features,
                           double* output) {
          int tid = omp_get_thread_num();
          if (num_feature_ > kFeatureThreshold &&
              features.size() < KSparseThreshold) {
            auto buf = CopyToPredictMap(features);
            boosting_->PredictRawByMap(buf, output, &early_stop_);
          } else {
            CopyToPredictBuffer(predict_buf_[tid].data(), features);
            boosting_->PredictRaw(predict_buf_[tid].data(), output,
                                  &early_stop_);
            ClearPredictBuffer(predict_buf_[tid].data(),
                               predict_buf_[tid].size(), features);
          }
        };
      } else {
        predict_fun_ = [=](const std::vector<std::pair<int, double>>& features,
                           double* output) {
          int tid = omp_get_thread_num();
          if (num_feature_ > kFeatureThreshold &&
              features.size() < KSparseThreshold) {
            auto buf = CopyToPredictMap(features);
            boosting_->PredictByMap(buf, output, &early_stop_);
          } else {
            CopyToPredictBuffer(predict_buf_[tid].data(), features);
            boosting_->Predict(predict_buf_[tid].data(), output, &early_stop_);
            ClearPredictBuffer(predict_buf_[tid].data(),
                               predict_buf_[tid].size(), features);
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


  inline const PredictSparseFunction& GetPredictSparseFunction() const {
    return predict_sparse_fun_;
  }

  /*!
  * \brief predicting on data, then saving result to disk
  * \param data_filename Filename of data
  * \param result_filename Filename of output result
  */
  void Predict(const char* data_filename, const char* result_filename, bool header, bool disable_shape_check, bool precise_float_parser) {
    auto writer = VirtualFileWriter::Make(result_filename);
    if (!writer->Init()) {
      Log::Fatal("Prediction results file %s cannot be created", result_filename);
    }
    auto label_idx = header ? -1 : boosting_->LabelIdx();
    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(data_filename, header, boosting_->MaxFeatureIdx() + 1, label_idx,
                                                               precise_float_parser, boosting_->ParserConfigStr()));

    if (parser == nullptr) {
      Log::Fatal("Could not recognize the data format of data file %s", data_filename);
    }
    if (!header && !disable_shape_check && parser->NumFeatures() != boosting_->MaxFeatureIdx() + 1) {
      Log::Fatal("The number of features in data (%d) is not the same as it was in training data (%d).\n" \
                 "You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.", parser->NumFeatures(), boosting_->MaxFeatureIdx() + 1);
    }
    TextReader<data_size_t> predict_data_reader(data_filename, header);
    std::vector<int> feature_remapper(parser->NumFeatures(), -1);
    bool need_adjust = false;
    // skip raw feature remapping if trained model has parser config str which may contain actual feature names.
    if (header && boosting_->ParserConfigStr().empty()) {
      std::string first_line = predict_data_reader.first_line();
      std::vector<std::string> header_words = Common::Split(first_line.c_str(), "\t,");
      std::unordered_map<std::string, int> header_mapper;
      for (int i = 0; i < static_cast<int>(header_words.size()); ++i) {
        if (header_mapper.count(header_words[i]) > 0) {
          Log::Fatal("Feature (%s) appears more than one time.", header_words[i].c_str());
        }
        header_mapper[header_words[i]] = i;
      }
      const auto& fnames = boosting_->FeatureNames();
      for (int i = 0; i < static_cast<int>(fnames.size()); ++i) {
        if (header_mapper.count(fnames[i]) <= 0) {
          Log::Warning("Feature (%s) is missed in data file. If it is weight/query/group/ignore_column, you can ignore this warning.", fnames[i].c_str());
        } else {
          feature_remapper[header_mapper.at(fnames[i])] = i;
        }
      }
      for (int i = 0; i < static_cast<int>(feature_remapper.size()); ++i) {
        if (feature_remapper[i] >= 0 && i != feature_remapper[i]) {
          need_adjust = true;
          break;
        }
      }
    }
    // function for parse data
    std::function<void(const char*, std::vector<std::pair<int, double>>*)> parser_fun;
    double tmp_label;
    parser_fun = [&parser, &feature_remapper, &tmp_label, need_adjust]
    (const char* buffer, std::vector<std::pair<int, double>>* feature) {
      parser->ParseOneLine(buffer, feature, &tmp_label);
      if (need_adjust) {
        int i = 0, j = static_cast<int>(feature->size());
        while (i < j) {
          if (feature_remapper[(*feature)[i].first] >= 0) {
            (*feature)[i].first = feature_remapper[(*feature)[i].first];
            ++i;
          } else {
            // move the non-used features to the end of the feature vector
            std::swap((*feature)[i], (*feature)[--j]);
          }
        }
        feature->resize(i);
      }
    };

    std::function<void(data_size_t, const std::vector<std::string>&)>
        process_fun = [&parser_fun, &writer, this](
                          data_size_t, const std::vector<std::string>& lines) {
      std::vector<std::pair<int, double>> oneline_features;
      std::vector<std::string> result_to_write(lines.size());
      OMP_INIT_EX();
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) firstprivate(oneline_features)
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

  void InitializeRandomPairs(data_size_t num_items_in_query, std::vector<std::pair<int, int>>* random_pairs) {
    // TODO
  }

  void ComputeNewPairs(data_size_t num_items_in_query, const std::vector<double>& score_of_pairs,
                       std::vector<std::pair<int, int>>* new_pairs) {
    // TODO(need to clear old pairs from new_pairs before return)
  }

  /*!
  * \brief predicting on data, then saving result to disk
  * \brief used only in ``pairwise_lambdarank`` objective
  * \param data_filename Filename of data
  * \param result_filename Filename of output result
  */
  void PredictPairwise(const char* data_filename, const char* result_filename, bool header, bool disable_shape_check, bool precise_float_parser, int num_iteration, bool use_differential_feature_in_pairwise_ranking) {
    std::unique_ptr<Metadata> metadata(new Metadata());
    metadata->Init(data_filename);
    auto writer = VirtualFileWriter::Make(result_filename);
    if (!writer->Init()) {
      Log::Fatal("Prediction results file %s cannot be created", result_filename);
    }
    auto label_idx = header ? -1 : boosting_->LabelIdx();
    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(data_filename, header, boosting_->MaxFeatureIdx() + 1, label_idx,
                                                               precise_float_parser, boosting_->ParserConfigStr()));

    if (parser == nullptr) {
      Log::Fatal("Could not recognize the data format of data file %s", data_filename);
    }
    if (!header && !disable_shape_check && parser->NumFeatures() != boosting_->MaxFeatureIdx() + 1) {
      Log::Fatal("The number of features in data (%d) is not the same as it was in training data (%d).\n" \
                 "You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.", parser->NumFeatures(), boosting_->MaxFeatureIdx() + 1);
    }
    TextReader<data_size_t> predict_data_reader(data_filename, header);
    std::vector<int> feature_remapper(parser->NumFeatures(), -1);
    bool need_adjust = false;
    // skip raw feature remapping if trained model has parser config str which may contain actual feature names.
    if (header && boosting_->ParserConfigStr().empty()) {
      std::string first_line = predict_data_reader.first_line();
      std::vector<std::string> header_words = Common::Split(first_line.c_str(), "\t,");
      std::unordered_map<std::string, int> header_mapper;
      for (int i = 0; i < static_cast<int>(header_words.size()); ++i) {
        if (header_mapper.count(header_words[i]) > 0) {
          Log::Fatal("Feature (%s) appears more than one time.", header_words[i].c_str());
        }
        header_mapper[header_words[i]] = i;
      }
      const auto& fnames = boosting_->FeatureNames();
      for (int i = 0; i < static_cast<int>(fnames.size()); ++i) {
        if (header_mapper.count(fnames[i]) <= 0) {
          Log::Warning("Feature (%s) is missed in data file. If it is weight/query/group/ignore_column, you can ignore this warning.", fnames[i].c_str());
        } else {
          feature_remapper[header_mapper.at(fnames[i])] = i;
        }
      }
      for (int i = 0; i < static_cast<int>(feature_remapper.size()); ++i) {
        if (feature_remapper[i] >= 0 && i != feature_remapper[i]) {
          need_adjust = true;
          break;
        }
      }
    }
    // function for parse data
    std::function<void(const char*, std::vector<std::pair<int, double>>*)> parser_fun;
    double tmp_label;
    parser_fun = [&parser, &feature_remapper, &tmp_label, need_adjust]
    (const char* buffer, std::vector<std::pair<int, double>>* feature) {
      parser->ParseOneLine(buffer, feature, &tmp_label);
      if (need_adjust) {
        int i = 0, j = static_cast<int>(feature->size());
        while (i < j) {
          if (feature_remapper[(*feature)[i].first] >= 0) {
            (*feature)[i].first = feature_remapper[(*feature)[i].first];
            ++i;
          } else {
            // move the non-used features to the end of the feature vector
            std::swap((*feature)[i], (*feature)[--j]);
          }
        }
        feature->resize(i);
      }
    };

    const data_size_t* query_boundaries = metadata->query_boundaries();
    const data_size_t num_queries = metadata->num_queries();
    const int num_features = parser->NumFeatures();

    // Use query-aware processing for ranking tasks
    std::function<void(data_size_t, data_size_t, const std::vector<std::string>&)>
        process_fun_by_query = [use_differential_feature_in_pairwise_ranking, num_features, num_iteration, &parser_fun, &writer, this](
                                  data_size_t query_idx, data_size_t query_start, const std::vector<std::string>& lines) {

      // Get num items in this query
      const data_size_t num_items_in_query = static_cast<data_size_t>(lines.size());

      // Parse all the data in the current query
      std::vector<std::vector<std::pair<int, double>>> oneline_features_in_query(lines.size());
      OMP_INIT_EX();
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) firstprivate(oneline_features_in_query)
      for (data_size_t i = 0; i < num_items_in_query; ++i) {
        OMP_LOOP_EX_BEGIN();
        // parser
        parser_fun(lines[i].c_str(), &oneline_features_in_query[i]);
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();

      // Prepare for final prediction results
      std::vector<std::string> result_to_write(lines.size());

      // result buffer in each iteration
      std::vector<double> result;

      // list of paired indices in this query, in range [0, lines.size() - 1]
      std::vector<std::pair<data_size_t, data_size_t>> pair_indices;

      data_size_t old_num_pairs = static_cast<data_size_t>(pair_indices.size()); // 0 at first iteration
      //TODO(Pavel) Initialize with random pairing
      InitializeRandomPairs(num_items_in_query, &pair_indices);

      for (int k = 0; k < num_iteration; ++k) {
        // resize result vector
        result.resize(pair_indices.size() * num_pred_one_row_);
        OMP_INIT_EX();
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = old_num_pairs; i < static_cast<data_size_t>(pair_indices.size()); ++i) {
          OMP_LOOP_EX_BEGIN();
          
          // concatenate features from the paired instances
          std::vector<std::pair<int, double>> oneline_features;
          oneline_features.insert(oneline_features.end(),
                                  oneline_features_in_query[pair_indices[i].first].begin(),
                                  oneline_features_in_query[pair_indices[i].first].end());
          for (const auto& pair : oneline_features_in_query[pair_indices[i].second]) {
            oneline_features.push_back(std::make_pair(pair.first + num_features, pair.second));
          }
          if (use_differential_feature_in_pairwise_ranking) {
            std::set<int> feature_set;
            for (const auto& pair : oneline_features_in_query[pair_indices[i].first]) {
              feature_set.insert(pair.first);
            }
            for (const auto& pair : oneline_features_in_query[pair_indices[i].second]) {
              feature_set.insert(pair.first);
            }
            std::vector<std::pair<int, double>>::iterator first_feature_iterator = oneline_features_in_query[pair_indices[i].first].begin();
            std::vector<std::pair<int, double>>::iterator second_feature_iterator = oneline_features_in_query[pair_indices[i].second].begin();
            const std::vector<std::pair<int, double>>::iterator first_feature_iterator_end = oneline_features_in_query[pair_indices[i].first].end();
            const std::vector<std::pair<int, double>>::iterator second_feature_iterator_end = oneline_features_in_query[pair_indices[i].second].end();
            // elements are sorted in order in feature_set, which is an ordered set
            for (const int feature : feature_set) {
              double val1 = 0.0f, val2 = 0.0f;
              while (first_feature_iterator != first_feature_iterator_end && first_feature_iterator->first < feature) {
                ++first_feature_iterator;
              }
              if (first_feature_iterator < first_feature_iterator_end && first_feature_iterator->first == feature) {
                val1 = first_feature_iterator->second;
              }
              while (second_feature_iterator != second_feature_iterator_end && second_feature_iterator->first < feature) {
                ++second_feature_iterator;
              }
              if (first_feature_iterator < first_feature_iterator_end && first_feature_iterator->first == feature) {
                val2 = second_feature_iterator->second;
              }
              oneline_features.push_back(std::make_pair(feature + 2 * num_features, val1 - val2));
            }
          }
          predict_fun_(oneline_features, result.data() + i * num_pred_one_row_);
          OMP_LOOP_EX_END();
        }
        OMP_THROW_EX();

        old_num_pairs = static_cast<data_size_t>(pair_indices.size());
        //TODO(@Pavel) Compute new pairs, by pushing back to pair_indices
        ComputeNewPairs(num_items_in_query, result, &pair_indices);
      }
      // TODO(@Pavel): write final result to result_to_write (of size num items in current query)
      
      // Write results for this query
      for (data_size_t i = 0; i < static_cast<data_size_t>(result_to_write.size()); ++i) {
        writer->Write(result_to_write[i].c_str(), result_to_write[i].size());
        writer->Write("\n", 1);
      }
    };

    predict_data_reader.ReadAllAndProcessParallelByQuery(process_fun_by_query, query_boundaries, num_queries);
  }

 private:
  void CopyToPredictBuffer(double* pred_buf, const std::vector<std::pair<int, double>>& features) {
    for (const auto &feature : features) {
      if (feature.first < num_feature_) {
        pred_buf[feature.first] = feature.second;
      }
    }
  }

  void ClearPredictBuffer(double* pred_buf, size_t buf_size, const std::vector<std::pair<int, double>>& features) {
    if (features.size() > static_cast<size_t>(buf_size / 2)) {
      std::memset(pred_buf, 0, sizeof(double)*(buf_size));
    } else {
      for (const auto &feature : features) {
        if (feature.first < num_feature_) {
          pred_buf[feature.first] = 0.0f;
        }
      }
    }
  }

  std::unordered_map<int, double> CopyToPredictMap(const std::vector<std::pair<int, double>>& features) {
    std::unordered_map<int, double> buf;
    for (const auto &feature : features) {
      if (feature.first < num_feature_) {
        buf[feature.first] = feature.second;
      }
    }
    return buf;
  }

  /*! \brief Boosting model */
  const Boosting* boosting_;
  /*! \brief function for prediction */
  PredictFunction predict_fun_;
  PredictSparseFunction predict_sparse_fun_;
  PredictionEarlyStopInstance early_stop_;
  int num_feature_;
  int num_pred_one_row_;
  std::vector<std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>> predict_buf_;
};

}  // namespace LightGBM

#endif   // LightGBM_PREDICTOR_HPP_
