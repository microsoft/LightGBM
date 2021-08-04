/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_predictor.hpp"
#include <chrono>

namespace LightGBM {

CUDAPredictor::CUDAPredictor(Boosting* boosting, int start_iteration, int num_iteration, bool is_raw_score,
  bool predict_leaf_index, bool predict_contrib, bool early_stop,
  int early_stop_freq, double early_stop_margin):
  Predictor(boosting, start_iteration, num_iteration, is_raw_score, predict_leaf_index, predict_contrib, early_stop, early_stop_freq, early_stop_margin),
  is_raw_score_(is_raw_score), predict_leaf_index_(predict_leaf_index), predict_contrib_(predict_contrib) {
  InitCUDAModel(start_iteration, num_iteration);
  num_pred_in_one_row_ = static_cast<int64_t>(boosting_->NumPredictOneRow(start_iteration, num_iteration, predict_leaf_index, predict_contrib));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));
}

CUDAPredictor::~CUDAPredictor() {}

void CUDAPredictor::Predict(const char* data_filename, const char* result_filename, bool header, bool disable_shape_check) {
  if (predict_leaf_index_) {
    CHECK_EQ(num_pred_in_one_row_, static_cast<int64_t>(num_iteration_));
  }
  auto label_idx = header ? -1 : boosting_->LabelIdx();
  auto parser = std::unique_ptr<Parser>(Parser::CreateParser(data_filename, header, boosting_->MaxFeatureIdx() + 1, label_idx));
  if (parser == nullptr) {
    Log::Fatal("Could not recognize the data format of data file %s", data_filename);
  }
  if (!header && !disable_shape_check && parser->NumFeatures() != boosting_->MaxFeatureIdx() + 1) {
    Log::Fatal("The number of features in data (%d) is not the same as it was in training data (%d).\n" \
                "You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.",
                parser->NumFeatures(), boosting_->MaxFeatureIdx() + 1);
  }
  TextReader<data_size_t> predict_data_reader(data_filename, header);
  std::vector<int> feature_remapper(parser->NumFeatures(), -1);
  bool need_adjust = false;
  if (header) {
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
  auto writer = VirtualFileWriter::Make(result_filename);
  if (!writer->Init()) {
    Log::Fatal("Prediction results file %s cannot be found", result_filename);
  }
  PredictWithParserFun(parser_fun, &predict_data_reader, writer.get());
}

void CUDAPredictor::PredictWithParserFun(std::function<void(const char*, std::vector<std::pair<int, double>>*)> parser_fun,
                            TextReader<data_size_t>* predict_data_reader,
                            VirtualFileWriter* writer) {
  // use lager buffer size to reduce the time spent in copying from Host to CUDA
  const data_size_t buffer_size = 50000;
  AllocateCUDAMemoryOuter<double>(&cuda_data_, static_cast<size_t>(buffer_size) * static_cast<size_t>(num_feature_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_result_buffer_, static_cast<size_t>(buffer_size) * static_cast<size_t>(num_pred_in_one_row_), __FILE__, __LINE__);
  std::vector<double> buffer(buffer_size * num_feature_, 0.0f);
  std::vector<double> result_buffer(buffer_size * num_pred_in_one_row_, 0.0f);
  auto process_fun = [&parser_fun, &writer, &buffer, &result_buffer, buffer_size, this]
    (data_size_t /*start_index*/, const std::vector<std::string>& lines) {
    std::vector<std::pair<int, double>> oneline_features;
    std::vector<std::string> result_to_write(lines.size());
    const data_size_t num_lines = static_cast<data_size_t>(lines.size());
    const int num_blocks = (num_lines + buffer_size - 1) / buffer_size;
    for (int block_index = 0; block_index < num_blocks; ++block_index) {
      const data_size_t block_start = block_index * buffer_size;
      const data_size_t block_end = std::min(block_start + buffer_size, num_lines);
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static) firstprivate(oneline_features)
      for (data_size_t i = block_start; i < block_end; ++i) {
        OMP_LOOP_EX_BEGIN();
        oneline_features.clear();
        // parser
        parser_fun(lines[i].c_str(), &oneline_features);
        // predict
        const data_size_t index_in_block = i - block_start;
        double* one_row_data = buffer.data() + index_in_block * num_feature_;
        for (int feature_index = 0; feature_index < num_feature_; ++feature_index) {
          one_row_data[feature_index] = 0.0f;
        }
        for (const auto& pair : oneline_features) {
          one_row_data[pair.first] = pair.second;
        }
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      SynchronizeCUDADeviceOuter(cuda_stream_, __FILE__, __LINE__);
      CopyFromHostToCUDADeviceAsyncOuter<double>(cuda_data_, buffer.data(), static_cast<size_t>(buffer_size * num_feature_), cuda_stream_, __FILE__, __LINE__);
      LaunchPredictKernelAsync(buffer_size, false);
      CopyFromCUDADeviceToHostAsyncOuter<double>(result_buffer.data(),
                                               cuda_result_buffer_,
                                               static_cast<size_t>(buffer_size) * static_cast<size_t>(num_pred_in_one_row_),
                                               cuda_stream_,
                                               __FILE__,
                                               __LINE__);
      #pragma omp parallel for schedule(static)
      for (data_size_t i = block_start; i < block_end; ++i) {
        OMP_LOOP_EX_BEGIN();
        const data_size_t index_in_block = i - block_start;
        const double* begin = result_buffer.data() + index_in_block * num_pred_in_one_row_;
        const double* end = begin + num_pred_in_one_row_;
        result_to_write[i] = Common::Join<double>(std::vector<double>(begin, end), "\t");
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    }
    for (data_size_t i = 0; i < static_cast<data_size_t>(result_to_write.size()); ++i) {
      writer->Write(result_to_write[i].c_str(), result_to_write[i].size());
      writer->Write("\n", 1);
    }
  };
  predict_data_reader->ReadAllAndProcessParallel(process_fun);
}

void CUDAPredictor::Predict(const data_size_t num_data,
                       const int64_t num_pred_in_one_row,
                       const std::function<std::vector<std::pair<int, double>>(int row_idx)>& get_row_fun,
                       double* out_result) {
  const data_size_t buffer_size = 50000;
  CHECK_EQ(num_pred_in_one_row_, num_pred_in_one_row);
  if (predict_leaf_index_) {
    CHECK_EQ(num_pred_in_one_row_, static_cast<int64_t>(num_iteration_));
  }
  AllocateCUDAMemoryOuter<double>(&cuda_data_, static_cast<size_t>(buffer_size) * static_cast<size_t>(num_feature_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_result_buffer_, static_cast<size_t>(buffer_size) * static_cast<size_t>(num_pred_in_one_row_), __FILE__, __LINE__);
  std::vector<double> buffer(buffer_size * num_feature_, 0.0f);
  const int num_blocks = (num_data + buffer_size - 1) / buffer_size;
  data_size_t block_offset = 0;
  for (int block_index = 0; block_index < num_blocks; ++block_index) {
    Threading::For<data_size_t>(0, buffer_size, 512,
      [block_offset, get_row_fun, &buffer, this] (int /*thread_index*/, data_size_t start, data_size_t end) {
        std::vector<std::pair<int, double>> oneline_feature;
        for (data_size_t i = start; i < end; ++i) {
          oneline_feature = get_row_fun(i + block_offset);
          double* one_row_data = buffer.data() + i * num_feature_;
          for (int feature_index = 0; feature_index < num_feature_; ++feature_index) {
            one_row_data[feature_index] = 0.0f;
          }
          for (const auto& pair : oneline_feature) {
            one_row_data[pair.first] = pair.second;
          }
        }
      });
    SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
    CopyFromHostToCUDADeviceAsyncOuter<double>(cuda_data_, buffer.data(), static_cast<size_t>(buffer_size * num_feature_), cuda_stream_, __FILE__, __LINE__);
    LaunchPredictKernelAsync(buffer_size, false);
    CopyFromCUDADeviceToHostAsyncOuter<double>(out_result + static_cast<size_t>(block_offset) * static_cast<size_t>(num_pred_in_one_row_),
                                               cuda_result_buffer_,
                                               static_cast<size_t>(buffer_size) * static_cast<size_t>(num_pred_in_one_row_),
                                               cuda_stream_,
                                               __FILE__,
                                               __LINE__);
    block_offset += buffer_size;
  }
}

void CUDAPredictor::InitCUDAModel(const int start_iteration, const int num_iteration) {
  const std::vector<std::unique_ptr<Tree>>& models = boosting_->models();
  cuda_convert_output_function_ = boosting_->GetCUDAConvertOutputFunc();
  const int num_tree_per_iteration = boosting_->num_tree_per_iteration();
  num_iteration_ = static_cast<int>(models.size()) / num_tree_per_iteration;
  start_iteration_ = std::max(start_iteration, 0);
  start_iteration_ = std::min(start_iteration_, num_iteration_);
  if (num_iteration > 0) {
    num_iteration_ = std::min(num_iteration, num_iteration_ - start_iteration_);
  } else {
    num_iteration_ = num_iteration_ - start_iteration_;
  }
  std::vector<int> tree_num_leaves(num_iteration_, 0);
  std::vector<const int*> tree_left_child(num_iteration_, nullptr);
  std::vector<const int*> tree_right_child(num_iteration_, nullptr);
  std::vector<const double*> tree_leaf_value(num_iteration_, nullptr);
  std::vector<const double*> tree_threshold(num_iteration_, nullptr);
  std::vector<const int8_t*> tree_decision_type(num_iteration_, nullptr);
  std::vector<const int*> tree_split_feature_index(num_iteration_, nullptr);
  const int num_threads = OMP_NUM_THREADS();
  #pragma omp parallel for schedule(static) num_threads(num_threads) if (num_iteration_ >= 1024)
  for (int tree_index = 0; tree_index < num_iteration_; ++tree_index) {
    CHECK(models[tree_index]->is_cuda_tree());
    const CUDATree* cuda_tree = reinterpret_cast<const CUDATree*>(models[tree_index + start_iteration_].get());
    tree_num_leaves[tree_index] = cuda_tree->num_leaves();
    tree_left_child[tree_index] = cuda_tree->cuda_left_child();
    tree_right_child[tree_index] = cuda_tree->cuda_right_child();
    tree_leaf_value[tree_index] = cuda_tree->cuda_leaf_value();
    tree_threshold[tree_index] = cuda_tree->cuda_threshold();
    tree_decision_type[tree_index] = cuda_tree->cuda_decision_type();
    tree_split_feature_index[tree_index] = cuda_tree->cuda_split_feature();
  }
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_tree_num_leaves_,
                                         tree_num_leaves.data(),
                                         tree_num_leaves.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int*>(&cuda_left_child_,
                                         tree_left_child.data(),
                                         tree_left_child.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int*>(&cuda_right_child_,
                                         tree_right_child.data(),
                                         tree_right_child.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const double*>(&cuda_leaf_value_,
                                         tree_leaf_value.data(),
                                         tree_leaf_value.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const double*>(&cuda_threshold_,
                                         tree_threshold.data(),
                                         tree_threshold.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int8_t*>(&cuda_decision_type_,
                                                   tree_decision_type.data(),
                                                   tree_decision_type.size(),
                                                   __FILE__,
                                                   __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int*>(&cuda_split_feature_index_,
                                                tree_split_feature_index.data(),
                                                tree_split_feature_index.size(),
                                                __FILE__,
                                                __LINE__);
}

}  // namespace LightGBM
