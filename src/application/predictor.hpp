#ifndef LIGHTGBM_PREDICTOR_HPP_
#define LIGHTGBM_PREDICTOR_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/boosting.h>
#include <LightGBM/utils/text_reader.h>
#include <LightGBM/dataset.h>

#include <LightGBM/utils/openmp_wrapper.h>

#include <cstring>
#include <cstdio>
#include <vector>
#include <utility>
#include <functional>
#include <string>
#include <memory>

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
  * \param is_predict_leaf_index True if output leaf index instead of prediction score
  */
  Predictor(Boosting* boosting, int num_iteration,
            bool is_raw_score, bool is_predict_leaf_index) {

    boosting->InitPredict(num_iteration);
    boosting_ = boosting;
    num_pred_one_row_ = boosting_->NumPredictOneRow(num_iteration, is_predict_leaf_index);
    predict_buf_ = std::vector<double>(boosting_->MaxFeatureIdx() + 1, 0.0f);

    if (is_predict_leaf_index) {
      predict_fun_ = [this](const std::vector<std::pair<int, double>>& features, double* output) {
        CopyToPredictBuffer(features);
        // get result for leaf index
        boosting_->PredictLeafIndex(predict_buf_.data(), output);
        ClearPredictBuffer(features);
      };

    } else {
      if (is_raw_score) {
        predict_fun_ = [this](const std::vector<std::pair<int, double>>& features, double* output) {
          CopyToPredictBuffer(features);
          boosting_->PredictRaw(predict_buf_.data(), output);
          ClearPredictBuffer(features);
        };
      } else {
        predict_fun_ = [this](const std::vector<std::pair<int, double>>& features, double* output) {
          CopyToPredictBuffer(features);
          boosting_->Predict(predict_buf_.data(), output);
          ClearPredictBuffer(features);
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
  void Predict(const char* data_filename, const char* result_filename, bool has_header) {
    FILE* result_file;

    #ifdef _MSC_VER
    fopen_s(&result_file, result_filename, "w");
    #else
    result_file = fopen(result_filename, "w");
    #endif

    if (result_file == NULL) {
      Log::Fatal("Prediction results file %s doesn't exist", data_filename);
    }
    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(data_filename, has_header, boosting_->MaxFeatureIdx() + 1, boosting_->LabelIdx()));

    if (parser == nullptr) {
      Log::Fatal("Could not recognize the data format of data file %s", data_filename);
    }

    // function for parse data
    std::function<void(const char*, std::vector<std::pair<int, double>>*)> parser_fun;
    double tmp_label;
    parser_fun = [this, &parser, &tmp_label]
    (const char* buffer, std::vector<std::pair<int, double>>* feature) {
      parser->ParseOneLine(buffer, feature, &tmp_label);
    };

    std::function<void(data_size_t, const std::vector<std::string>&)> process_fun =
      [this, &parser_fun, &result_file]
    (data_size_t, const std::vector<std::string>& lines) {
      std::vector<std::pair<int, double>> oneline_features;
      for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); ++i) {
        oneline_features.clear();
        // parser
        parser_fun(lines[i].c_str(), &oneline_features);
        // predict
        std::vector<double> result(num_pred_one_row_);
        predict_fun_(oneline_features, result.data());
        auto str_result = Common::Join<double>(result, "\t");
        fprintf(result_file, "%s\n", str_result.c_str());
      }
    };
    TextReader<data_size_t> predict_data_reader(data_filename, has_header);
    predict_data_reader.ReadAllAndProcessParallel(process_fun);
    fclose(result_file);
  }

private:

  void CopyToPredictBuffer(const std::vector<std::pair<int, double>>& features) {
    int loop_size = static_cast<int>(features.size());
    #pragma omp parallel for schedule(static,128) if (loop_size >= 256)
    for (int i = 0; i < loop_size; ++i) {
      predict_buf_[features[i].first] = features[i].second;
    }
  }

  void ClearPredictBuffer(const std::vector<std::pair<int, double>>& features) {
    if (features.size() < static_cast<size_t>(predict_buf_.size() / 2)) {
      std::memset(predict_buf_.data(), 0, sizeof(double)*(predict_buf_.size()));
    } else {
      int loop_size = static_cast<int>(features.size());
      #pragma omp parallel for schedule(static,128) if (loop_size >= 256)
      for (int i = 0; i < loop_size; ++i) {
        predict_buf_[features[i].first] = 0.0f;
      }
    }
  }

  /*! \brief Boosting model */
  const Boosting* boosting_;
  /*! \brief function for prediction */
  PredictFunction predict_fun_;
  int num_pred_one_row_;
  std::vector<double> predict_buf_;
};

}  // namespace LightGBM

#endif   // LightGBM_PREDICTOR_HPP_
