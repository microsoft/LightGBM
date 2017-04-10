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
  * \param is_raw_score True if need to predict result with raw score
  * \param is_predict_leaf_index True if output leaf index instead of prediction score
  */
  Predictor(const Boosting* boosting, int num_iteration, bool is_raw_score, bool is_predict_leaf_index) {
    boosting_ = boosting;
    num_pred_one_row_ = boosting_->NumPredictOneRow(num_iteration, is_predict_leaf_index);
    num_features_ = boosting_->MaxFeatureIdx() + 1;
    features_ = std::vector<double>(num_features_);
    if (is_predict_leaf_index) {
      predict_fun_ = [this](const std::vector<std::pair<int, double>>& features, double* output) {
        PutFeatureValuesToBuffer(features);
        // get result for leaf index
        boosting_->PredictLeafIndex(features_.data(), output);
      };

    } else {
      if (is_raw_score) {
        predict_fun_ = [this](const std::vector<std::pair<int, double>>& features, double* output) {
          PutFeatureValuesToBuffer(features);
          // get result without sigmoid transformation
          boosting_->PredictRaw(features_.data(), output);
        };
      } else {
        predict_fun_ = [this](const std::vector<std::pair<int, double>>& features, double* output) {
          PutFeatureValuesToBuffer(features);
          boosting_->Predict(features_.data(), output);
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
    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(data_filename, has_header, num_features_, boosting_->LabelIdx()));

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
  void PutFeatureValuesToBuffer(const std::vector<std::pair<int, double>>& features) {
    std::memset(features_.data(), 0, sizeof(double)*num_features_);
    // put feature value
    int loop_size = static_cast<int>(features.size());
    #pragma omp parallel for schedule(static, 512) if(loop_size >= 1024) 
    for (int i = 0; i < loop_size; ++i) {
      if (features[i].first < num_features_) {
        features_[features[i].first] = features[i].second;
      }
    }
  }
  /*! \brief Boosting model */
  const Boosting* boosting_;
  /*! \brief Buffer for feature values */
  std::vector<double> features_;
  /*! \brief Number of features */
  int num_features_;
  /*! \brief function for prediction */
  PredictFunction predict_fun_;
  int num_pred_one_row_;
};

}  // namespace LightGBM

#endif   // LightGBM_PREDICTOR_HPP_
