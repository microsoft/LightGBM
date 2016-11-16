#ifndef LIGHTGBM_PREDICTOR_HPP_
#define LIGHTGBM_PREDICTOR_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/boosting.h>
#include <LightGBM/utils/text_reader.h>
#include <LightGBM/dataset.h>

#include <omp.h>

#include <cstring>
#include <cstdio>
#include <vector>
#include <utility>
#include <functional>
#include <string>
#include <memory>

namespace LightGBM {

/*!
* \brief Used to prediction data with input model
*/
class Predictor {
public:
  /*!
  * \brief Constructor
  * \param boosting Input boosting model
  * \param is_raw_score True if need to predict result with raw score
  * \param predict_leaf_index True if output leaf index instead of prediction score
  */
  Predictor(const Boosting* boosting, bool is_raw_score, bool is_predict_leaf_index) {
    boosting_ = boosting;
    num_features_ = boosting_->MaxFeatureIdx() + 1;
#pragma omp parallel
#pragma omp master
    {
      num_threads_ = omp_get_num_threads();
    }
    for (int i = 0; i < num_threads_; ++i) {
      features_.push_back(std::vector<double>(num_features_));
    }

    if (is_predict_leaf_index) {
      predict_fun_ = [this](const std::vector<std::pair<int, double>>& features) {
        const int tid = PutFeatureValuesToBuffer(features);
        // get result for leaf index
        auto result = boosting_->PredictLeafIndex(features_[tid].data());
        return std::vector<double>(result.begin(), result.end());
      };
    } else {
      if (is_raw_score) {
        predict_fun_ = [this](const std::vector<std::pair<int, double>>& features) {
          const int tid = PutFeatureValuesToBuffer(features);
          // get result without sigmoid transformation
          return boosting_->PredictRaw(features_[tid].data());
        };
      } else {
        predict_fun_ = [this](const std::vector<std::pair<int, double>>& features) {
          const int tid = PutFeatureValuesToBuffer(features);
          return boosting_->Predict(features_[tid].data());
        };
      }
    }
  }
  /*!
  * \brief Destructor
  */
  ~Predictor() {
  }

  inline const PredictFunction& GetPredictFunction() {
    return predict_fun_;
  }

  /*!
  * \brief predicting on data, then saving result to disk
  * \param data_filename Filename of data
  * \param has_label True if this data contains label
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
      std::vector<std::string> pred_result(lines.size(), "");
#pragma omp parallel for schedule(static) private(oneline_features)
      for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); ++i) {
        oneline_features.clear();
        // parser
        parser_fun(lines[i].c_str(), &oneline_features);
        // predict
        pred_result[i] = Common::Join<double>(predict_fun_(oneline_features), '\t');
      }

      for (size_t i = 0; i < pred_result.size(); ++i) {
        fprintf(result_file, "%s\n", pred_result[i].c_str());
      }
    };
    TextReader<data_size_t> predict_data_reader(data_filename, has_header);
    predict_data_reader.ReadAllAndProcessParallel(process_fun);

    fclose(result_file);
  }

private:
  int PutFeatureValuesToBuffer(const std::vector<std::pair<int, double>>& features) {
    int tid = omp_get_thread_num();
    // init feature value
    std::memset(features_[tid].data(), 0, sizeof(double)*num_features_);
    // put feature value
    for (const auto& p : features) {
      if (p.first < num_features_) {
        features_[tid][p.first] = p.second;
      }
    }
    return tid;
  }
  /*! \brief Boosting model */
  const Boosting* boosting_;
  /*! \brief Buffer for feature values */
  std::vector<std::vector<double>> features_;
  /*! \brief Number of features */
  int num_features_;
  /*! \brief Number of threads */
  int num_threads_;
  /*! \brief function for prediction */
  PredictFunction predict_fun_;
};

}  // namespace LightGBM

#endif   // LightGBM_PREDICTOR_HPP_
