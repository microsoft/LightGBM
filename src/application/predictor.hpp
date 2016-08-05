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

namespace LightGBM {

/*!
* \brief Used to prediction data with input model
*/
class Predictor {
public:
  /*!
  * \brief Constructor
  * \param boosting Input boosting model
  * \param is_sigmoid True if need to predict result with sigmoid transform(if needed, like binary classification)
  */
  Predictor(const Boosting* boosting, bool is_simgoid)
    : is_simgoid_(is_simgoid) {
    boosting_ = boosting;
    num_features_ = boosting_->MaxFeatureIdx() + 1;
#pragma omp parallel
#pragma omp master
    {
      num_threads_ = omp_get_num_threads();
    }
    features_ = new double*[num_threads_];
    for (int i = 0; i < num_threads_; ++i) {
      features_[i] = new double[num_features_];
    }
  }
  /*!
  * \brief Destructor
  */
  ~Predictor() {
    if (features_ != nullptr) {
      for (int i = 0; i < num_threads_; ++i) {
        delete[] features_[i];
      }
      delete[] features_;
    }
  }

  /*!
  * \brief prediction for one record, only raw result(not sigmoid transform)
  * \param features Feature for this record
  * \return Prediction result
  */
  double PredictRawOneLine(const std::vector<std::pair<int, double>>& features) {
    const int tid = omp_get_thread_num();
    // init feature value
    std::memset(features_[tid], 0, sizeof(double)*num_features_);
    // put feature value
    for (const auto& p : features) {
      if (p.first < num_features_) {
        features_[tid][p.first] = p.second;
      }
    }
    // get result without sigmoid transform
    return boosting_->PredictRaw(features_[tid]);
  }

  /*!
  * \brief prediction for one record, will use sigmoid transform if needed(only needs in binary classification now)
  * \param features Feature for this record
  * \return Prediction result
  */
  double PredictOneLine(const std::vector<std::pair<int, double>>& features) {
    const int tid = omp_get_thread_num();
    // init feature value
    std::memset(features_[tid], 0, sizeof(double)*num_features_);
    // put feature value
    for (const auto& p : features) {
      if (p.first < num_features_) {
        features_[tid][p.first] = p.second;
      }
    }
    // get result with sigmoid transform
    return boosting_->Predict(features_[tid]);
  }
  /*!
  * \brief prediction for a data, and save result
  * \param data_filename Filename of data
  * \param has_label True if this data contains label
  * \param result_filename Filename of output result
  */
  void Predict(const char* data_filename, bool has_label, const char* result_filename) {
    FILE* result_file;

#ifdef _MSC_VER
    fopen_s(&result_file, result_filename, "w");
#else
    result_file = fopen(result_filename, "w");
#endif

    if (result_file == NULL) {
      Log::Stderr("predition result file %s doesn't exists", data_filename);
    }

    Parser* parser = Parser::CreateParser(data_filename);

    if (parser == nullptr) {
      Log::Stderr("can regonise input data format, filename %s", data_filename);
    }

    // function for parse data
    std::function<void(const char*, std::vector<std::pair<int, double>>*)> parser_fun;
    double tmp_label;
    if (has_label) {
      // parse function with label
      parser_fun = [this, &parser, &tmp_label]
      (const char* buffer, std::vector<std::pair<int, double>>* feature) {
        parser->ParseOneLine(buffer, feature, &tmp_label);
      };
      Log::Stdout("start prediction for data %s, and data has label", data_filename);
    } else {
      // parse function without label
      parser_fun = [this, &parser]
      (const char* buffer, std::vector<std::pair<int, double>>* feature) {
        parser->ParseOneLine(buffer, feature);
      };
      Log::Stdout("start prediction for data %s, and data doesn't has label", data_filename);
    }
    std::function<double(const std::vector<std::pair<int, double>>&)> predict_fun;
    if (is_simgoid_) {
      predict_fun = [this](const std::vector<std::pair<int, double>>& features) {
        return PredictOneLine(features);
      };
    } else {
      predict_fun = [this](const std::vector<std::pair<int, double>>& features) {
        return PredictRawOneLine(features);
      };
    }
    std::function<void(data_size_t, const std::vector<std::string>&)> process_fun =
      [this, &parser_fun, &predict_fun, &result_file]
    (data_size_t, const std::vector<std::string>& lines) {
      std::vector<std::pair<int, double>> oneline_features;
      std::vector<double> pred_result(lines.size(), 0.0f);
#pragma omp parallel for schedule(static) private(oneline_features)
      for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); i++) {
        oneline_features.clear();
        // parser
        parser_fun(lines[i].c_str(), &oneline_features);
        // predict
        pred_result[i] = predict_fun(oneline_features);
      }

      for (size_t i = 0; i < pred_result.size(); ++i) {
        fprintf(result_file, "%f\n", pred_result[i]);
      }
    };

    TextReader<data_size_t> predict_data_reader(data_filename);
    predict_data_reader.ReadAllAndProcessParallel(process_fun);

    fclose(result_file);
    delete parser;
  }

private:
  /*! \brief Boosting model */
  const Boosting* boosting_;
  /*! \brief Buffer for feature values */
  double** features_;
  /*! \brief Number of features */
  int num_features_;
  /*! \brief True if need to predict result with sigmoid transform */
  bool is_simgoid_;
  /*! \brief Number of threads */
  int num_threads_;
};

}  // namespace LightGBM

#endif  #endif  // LightGBM_PREDICTOR_HPP_
