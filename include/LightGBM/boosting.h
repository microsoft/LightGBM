#ifndef LIGHTGBM_BOOSTING_H_
#define LIGHTGBM_BOOSTING_H_

#include <LightGBM/meta.h>
#include <LightGBM/config.h>

#include <vector>
#include <string>

namespace LightGBM {

/*! \brief forward declaration */
class Dataset;
class ObjectiveFunction;
class Metric;

/*!
* \brief The interface for Boosting
*/
class Boosting {
public:
  /*! \brief virtual destructor */
  virtual ~Boosting() {}

  /*!
  * \brief Initialization logic
  * \param config Configs for boosting
  * \param train_data Training data
  * \param object_function Training objective function
  * \param training_metrics Training metric
  * \param output_model_filename Filename of output model
  */
  virtual void Init(const Dataset* train_data,
    const ObjectiveFunction* object_function,
    const std::vector<const Metric*>& training_metrics,
    const char* output_model_filename) = 0;

  /*!
  * \brief Add a validation data
  * \param valid_data Validation data
  * \param valid_metrics Metric for validation data
  */
  virtual void AddDataset(const Dataset* valid_data,
    const std::vector<const Metric*>& valid_metrics) = 0;

  /*! \brief Training logic */
  virtual void Train() = 0;

  /*!
  * \brief Prediction for one record, not sigmoid transform
  * \param feature_values Feature value on this record
  * \return Prediction result for this record
  */
  virtual double PredictRaw(const double * feature_values) const = 0;

  /*!
  * \brief Prediction for one record, sigmoid transformation will be used if needed
  * \param feature_values Feature value on this record
  * \return Prediction result for this record
  */
  virtual double Predict(const double * feature_values) const = 0;
  
  /*!
  * \brief Predtion for one record with leaf index
  * \param feature_values Feature value on this record
  * \return Predicted leaf index for this record
  */
  virtual std::vector<int> PredictLeafIndex(const double * feature_values) const = 0;
  
  /*!
  * \brief Serialize models by string
  * \return String output of tranined model
  */
  virtual std::string ModelsToString() const = 0;

  /*!
  * \brief Restore from a serialized string
  * \param model_str The string of model
  */
  virtual void ModelsFromString(const std::string& model_str, int num_used_model) = 0;

  /*!
  * \brief Calculate feature importances
  * \param last_iter Last tree use to calculate
  */
  virtual std::string FeatureImportance(int last_iter) const = 0;

  /*!
  * \brief Get max feature index of this model
  * \return Max feature index of this model
  */
  virtual int MaxFeatureIdx() const = 0;

  /*!
  * \brief Get number of weak sub-models
  * \return Number of weak sub-models
  */
  virtual int NumberOfSubModels() const = 0;

  /*!
  * \brief Create boosting object
  * \param type Type of boosting
  * \return The boosting object
  */
  static Boosting* CreateBoosting(BoostingType type,
    const BoostingConfig* config);
};

}  // namespace LightGBM

#endif   // LightGBM_BOOSTING_H_
