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
  */
  virtual void Init(
    const BoostingConfig* config,
    const Dataset* train_data,
    const ObjectiveFunction* object_function,
    const std::vector<const Metric*>& training_metrics) = 0;

  /*!
  * \brief Add a validation data
  * \param valid_data Validation data
  * \param valid_metrics Metric for validation data
  */
  virtual void AddDataset(const Dataset* valid_data,
    const std::vector<const Metric*>& valid_metrics) = 0;

  /*! \brief Training logic */
  virtual bool TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) = 0;

  virtual std::vector<double> GetEvalAt(int data_idx) const = 0;

  virtual const score_t* GetTrainingScore(data_size_t* out_len) const = 0;

  /*!
  * \brief Prediction for one record, not sigmoid transform
  * \param feature_values Feature value on this record
  * \return Prediction result for this record
  */
  virtual std::vector<double> PredictRaw(const double* feature_values) const = 0;

  /*!
  * \brief Prediction for one record, sigmoid transformation will be used if needed
  * \param feature_values Feature value on this record
  * \return Prediction result for this record
  */
  virtual std::vector<double> Predict(const double* feature_values) const = 0;
  
  /*!
  * \brief Predtion for one record with leaf index
  * \param feature_values Feature value on this record
  * \return Predicted leaf index for this record
  */
  virtual std::vector<int> PredictLeafIndex(
    const double* feature_values) const = 0;

  /*!
  * \brief save model to file
  */
  virtual void SaveModelToFile(bool is_finish, const char* filename) = 0;

  /*!
  * \brief Restore from a serialized string
  * \param model_str The string of model
  */
  virtual void ModelsFromString(const std::string& model_str) = 0;

  /*!
  * \brief Get max feature index of this model
  * \return Max feature index of this model
  */
  virtual int MaxFeatureIdx() const = 0;

  /*!
  * \brief Get index of label column
  * \return index of label column
  */
  virtual int LabelIdx() const = 0;

  /*!
  * \brief Get number of weak sub-models
  * \return Number of weak sub-models
  */
  virtual int NumberOfSubModels() const = 0;
  
  /*!
  * \brief Get number of classes
  * \return Number of classes
  */
  virtual int NumberOfClass() const = 0;

  /*!
  * \brief Set number of used model for prediction
  */
  virtual void SetNumUsedModel(int num_used_model) = 0;
  
  /*!
  * \brief Get Type name of this boosting object
  */
  virtual const char* Name() const = 0;

  /*!
  * \brief Create boosting object
  * \param type Type of boosting
  * \param config config for boosting
  * \param filename name of model file, if existing will continue to train from this model
  * \return The boosting object
  */
  static Boosting* CreateBoosting(BoostingType type, const char* filename);

  /*!
  * \brief Create boosting object from model file
  * \param filename name of model file
  * \return The boosting object
  */
  static Boosting* CreateBoosting(const char* filename);

};

}  // namespace LightGBM

#endif   // LightGBM_BOOSTING_H_
