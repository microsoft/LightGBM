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
  * \brief Merge model from other boosting object
           Will insert to the front of current boosting object
  * \param other
  */
  virtual void MergeFrom(const Boosting* other) = 0;

  /*!
  * \brief Reset training data for current boosting
  * \param config Configs for boosting
  * \param train_data Training data
  * \param object_function Training objective function
  * \param training_metrics Training metric
  */
  virtual void ResetTrainingData(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* object_function, const std::vector<const Metric*>& training_metrics) = 0;

  /*!
  * \brief Add a validation data
  * \param valid_data Validation data
  * \param valid_metrics Metric for validation data
  */
  virtual void AddValidDataset(const Dataset* valid_data,
    const std::vector<const Metric*>& valid_metrics) = 0;

  /*!
  * \brief Training logic
  * \param gradient nullptr for using default objective, otherwise use self-defined boosting
  * \param hessian nullptr for using default objective, otherwise use self-defined boosting
  * \param is_eval true if need evaluation or early stop
  * \return True if meet early stopping or cannot boosting
  */
  virtual bool TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) = 0;

  /*!
  * \brief Rollback one iteration
  */
  virtual void RollbackOneIter() = 0;

  /*!
  * \brief return current iteration
  */
  virtual int GetCurrentIteration() const = 0;

  /*!
  * \brief Eval metrics and check is met early stopping or not
  */
  virtual bool EvalAndCheckEarlyStopping() = 0;
  /*!
  * \brief Get evaluation result at data_idx data
  * \param data_idx 0: training data, 1: 1st validation data
  * \return evaluation result
  */
  virtual std::vector<double> GetEvalAt(int data_idx) const = 0;

  /*!
  * \brief Get current training score
  * \param out_len length of returned score
  * \return training score
  */
  virtual const score_t* GetTrainingScore(int64_t* out_len) = 0;

  /*!
  * \brief Get prediction result at data_idx data
  * \param data_idx 0: training data, 1: 1st validation data
  * \return out_len lenght of returned score
  */
  virtual int64_t GetNumPredictAt(int data_idx) const = 0;
  /*!
  * \brief Get prediction result at data_idx data
  * \param data_idx 0: training data, 1: 1st validation data
  * \param result used to store prediction result, should allocate memory before call this function
  * \param out_len lenght of returned score
  */
  virtual void GetPredictAt(int data_idx, double* result, int64_t* out_len) = 0;

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
  * \brief Dump model to json format string
  * \return Json format string of model
  */
  virtual std::string DumpModel(int num_iteration) const = 0;

  /*!
  * \brief Save model to file
  * \param num_used_model Number of model that want to save, -1 means save all
  * \param is_finish Is training finished or not
  * \param filename Filename that want to save to
  */
  virtual void SaveModelToFile(int num_iterations, const char* filename) const = 0;

  /*!
  * \brief Restore from a serialized string
  * \param model_str The string of model
  */
  virtual void LoadModelFromString(const std::string& model_str) = 0;

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
  virtual int NumberOfTotalModel() const = 0;
  
  /*!
  * \brief Get number of classes
  * \return Number of classes
  */
  virtual int NumberOfClasses() const = 0;

  /*!
  * \brief Set number of used model for prediction
  */
  virtual void SetNumIterationForPred(int num_iteration) = 0;
  
  /*!
  * \brief Get Type name of this boosting object
  */
  virtual const char* Name() const = 0;

  Boosting() = default;
  /*! \brief Disable copy */
  Boosting& operator=(const Boosting&) = delete;
  /*! \brief Disable copy */
  Boosting(const Boosting&) = delete;

  static void LoadFileToBoosting(Boosting* boosting, const char* filename);

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
