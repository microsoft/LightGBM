#ifndef LIGHTGBM_BOOSTING_GBDT_H_
#define LIGHTGBM_BOOSTING_GBDT_H_

#include <LightGBM/boosting.h>
#include "score_updater.hpp"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>

namespace LightGBM {
/*!
* \brief GBDT algorithm implementation. including Training, prediction, bagging.
*/
class GBDT: public Boosting {
public:
  /*!
  * \brief Constructor
  */
  GBDT();
  /*!
  * \brief Destructor
  */
  ~GBDT();
  /*!
  * \brief Initialization logic
  * \param config Config for boosting
  * \param train_data Training data
  * \param object_function Training objective function
  * \param training_metrics Training metrics
  * \param output_model_filename Filename of output model
  */
  void Init(const BoostingConfig* gbdt_config, const Dataset* train_data, const ObjectiveFunction* object_function,
                             const std::vector<const Metric*>& training_metrics)
                                                                       override;
  /*!
  * \brief Adding a validation dataset
  * \param valid_data Validation dataset
  * \param valid_metrics Metrics for validation dataset
  */
  void AddDataset(const Dataset* valid_data,
       const std::vector<const Metric*>& valid_metrics) override;
  /*!
  * \brief one training iteration
  */
  bool TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) override;
  /*!
  * \brief Predtion for one record without sigmoid transformation
  * \param feature_values Feature value on this record
  * \param num_used_model Number of used model
  * \return Prediction result for this record
  */
  double PredictRaw(const double * feature_values, size_t num_used_model) const override;

  /*!
  * \brief Predtion for one record with sigmoid transformation if enabled
  * \param feature_values Feature value on this record
  * \param num_used_model Number of used model
  * \return Prediction result for this record
  */
  double Predict(const double * feature_values, size_t num_used_model) const override;
  
  /*!
  * \brief Predtion for one record with leaf index
  * \param feature_values Feature value on this record
  * \param num_used_model Number of used model
  * \return Predicted leaf index for this record
  */
 std::vector<int> PredictLeafIndex(const double* value, size_t num_used_model) const override;
  
  /*!
  * \brief Serialize models by string
  * \return String output of tranined model
  */
  void SaveModelToFile(bool is_finish, const char* filename) override;
  /*!
  * \brief Restore from a serialized string
  */
  void ModelsFromString(const std::string& model_str) override;
  /*!
  * \brief Get max feature index of this model
  * \return Max feature index of this model
  */
  inline int MaxFeatureIdx() const override { return max_feature_idx_; }

  /*!
  * \brief Get index of label column
  * \return index of label column
  */
  inline int LabelIdx() const override { return label_idx_; }

  /*!
  * \brief Get number of weak sub-models
  * \return Number of weak sub-models
  */
  inline int NumberOfSubModels() const override { return static_cast<int>(models_.size()); }

private:
  /*!
  * \brief Implement bagging logic
  * \param iter Current interation
  */
  void Bagging(int iter);
  /*!
  * \brief updating score for out-of-bag data.
  *        Data should be update since we may re-bagging data on training
  * \param tree Trained tree of this iteration
  */
  void UpdateScoreOutOfBag(const Tree* tree);
  /*!
  * \brief calculate the object function
  */
  void Boosting();
  /*!
  * \brief updating score after tree was trained
  * \param tree Trained tree of this iteration
  */
  void UpdateScore(const Tree* tree);
  /*!
  * \brief Print metric result of current iteration
  * \param iter Current interation
  */
  bool OutputMetric(int iter);
  /*!
  * \brief Calculate feature importances
  * \param last_iter Last tree use to calculate
  */
  std::string FeatureImportance() const;
  /*! \brief current iteration */
  int iter_;
  /*! \brief Pointer to training data */
  const Dataset* train_data_;
  /*! \brief Config of gbdt */
  const GBDTConfig* gbdt_config_;
  /*! \brief Tree learner, will use this class to learn trees */
  TreeLearner* tree_learner_;
  /*! \brief Objective function */
  const ObjectiveFunction* object_function_;
  /*! \brief Store and update training data's score */
  ScoreUpdater* train_score_updater_;
  /*! \brief Metrics for training data */
  std::vector<const Metric*> training_metrics_;
  /*! \brief Store and update validation data's scores */
  std::vector<ScoreUpdater*> valid_score_updater_;
  /*! \brief Metric for validation data */
  std::vector<std::vector<const Metric*>> valid_metrics_;
  /*! \brief Number of rounds for early stopping */
  int early_stopping_round_;
  /*! \brief Best score(s) for early stopping */
  std::vector<std::vector<int>> best_iter_;
  std::vector<std::vector<score_t>> best_score_;
  /*! \brief Trained models(trees) */
  std::vector<Tree*> models_;
  /*! \brief Max feature index of training data*/
  int max_feature_idx_;
  /*! \brief First order derivative of training data */
  score_t* gradients_;
  /*! \brief Secend order derivative of training data */
  score_t* hessians_;
  /*! \brief Store the data indices of out-of-bag */
  data_size_t* out_of_bag_data_indices_;
  /*! \brief Number of out-of-bag data */
  data_size_t out_of_bag_data_cnt_;
  /*! \brief Store the indices of in-bag data */
  data_size_t* bag_data_indices_;
  /*! \brief Number of in-bag data */
  data_size_t bag_data_cnt_;
  /*! \brief Number of traning data */
  data_size_t num_data_;
  /*! \brief Random generator, used for bagging */
  Random random_;
  /*!
  *   \brief Sigmoid parameter, used for prediction.
  *          if > 0 meas output score will transform by sigmoid function
  */
  double sigmoid_;
  /*! \brief Index of label column */
  data_size_t label_idx_;
  /*! \brief Saved number of models */
  int saved_model_size_ = -1;
  /*! \brief File to write models */
  std::ofstream model_output_file_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_GBDT_H_
