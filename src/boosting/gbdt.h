#ifndef LIGHTGBM_BOOSTING_GBDT_H_
#define LIGHTGBM_BOOSTING_GBDT_H_

#include <LightGBM/boosting.h>
#include "score_updater.hpp"

#include <cstdio>
#include <vector>
#include <string>

namespace LightGBM {
/*!
* \brief GBDT algorithm implementation. including Training, prediction, bagging.
*/
class GBDT: public Boosting {
public:
  /*!
  * \brief Constructor
  * \param config Config of GBDT
  */
  explicit GBDT(const BoostingConfig* config);
  /*!
  * \brief Destructor
  */
  ~GBDT();
  /*!
  * \brief Initial logic
  * \param config Config for boosting
  * \param train_data Training data
  * \param object_function Training objective function
  * \param training_metrics Training metrics
  * \param output_model_filename Filename of output model
  */
  void Init(const Dataset* train_data, const ObjectiveFunction* object_function,
                             const std::vector<const Metric*>& training_metrics,
                                              const char* output_model_filename)
                                                                       override;
  /*!
  * \brief Add a validation data
  * \param valid_data Validation data
  * \param valid_metrics Metrics for validation data
  */
  void AddDataset(const Dataset* valid_data,
       const std::vector<const Metric*>& valid_metrics) override;
  /*!
  * \brief one training iteration
  */
  void Train() override;
  /*!
  * \brief Predtion for one record, not use sigmoid
  * \param feature_values Feature value on this record
  * \return Prediction result for this record
  */
  double PredictRaw(const double * feature_values) const override;

  /*!
  * \brief Predtion for one record, will use sigmoid transform if needed
  * \param feature_values Feature value on this record
  * \return Prediction result for this record
  */
  double Predict(const double * feature_values) const override;
  /*!
  * \brief Serialize models by string
  * \return String output of tranined model
  */
  std::string ModelsToString() const override;
  /*!
  * \brief Restore from a serialized string
  * \param model_str The string of model
  */
  void ModelsFromString(const std::string& model_str, int num_used_model) override;
  /*!
  * \brief Get max feature index of this model
  * \return Max feature index of this model
  */
  inline int MaxFeatureIdx() const override { return max_feature_idx_; }
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
  * \brief update score for out-of-bag data.
  * It is necessary for this update, since we may re-bagging data on training
  * \param tree Trained tree of this iteration
  */
  void UpdateScoreOutOfBag(const Tree* tree);
  /*!
  * \brief calculate the object function
  */
  void Boosting();
  /*!
  * \brief train one tree
  * \return Trained tree of this iteration
  */
  Tree* TrainOneTree();
  /*!
  * \brief update score after tree trained
  * \param tree Trained tree of this iteration
  */
  void UpdateScore(const Tree* tree);
  /*!
  * \brief Print Metric result of current iteration
  * \param iter Current interation
  */
  void OutputMetric(int iter);

  /*! \brief Pointer to training data */
  const Dataset* train_data_;
  /*! \brief Config of gbdt */
  const GBDTConfig* gbdt_config_;
  /*! \brief Tree learner, will use tihs class to learn trees */
  TreeLearner* tree_learner_;
  /*! \brief Objective function */
  const ObjectiveFunction* object_function_;
  /*! \brief Store and update traning data's score */
  ScoreUpdater* train_score_updater_;
  /*! \brief Metrics for training data */
  std::vector<const Metric*> training_metrics_;
  /*! \brief Store and update validation data's scores */
  std::vector<ScoreUpdater*> valid_score_updater_;
  /*! \brief Metric for validation data */
  std::vector<std::vector<const Metric*>> valid_metrics_;
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
  /*! \brief The filename that the models will save to */
  FILE * output_model_file;
  /*!
  *   \brief Sigmoid parameter, used for prediction.
  *          if > 0 meas output score will transform by sigmoid function
  */
  double sigmoid_;
};

}  // namespace LightGBM
#endif  #endif  // LightGBM_BOOSTING_GBDT_H_
