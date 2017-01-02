#ifndef LIGHTGBM_BOOSTING_GBDT_H_
#define LIGHTGBM_BOOSTING_GBDT_H_

#include <LightGBM/boosting.h>
#include "score_updater.hpp"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <memory>

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
  * \brief Merge model from other boosting object
           Will insert to the front of current boosting object
  * \param other
  */
  void MergeFrom(const Boosting* other) override {
    auto other_gbdt = reinterpret_cast<const GBDT*>(other);
    // tmp move to other vector
    auto original_models = std::move(models_);
    models_ = std::vector<std::unique_ptr<Tree>>();
    // push model from other first
    for (const auto& tree : other_gbdt->models_) {
      auto new_tree = std::unique_ptr<Tree>(new Tree(*(tree.get())));
      models_.push_back(std::move(new_tree));
    }
    num_init_iteration_ = static_cast<int>(models_.size()) / num_class_;
    // push model in current object
    for (const auto& tree : original_models) {
      auto new_tree = std::unique_ptr<Tree>(new Tree(*(tree.get())));
      models_.push_back(std::move(new_tree));
    }
    num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_class_;
  }

  /*!
  * \brief Reset training data for current boosting
  * \param train_data Training data
  * \param object_function Training objective function
  * \param training_metrics Training metric
  */
  void ResetTrainingData(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* object_function, const std::vector<const Metric*>& training_metrics) override;

  /*!
  * \brief Adding a validation dataset
  * \param valid_data Validation dataset
  * \param valid_metrics Metrics for validation dataset
  */
  void AddValidDataset(const Dataset* valid_data,
       const std::vector<const Metric*>& valid_metrics) override;
  /*!
  * \brief Training logic
  * \param gradient nullptr for using default objective, otherwise use self-defined boosting
  * \param hessian nullptr for using default objective, otherwise use self-defined boosting
  * \param is_eval true if need evaluation or early stop
  * \return True if meet early stopping or cannot boosting
  */
  virtual bool TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) override;

  /*!
  * \brief Rollback one iteration
  */
  void RollbackOneIter() override;

  int GetCurrentIteration() const override { return iter_ + num_init_iteration_; }

  bool EvalAndCheckEarlyStopping() override;

  /*!
  * \brief Get evaluation result at data_idx data
  * \param data_idx 0: training data, 1: 1st validation data
  * \return evaluation result
  */
  std::vector<double> GetEvalAt(int data_idx) const override;

  /*!
  * \brief Get current training score
  * \param out_len length of returned score
  * \return training score
  */
  virtual const score_t* GetTrainingScore(int64_t* out_len) override;

  virtual int64_t GetNumPredictAt(int data_idx) const override {
    CHECK(data_idx >= 0 && data_idx <= static_cast<int>(valid_score_updater_.size()));
    data_size_t num_data = train_data_->num_data();
    if (data_idx > 0) {
      num_data = valid_score_updater_[data_idx - 1]->num_data();
    }
    return num_data * num_class_;
  }
  /*!
  * \brief Get prediction result at data_idx data
  * \param data_idx 0: training data, 1: 1st validation data
  * \param result used to store prediction result, should allocate memory before call this function
  * \param out_len lenght of returned score
  */
  void GetPredictAt(int data_idx, double* out_result, int64_t* out_len) override;

  /*!
  * \brief Prediction for one record without sigmoid transformation
  * \param feature_values Feature value on this record
  * \return Prediction result for this record
  */
  std::vector<double> PredictRaw(const double* feature_values) const override;

  /*!
  * \brief Prediction for one record with sigmoid transformation if enabled
  * \param feature_values Feature value on this record
  * \return Prediction result for this record
  */
  std::vector<double> Predict(const double* feature_values) const override;

  /*!
  * \brief Prediction for one record with leaf index
  * \param feature_values Feature value on this record
  * \return Predicted leaf index for this record
  */
  std::vector<int> PredictLeafIndex(const double* value) const override;

  /*!
  * \brief Dump model to json format string
  * \return Json format string of model
  */
  std::string DumpModel(int num_iteration) const override;

  /*!
  * \brief Save model to file
  * \param num_used_model Number of model that want to save, -1 means save all
  * \param is_finish Is training finished or not
  * \param filename Filename that want to save to
  */
  virtual void SaveModelToFile(int num_iterations, const char* filename) const override ;

  /*!
  * \brief Restore from a serialized string
  */
  void LoadModelFromString(const std::string& model_str) override;

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
  inline int NumberOfTotalModel() const override { return static_cast<int>(models_.size()); }

  /*!
  * \brief Get number of classes
  * \return Number of classes
  */
  inline int NumberOfClasses() const override { return num_class_; }

  /*!
  * \brief Set number of iterations for prediction
  */
  inline void SetNumIterationForPred(int num_iteration) override {
    num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_class_;
    if (num_iteration > 0) {
      num_iteration_for_pred_ = std::min(num_iteration, num_iteration_for_pred_);
    }
  }

  inline double GetLeafValue(int tree_idx, int leaf_idx) const {
    CHECK(tree_idx >= 0 && static_cast<size_t>(tree_idx) < models_.size());
    CHECK(leaf_idx >= 0 && leaf_idx < models_[tree_idx]->num_leaves());
    return models_[tree_idx]->LeafOutput(leaf_idx);
  }

  inline void SetLeafValue(int tree_idx, int leaf_idx, double val) {
    CHECK(tree_idx >= 0 && static_cast<size_t>(tree_idx) < models_.size());
    CHECK(leaf_idx >= 0 && leaf_idx < models_[tree_idx]->num_leaves());
    models_[tree_idx]->SetLeafOutput(leaf_idx, val);
  }

  /*!
  * \brief Get Type name of this boosting object
  */
  virtual const char* Name() const override { return "gbdt"; }

protected:
  /*!
  * \brief Implement bagging logic
  * \param iter Current interation
  */
  void Bagging(int iter);
  /*!
  * \brief updating score for out-of-bag data.
  *        Data should be update since we may re-bagging data on training
  * \param tree Trained tree of this iteration
  * \param curr_class Current class for multiclass training
  */
  void UpdateScoreOutOfBag(const Tree* tree, const int curr_class);
  /*!
  * \brief calculate the object function
  */
  void Boosting();
  /*!
  * \brief updating score after tree was trained
  * \param tree Trained tree of this iteration
  * \param curr_class Current class for multiclass training
  */
  virtual void UpdateScore(const Tree* tree, const int curr_class);
  /*!
  * \brief Print metric result of current iteration
  * \param iter Current interation
  * \return best_msg if met early_stopping
  */
  std::string OutputMetric(int iter);
  /*!
  * \brief Calculate feature importances
  * \param last_iter Last tree use to calculate
  */
  std::vector<std::pair<size_t, std::string>> FeatureImportance() const;
  /*! \brief current iteration */
  int iter_;
  /*! \brief Pointer to training data */
  const Dataset* train_data_;
  /*! \brief Config of gbdt */
  std::unique_ptr<BoostingConfig> gbdt_config_;
  /*! \brief Tree learner, will use this class to learn trees */
  std::unique_ptr<TreeLearner> tree_learner_;
  /*! \brief Objective function */
  const ObjectiveFunction* object_function_;
  /*! \brief Store and update training data's score */
  std::unique_ptr<ScoreUpdater> train_score_updater_;
  /*! \brief Metrics for training data */
  std::vector<const Metric*> training_metrics_;
  /*! \brief Store and update validation data's scores */
  std::vector<std::unique_ptr<ScoreUpdater>> valid_score_updater_;
  /*! \brief Metric for validation data */
  std::vector<std::vector<const Metric*>> valid_metrics_;
  /*! \brief Number of rounds for early stopping */
  int early_stopping_round_;
  /*! \brief Best iteration(s) for early stopping */
  std::vector<std::vector<int>> best_iter_;
  /*! \brief Best score(s) for early stopping */
  std::vector<std::vector<double>> best_score_;
  /*! \brief output message of best iteration */
  std::vector<std::vector<std::string>> best_msg_;
  /*! \brief Trained models(trees) */
  std::vector<std::unique_ptr<Tree>> models_;
  /*! \brief Max feature index of training data*/
  int max_feature_idx_;
  /*! \brief First order derivative of training data */
  std::vector<score_t> gradients_;
  /*! \brief Secend order derivative of training data */
  std::vector<score_t> hessians_;
  /*! \brief Store the data indices of out-of-bag */
  std::vector<data_size_t> out_of_bag_data_indices_;
  /*! \brief Number of out-of-bag data */
  data_size_t out_of_bag_data_cnt_;
  /*! \brief Store the indices of in-bag data */
  std::vector<data_size_t> bag_data_indices_;
  /*! \brief Number of in-bag data */
  data_size_t bag_data_cnt_;
  /*! \brief Number of training data */
  data_size_t num_data_;
  /*! \brief Number of classes */
  int num_class_;
  /*! \brief Random generator, used for bagging */
  Random random_;
  /*!
  *   \brief Sigmoid parameter, used for prediction.
  *          if > 0 means output score will transform by sigmoid function
  */
  double sigmoid_;
  /*! \brief Index of label column */
  data_size_t label_idx_;
  /*! \brief number of used model */
  int num_iteration_for_pred_;
  /*! \brief Shrinkage rate for one iteration */
  double shrinkage_rate_;
  /*! \brief Number of loaded initial models */
  int num_init_iteration_;
  /*! \brief Feature names */
  std::vector<std::string> feature_names_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_GBDT_H_
