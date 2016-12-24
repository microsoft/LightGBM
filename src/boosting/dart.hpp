#ifndef LIGHTGBM_BOOSTING_DART_H_
#define LIGHTGBM_BOOSTING_DART_H_

#include <LightGBM/boosting.h>
#include "score_updater.hpp"
#include "gbdt.h"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>

namespace LightGBM {
/*!
* \brief DART algorithm implementation. including Training, prediction, bagging.
*/
class DART: public GBDT {
public:
  /*!
  * \brief Constructor
  */
  DART(): GBDT() { }
  /*!
  * \brief Destructor
  */
  ~DART() { }
  /*!
  * \brief Initialization logic
  * \param config Config for boosting
  * \param train_data Training data
  * \param object_function Training objective function
  * \param training_metrics Training metrics
  * \param output_model_filename Filename of output model
  */
  void Init(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* object_function,
    const std::vector<const Metric*>& training_metrics) override {
    GBDT::Init(config, train_data, object_function, training_metrics);
    random_for_drop_ = Random(gbdt_config_->drop_seed);
  }
  /*!
  * \brief one training iteration
  */
  bool TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) override {
    is_update_score_cur_iter_ = false;
    GBDT::TrainOneIter(gradient, hessian, false);
    // normalize
    Normalize();
    if (is_eval) {
      return EvalAndCheckEarlyStopping();
    } else {
      return false;
    }
  }

  void ResetTrainingData(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* object_function,
    const std::vector<const Metric*>& training_metrics) {
    GBDT::ResetTrainingData(config, train_data, object_function, training_metrics);
    shrinkage_rate_ = gbdt_config_->learning_rate / (gbdt_config_->learning_rate + static_cast<double>(drop_index_.size()));
  }

  /*!
  * \brief Get current training score
  * \param out_len length of returned score
  * \return training score
  */
  const score_t* GetTrainingScore(int64_t* out_len) override {
    if (!is_update_score_cur_iter_) {
      // only drop one time in one iteration
      DroppingTrees();
      is_update_score_cur_iter_ = true;
    }
    *out_len = static_cast<int64_t>(train_score_updater_->num_data()) * num_class_;
    return train_score_updater_->score();
  }

  /*!
  * \brief Get Type name of this boosting object
  */
  const char* Name() const override { return "dart"; }

private:
  /*!
  * \brief drop trees based on drop_rate
  */
  void DroppingTrees() {
    drop_index_.clear();
    // select dropping tree indexes based on drop_rate
    // if drop rate is too small, skip this step, drop one tree randomly
    if (gbdt_config_->drop_rate > kEpsilon) {
      for (int i = 0; i < iter_; ++i) {
        if (random_for_drop_.NextDouble() < gbdt_config_->drop_rate) {
          drop_index_.push_back(i);
        }
      }
    }
    // binomial-plus-one, at least one tree will be dropped
    if (drop_index_.empty()) {
      drop_index_ = random_for_drop_.Sample(iter_, 1);
    }
    // drop trees
    for (auto i : drop_index_) {
      for (int curr_class = 0; curr_class < num_class_; ++curr_class) {
        auto curr_tree = i * num_class_ + curr_class;
        models_[curr_tree]->Shrinkage(-1.0);
        train_score_updater_->AddScore(models_[curr_tree].get(), curr_class);
      }
    }
    shrinkage_rate_ = gbdt_config_->learning_rate / (gbdt_config_->learning_rate + static_cast<double>(drop_index_.size()));
  }
  /*!
  * \brief normalize dropped trees
  * NOTE: num_drop_tree(k), learning_rate(lr), shrinkage_rate_ = lr / (k + lr)
  *       step 1: shrink tree to -1 -> drop tree
  *       step 2: shrink tree to k / (k + lr) - 1 from -1
  *               -> normalize for valid data
  *       step 3: shrink tree to k / (k + lr) from k / (k + lr) - 1
  *               -> normalize for train data
  *       end with tree weight = k / (k + lr)
  */
  void Normalize() {
    double k = static_cast<double>(drop_index_.size());
    for (auto i : drop_index_) {
      for (int curr_class = 0; curr_class < num_class_; ++curr_class) {
        auto curr_tree = i * num_class_ + curr_class;
        // update validation score
        models_[curr_tree]->Shrinkage(shrinkage_rate_);
        for (auto& score_updater : valid_score_updater_) {
          score_updater->AddScore(models_[curr_tree].get(), curr_class);
        }
        // update training score
        models_[curr_tree]->Shrinkage(-k / gbdt_config_->learning_rate);
        train_score_updater_->AddScore(models_[curr_tree].get(), curr_class);
      }
    }
  }
  /*! \brief The indexes of dropping trees */
  std::vector<int> drop_index_;
  /*! \brief Random generator, used to select dropping trees */
  Random random_for_drop_;
  /*! \brief Flag that the score is update on current iter or not*/
  bool is_update_score_cur_iter_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_DART_H_
