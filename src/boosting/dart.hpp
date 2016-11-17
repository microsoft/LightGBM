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
    drop_rate_ = gbdt_config_->drop_rate;
    shrinkage_rate_ = 1.0;
    random_for_drop_ = Random(gbdt_config_->drop_seed);
  }
  /*!
  * \brief one training iteration
  */
  bool TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) override {
    GBDT::TrainOneIter(gradient, hessian, false);
    // normalize
    Normalize();
    if (is_eval) {
      return EvalAndCheckEarlyStopping();
    } else {
      return false;
    }
  }
  /*!
  * \brief Get current training score
  * \param out_len length of returned score
  * \return training score
  */
  const score_t* GetTrainingScore(data_size_t* out_len) override {
    DroppingTrees();
    *out_len = train_score_updater_->num_data() * num_class_;
    return train_score_updater_->score();
  }
  /*!
  * \brief save model to file
  * \param num_used_model number of model that want to save, -1 means save all
  * \param is_finish is training finished or not
  * \param filename filename that want to save to
  */
  void SaveModelToFile(int num_used_model, bool is_finish, const char* filename) override {
    // only save model once when is_finish = true
    if (is_finish && saved_model_size_ < 0) {
      GBDT::SaveModelToFile(num_used_model, is_finish, filename);
    }
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
    if (drop_rate_ > kEpsilon) {
      for (size_t i = 0; i < static_cast<size_t>(iter_); ++i) {
        if (random_for_drop_.NextDouble() < drop_rate_) {
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
    shrinkage_rate_ = 1.0 / (1.0 + drop_index_.size());
  }
  /*!
  * \brief normalize dropped trees
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
        models_[curr_tree]->Shrinkage(-k);
        train_score_updater_->AddScore(models_[curr_tree].get(), curr_class);
      }
    }
  }
  /*! \brief The indexes of dropping trees */
  std::vector<size_t> drop_index_;
  /*! \brief Dropping rate */
  double drop_rate_;
  /*! \brief Random generator, used to select dropping trees */
  Random random_for_drop_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_DART_H_
