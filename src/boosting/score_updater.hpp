#ifndef LIGHTGBM_BOOSTING_SCORE_UPDATER_HPP_
#define LIGHTGBM_BOOSTING_SCORE_UPDATER_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>
#include <LightGBM/tree_learner.h>

#include <cstring>

namespace LightGBM {
/*!
* \brief Used to store and update score for data
*/
class ScoreUpdater {
public:
  /*!
  * \brief Constructor, will pass a const pointer of dataset
  * \param data This class will bind with this data set
  */
  explicit ScoreUpdater(const Dataset* data)
    :data_(data) {
    num_data_ = data->num_data();
    score_ = new score_t[num_data_];
    // default start score is zero
    std::memset(score_, 0, sizeof(score_t)*num_data_);
    const score_t* init_score = data->metadata().init_score();
    // if exists initial score, will start from it
    if (init_score != nullptr) {
      for (data_size_t i = 0; i < num_data_; ++i) {
        score_[i] = init_score[i];
      }
    }
  }
  /*! \brief Destructor */
  ~ScoreUpdater() {
    delete[] score_;
  }
  /*!
  * \brief Use tree model to get prediction, then add to score for all data
  * Note: this function generally will be used for validation data.
  * \param tree Trained tree model
  */
  inline void AddScore(const Tree* tree) {
    tree->AddPredictionToScore(data_, num_data_, score_);
  }
  /*!
  * \brief Add prediction score, only used for training data.
  * After trained a tree, the training data is partitioned into tree leaves. 
  * We can get prediction by faster speed based on this.
  * \param tree_learner
  */
  inline void AddScore(const TreeLearner* tree_learner) {
    tree_learner->AddPredictionToScore(score_);
  }
  /*!
  * \brief Like AddScore(const Tree* tree), but only for part of data
  * Used for prediction of training out-of-bad data
  * \param tree Trained tree model
  * \param data_indices Indices of data that want proccess to
  * \param data_cnt Number of data that want proccess to
  */
  inline void AddScore(const Tree* tree, const data_size_t* data_indices,
                                                  data_size_t data_cnt) {
    tree->AddPredictionToScore(data_, data_indices, data_cnt, score_);
  }
  /*! \brief Pointer of score */
  inline const score_t * score() { return score_; }

private:
  /*! \brief Number of total data */
  data_size_t num_data_;
  /*! \brief Pointer of data set */
  const Dataset* data_;
  /*! \brief scores for data set */
  score_t* score_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_SCORE_UPDATER_HPP_
