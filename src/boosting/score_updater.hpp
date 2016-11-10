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
  explicit ScoreUpdater(const Dataset* data, int num_class)
    :data_(data) {
    num_data_ = data->num_data();
    score_ = new score_t[num_data_ * num_class];
    // default start score is zero
    std::memset(score_, 0, sizeof(score_t) * num_data_ * num_class);
    const float* init_score = data->metadata().init_score();
    // if exists initial score, will start from it
    if (init_score != nullptr) {
      for (data_size_t i = 0; i < num_data_ * num_class; ++i) {
        score_[i] = init_score[i];
      }
    }
  }
  /*! \brief Destructor */
  ~ScoreUpdater() {
    delete[] score_;
  }
  /*!
  * \brief Using tree model to get prediction number, then adding to scores for all data
  *        Note: this function generally will be used on validation data too.
  * \param tree Trained tree model
  * \param curr_class Current class for multiclass training
  */
  inline void AddScore(const Tree* tree, int curr_class) {
    tree->AddPredictionToScore(data_, num_data_, score_ + curr_class * num_data_);
  }
  /*!
  * \brief Adding prediction score, only used for training data.
  *        The training data is partitioned into tree leaves after training
  *        Based on which We can get prediction quckily.
  * \param tree_learner
  * \param curr_class Current class for multiclass training
  */
  inline void AddScore(const TreeLearner* tree_learner, int curr_class) {
    tree_learner->AddPredictionToScore(score_ + curr_class * num_data_);
  }
  /*!
  * \brief Using tree model to get prediction number, then adding to scores for parts of data
  *        Used for prediction of training out-of-bag data
  * \param tree Trained tree model
  * \param data_indices Indices of data that will be proccessed
  * \param data_cnt Number of data that will be proccessed
  * \param curr_class Current class for multiclass training
  */
  inline void AddScore(const Tree* tree, const data_size_t* data_indices,
                                                  data_size_t data_cnt, int curr_class) {
    tree->AddPredictionToScore(data_, data_indices, data_cnt, score_ + curr_class * num_data_);
  }
  /*! \brief Pointer of score */
  inline const score_t* score() { return score_; }
  inline const data_size_t num_data() { return num_data_; }
private:
  /*! \brief Number of total data */
  data_size_t num_data_;
  /*! \brief Pointer of data set */
  const Dataset* data_;
  /*! \brief Scores for data set */
  score_t* score_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_SCORE_UPDATER_HPP_
