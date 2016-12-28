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
  ScoreUpdater(const Dataset* data, int num_class) : data_(data) {
    num_data_ = data->num_data();
    size_t total_size = static_cast<size_t>(num_data_) * num_class;
    score_.resize(total_size);
    // default start score is zero
    std::fill(score_.begin(), score_.end(), 0.0f);
    const float* init_score = data->metadata().init_score();
    // if exists initial score, will start from it
    if (init_score != nullptr) {
      if ((data->metadata().num_init_score() % num_data_) != 0 
        || (data->metadata().num_init_score() / num_data_) != num_class) {
        Log::Fatal("number of class for initial score error");
      }
      for (size_t i = 0; i < total_size; ++i) {
        score_[i] = init_score[i];
      }
    }
  }
  /*! \brief Destructor */
  ~ScoreUpdater() {

  }
  /*!
  * \brief Using tree model to get prediction number, then adding to scores for all data
  *        Note: this function generally will be used on validation data too.
  * \param tree Trained tree model
  * \param curr_class Current class for multiclass training
  */
  inline void AddScore(const Tree* tree, int curr_class) {
    tree->AddPredictionToScore(data_, num_data_, score_.data() + curr_class * num_data_);
  }
  /*!
  * \brief Adding prediction score, only used for training data.
  *        The training data is partitioned into tree leaves after training
  *        Based on which We can get prediction quckily.
  * \param tree_learner
  * \param curr_class Current class for multiclass training
  */
  inline void AddScore(const TreeLearner* tree_learner, int curr_class) {
    tree_learner->AddPredictionToScore(score_.data() + curr_class * num_data_);
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
    tree->AddPredictionToScore(data_, data_indices, data_cnt, score_.data() + curr_class * num_data_);
  }
  /*! \brief Pointer of score */
  inline const score_t* score() const { return score_.data(); }
  inline const data_size_t num_data() const { return num_data_; }

  /*! \brief Disable copy */
  ScoreUpdater& operator=(const ScoreUpdater&) = delete;
  /*! \brief Disable copy */
  ScoreUpdater(const ScoreUpdater&) = delete;
private:
  /*! \brief Number of total data */
  data_size_t num_data_;
  /*! \brief Pointer of data set */
  const Dataset* data_;
  /*! \brief Scores for data set */
  std::vector<score_t> score_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_SCORE_UPDATER_HPP_
