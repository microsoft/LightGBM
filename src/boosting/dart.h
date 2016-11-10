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
  DART();
  /*!
  * \brief Destructor
  */
  ~DART();
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
  * \brief one training iteration
  */
  bool TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) override;
  /*!
  * \brief Get current training score
  * \param out_len lenght of returned score
  * \return training score
  */
  const score_t* GetTrainingScore(data_size_t* out_len) const override;
  /*!
  * \brief Serialize models by string
  * \return String output of tranined model
  */
  void SaveModelToFile(int num_used_model, bool is_finish, const char* filename) override;
  /*!
  * \brief Get Type name of this boosting object
  */
  const char* Name() const override { return "dart"; }

private:
  /*!
  * \brief select drop trees based on drop_rate
  */
  void SelectDroppingTrees();
  /*!
  * \brief normalize dropped trees
  */
  void Normalize();
  /*! \brief The indexes of dropping trees */
  std::vector<size_t> drop_index_;
  /*! \brief Dropping rate */
  double drop_rate_;
  /*! \brief Shrinkage rate for one iteration */
  double shrinkage_rate_;
  /*! \brief Random generator, used to select dropping trees */
  Random random_for_drop_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_DART_H_
