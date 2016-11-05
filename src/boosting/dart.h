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
  * \brief Get Type name of this boosting object
  */
  const char* Name() const override { return "dart"; }

private:

  /*!
  * \brief updating score after tree was trained
  * \param tree Trained tree of this iteration
  * \param curr_class Current class for multiclass training
  */
  void UpdateScore(const Tree* tree, const int curr_class) override;
  /*!
  * \brief select dropping tree and normailize
  * \param curr_class Current class for multiclass training
  */
  double SelectDroppingTreesAndNormalize(int curr_class);
  /*! \brief Dropping rate */
  double drop_rate_;
  /*! \brief Random generator, used to select dropping trees */
  Random random_for_drop_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_DART_H_
