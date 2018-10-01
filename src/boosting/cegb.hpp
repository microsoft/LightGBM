#ifndef LIGHTGBM_BOOSTING_CEGB_H_
#define LIGHTGBM_BOOSTING_CEGB_H_

#include <LightGBM/boosting.h>
#include "score_updater.hpp"
#include "gbdt.h"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>

namespace LightGBM {
/*!
* \brief CEGB algorithm implementation. including Training, prediction.
*/
class CEGB: public GBDT {
public:
  /*!
  * \brief Constructor
  */
  CEGB() : GBDT() { }
  /*!
  * \brief Destructor
  */
  ~CEGB() { }
  /*!
  * \brief Initialization logic
  * \param config Config for boosting
  * \param train_data Training data
  * \param objective_function Training objective function
  * \param training_metrics Training metrics
  * \param output_model_filename Filename of output model
  */
  void Init(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override {
    GBDT::Init(config, train_data, objective_function, training_metrics);
  }

  void ResetTrainingData(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
                         const std::vector<const Metric*>& training_metrics) override {
    GBDT::ResetTrainingData(config, train_data, objective_function, training_metrics);
  }
  /*!
  * \brief one training iteration
  */
  bool TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) override {
    bool gbdt_res = GBDT::TrainOneIter(gradient, hessian, is_eval);
    return gbdt_res;
  }

private:
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_CEGB_H_
