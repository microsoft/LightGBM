#ifndef LIGHTGBM_OBJECTIVE_FUNCTION_H_
#define LIGHTGBM_OBJECTIVE_FUNCTION_H_

#include <LightGBM/meta.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>

namespace LightGBM {

/*!
* \brief The interface of Objective Function.
* Objective function is used to get gradients
*/
class ObjectiveFunction {
public:
  /*! \brief virtual destructor */
  virtual ~ObjectiveFunction() {}

  /*!
  * \brief Initialize
  * \param metadata Label data
  * \param num_data Number of data
  */
  virtual void Init(const Metadata& metadata, data_size_t num_data) = 0;

  /*!
  * \brief calculate first order derivative of loss function
  * \param score Current prediction score
  * \gradients Output gradients
  * \hessians Output hessians
  */
  virtual void GetGradients(const score_t* score,
    score_t* gradients, score_t* hessians) const = 0;

  /*!
  * \brief Get sigmoid param for this objective if has. 
  * This function is used for prediction task, if has sigmoid param, the prediction value will be transform by sigmoid function.
  * \return Sigmoid param, if <=0.0 means don't use sigmoid transform on this objective.
  */
  virtual double GetSigmoid() const = 0;

  /*!
  * \brief Create object of objective function
  * \param type Specific type of objective function
  * \param config Config for objective function
  */
  static ObjectiveFunction* CreateObjectiveFunction(const std::string& type,
    const ObjectiveConfig& config);
};

}  // namespace LightGBM

#endif  #endif  // LightGBM_OBJECTIVE_FUNCTION_H_
