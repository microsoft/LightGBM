#ifndef LIGHTGBM_OBJECTIVE_FUNCTION_H_
#define LIGHTGBM_OBJECTIVE_FUNCTION_H_

#include <LightGBM/meta.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <functional>

namespace LightGBM {
/*!
* \brief The interface of Objective Function.
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
  * \brief calculating first order derivative of loss function
  * \param score prediction score in this round
  * \gradients Output gradients
  * \hessians Output hessians
  */
  virtual void GetGradients(const double* score,
    score_t* gradients, score_t* hessians) const = 0;

  virtual const char* GetName() const = 0;

  virtual bool IsConstantHessian() const { return false; }

  virtual bool IsRenewTreeOutput() const { return false; }

  virtual double RenewTreeOutput(double ori_output, const double*,
                                 const data_size_t*,
                                 const data_size_t*,
                                 data_size_t) const { return ori_output; }

  virtual double RenewTreeOutput(double ori_output, double,
                                  const data_size_t*,
                                  const data_size_t*,
                                  data_size_t) const {
                                  return ori_output;
  }

  virtual double BoostFromScore(int /*class_id*/) const { return 0.0; }

  virtual bool ClassNeedTrain(int /*class_id*/) const { return true; }

  virtual bool SkipEmptyClass() const { return false; }

  virtual int NumModelPerIteration() const { return 1; }

  virtual int NumPredictOneRow() const { return 1; }

  /*! \brief The prediction should be accurate or not. True will disable early stopping for prediction. */
  virtual bool NeedAccuratePrediction() const { return true; }

  virtual void ConvertOutput(const double* input, double* output) const {
    output[0] = input[0];
  }

  virtual std::string ToString() const = 0;

  ObjectiveFunction() = default;
  /*! \brief Disable copy */
  ObjectiveFunction& operator=(const ObjectiveFunction&) = delete;
  /*! \brief Disable copy */
  ObjectiveFunction(const ObjectiveFunction&) = delete;

  /*!
  * \brief Create object of objective function
  * \param type Specific type of objective function
  * \param config Config for objective function
  */
  LIGHTGBM_EXPORT static ObjectiveFunction* CreateObjectiveFunction(const std::string& type,
    const Config& config);

  /*!
  * \brief Load objective function from string object
  */
  LIGHTGBM_EXPORT static ObjectiveFunction* CreateObjectiveFunction(const std::string& str);
};

}  // namespace LightGBM

#endif   // LightGBM_OBJECTIVE_FUNCTION_H_
