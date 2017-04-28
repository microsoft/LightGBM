#ifndef LIGHTGBM_APPLICATION_H_
#define LIGHTGBM_APPLICATION_H_

#include <LightGBM/meta.h>
#include <LightGBM/config.h>

#include <vector>
#include <memory>

namespace LightGBM {

class DatasetLoader;
class Dataset;
class Boosting;
class ObjectiveFunction;
class Metric;

/*!
* \brief The main entrance of LightGBM. this application has two tasks:
*        Train and Predict.
*        Train task will train a new model
*        Predict task will predict the scores of test data using exsisting model,
*        and save the score to disk.
*/
class Application {
public:
  Application(int argc, char** argv);

  /*! \brief Destructor */
  ~Application();

  /*! \brief To call this funciton to run application*/
  inline void Run();

private:
  /*! 
  * \brief Global Sync by minimal, will return minimal T across nodes
  * \param local Local data
  * \return minimal values across nodes 
  */
  template<typename T>
  T GlobalSyncUpByMin(T& local);

  /*! \brief Load parameters from command line and config file*/
  void LoadParameters(int argc, char** argv);

  /*! \brief Load data, including training data and validation data*/
  void LoadData();

  /*! \brief Initialization before training*/
  void InitTrain();

  /*! \brief Main Training logic */
  void Train();

  /*! \brief Initializations before prediction */
  void InitPredict();

  /*! \brief Main predicting logic */
  void Predict();

  /*! \brief Main Convert model logic */
  void ConvertModel();

  /*! \brief All configs */
  OverallConfig config_;
  /*! \brief Training data */
  std::unique_ptr<Dataset> train_data_;
  /*! \brief Validation data */
  std::vector<std::unique_ptr<Dataset>> valid_datas_;
  /*! \brief Metric for training data */
  std::vector<std::unique_ptr<Metric>> train_metric_;
  /*! \brief Metrics for validation data */
  std::vector<std::vector<std::unique_ptr<Metric>>> valid_metrics_;
  /*! \brief Boosting object */
  std::unique_ptr<Boosting> boosting_;
  /*! \brief Training objective function */
  std::unique_ptr<ObjectiveFunction> objective_fun_;
};


inline void Application::Run() {
  if (config_.task_type == TaskType::kPredict) {
    InitPredict();
    Predict();
  } else if (config_.task_type == TaskType::kConvertModel) {
    ConvertModel();
  } else {
    InitTrain();
    Train();
  }
}

}  // namespace LightGBM

#endif   // LightGBM_APPLICATION_H_
