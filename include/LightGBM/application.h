#ifndef LIGHTGBM_APPLICATION_H_
#define LIGHTGBM_APPLICATION_H_

#include <LightGBM/meta.h>
#include <LightGBM/config.h>

#include <vector>

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
*        Predict task will predicting the scores of test data using exsiting model,
*        and saving the score to disk.
*/
class Application {
public:
  Application(int argc, char** argv);

  /*! \brief Destructor */
  ~Application();

  /*! \brief To call this funciton  to run application*/
  inline void Run();

private:
  /*! 
  * \brief Global Sync by minimal, will return minimal T across nodes
  * \param local Local data
  * \return minimal values across nodes 
  */
  template<typename T>
  T GlobalSyncUpByMin(T& local);

  /*! \brief Load parametes from command line and config file*/
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

  /*! \brief All configs */
  OverallConfig config_;
  /*! \brief Dataset loader */
  DatasetLoader* dataset_loader_;
  /*! \brief Training data */
  Dataset* train_data_;
  /*! \brief Validation data */
  std::vector<Dataset*> valid_datas_;
  /*! \brief Metric for training data */
  std::vector<Metric*> train_metric_;
  /*! \brief Metrics for validation data */
  std::vector<std::vector<Metric*>> valid_metrics_;
  /*! \brief Boosting object */
  Boosting* boosting_;
  /*! \brief Training objective function */
  ObjectiveFunction* objective_fun_;
};


inline void Application::Run() {
  if (config_.task_type == TaskType::kPredict) {
    InitPredict();
    Predict();
  } else {
    InitTrain();
    Train();
  }
}

}  // namespace LightGBM

#endif   // LightGBM_APPLICATION_H_
