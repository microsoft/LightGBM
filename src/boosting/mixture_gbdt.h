/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Mixture-of-Experts GBDT extension for regime-switching models.
 */
#ifndef LIGHTGBM_SRC_BOOSTING_MIXTURE_GBDT_H_
#define LIGHTGBM_SRC_BOOSTING_MIXTURE_GBDT_H_

#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>

#include <memory>
#include <string>
#include <vector>

#include "gbdt.h"

namespace LightGBM {

/*!
 * \brief Mixture-of-Experts GBDT implementation.
 *
 * This class implements a mixture-of-experts model where:
 * - K expert GBDTs specialize in different data regimes
 * - 1 gate GBDT determines which expert(s) to use for each sample
 * - Final prediction: yhat = sum_k(gate_k * expert_k)
 *
 * Training uses EM-style updates:
 * - E-step: Update responsibilities based on expert fit and gate probability
 * - M-step: Update experts with responsibility-weighted gradients
 * - M-step: Update gate with argmax responsibilities as pseudo-labels
 */
class MixtureGBDT : public GBDTBase {
 public:
  MixtureGBDT();
  ~MixtureGBDT();

  void Init(const Config* config, const Dataset* train_data,
            const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override;

  void MergeFrom(const Boosting* other) override;
  void ShuffleModels(int start_iter, int end_iter) override;
  void ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
                         const std::vector<const Metric*>& training_metrics) override;
  void ResetConfig(const Config* config) override;
  void AddValidDataset(const Dataset* valid_data,
                       const std::vector<const Metric*>& valid_metrics) override;
  void Train(int snapshot_freq, const std::string& model_output_path) override;
  void RefitTree(const int* tree_leaf_prediction, const size_t nrow, const size_t ncol) override;

  /*!
   * \brief Training logic for one iteration (EM-style update)
   */
  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override;

  void RollbackOneIter() override;
  int GetCurrentIteration() const override;
  std::vector<double> GetEvalAt(int data_idx) const override;
  const double* GetTrainingScore(int64_t* out_len) override;
  int64_t GetNumPredictAt(int data_idx) const override;
  void GetPredictAt(int data_idx, double* result, int64_t* out_len) override;
  int NumPredictOneRow(int start_iteration, int num_iteration, bool is_pred_leaf, bool is_pred_contrib) const override;

  void PredictRaw(const double* features, double* output,
                  const PredictionEarlyStopInstance* earlyStop) const override;
  void PredictRawByMap(const std::unordered_map<int, double>& features, double* output,
                       const PredictionEarlyStopInstance* early_stop) const override;
  void Predict(const double* features, double* output,
               const PredictionEarlyStopInstance* earlyStop) const override;
  void PredictByMap(const std::unordered_map<int, double>& features, double* output,
                    const PredictionEarlyStopInstance* early_stop) const override;
  void PredictLeafIndex(const double* features, double* output) const override;
  void PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const override;
  void PredictContrib(const double* features, double* output) const override;
  void PredictContribByMap(const std::unordered_map<int, double>& features,
                           std::vector<std::unordered_map<int, double>>* output) const override;

  std::string DumpModel(int start_iteration, int num_iteration, int feature_importance_type) const override;
  std::string ModelToIfElse(int num_iteration) const override;
  bool SaveModelToIfElse(int num_iteration, const char* filename) const override;
  bool SaveModelToFile(int start_iteration, int num_iterations, int feature_importance_type, const char* filename) const override;
  std::string SaveModelToString(int start_iteration, int num_iterations, int feature_importance_type) const override;
  bool LoadModelFromString(const char* buffer, size_t len) override;

  std::vector<double> FeatureImportance(int num_iteration, int importance_type) const override;
  double GetUpperBoundValue() const override;
  double GetLowerBoundValue() const override;
  int MaxFeatureIdx() const override;
  std::vector<std::string> FeatureNames() const override;
  int LabelIdx() const override;
  int NumberOfTotalModel() const override;
  int NumModelPerIteration() const override;
  int NumberOfClasses() const override;
  bool NeedAccuratePrediction() const override;
  void InitPredict(int start_iteration, int num_iteration, bool is_pred_contrib) override;

  double GetLeafValue(int tree_idx, int leaf_idx) const override;
  void SetLeafValue(int tree_idx, int leaf_idx, double val) override;

  const char* SubModelName() const override { return "mixture"; }
  std::string GetLoadedParam() const override;
  std::string ParserConfigStr() const override;

  // MoE-specific prediction methods
  /*!
   * \brief Get regime (argmax of gate probabilities) for each sample
   * \param features Feature values
   * \param output Output array for regime indices
   */
  void PredictRegime(const double* features, int* output) const;

  /*!
   * \brief Get gate probabilities (regime probabilities) for each sample
   * \param features Feature values
   * \param output Output array of size K for probabilities
   */
  void PredictRegimeProba(const double* features, double* output) const;

  /*!
   * \brief Get individual expert predictions for each sample
   * \param features Feature values
   * \param output Output array of size K for expert predictions
   */
  void PredictExpertPred(const double* features, double* output) const;

  /*!
   * \brief Get number of experts
   */
  int NumExperts() const { return num_experts_; }

 protected:
  /*!
   * \brief Initialize expert responsibilities (uniform, kmeans, etc.)
   */
  void InitResponsibilities();

  /*!
   * \brief Forward pass: compute expert predictions and gate probabilities
   */
  void Forward();

  /*!
   * \brief E-step: update responsibilities based on expert fit and gate probability
   */
  void EStep();

  /*!
   * \brief Apply time-series smoothing to responsibilities (EMA)
   */
  void SmoothResponsibilities();

  /*!
   * \brief M-step for experts: update with responsibility-weighted gradients
   */
  void MStepExperts();

  /*!
   * \brief M-step for gate: update with argmax responsibilities as pseudo-labels
   */
  void MStepGate();

  /*!
   * \brief Compute pointwise loss for E-step
   */
  double ComputePointwiseLoss(double y, double pred) const;

  /*!
   * \brief Numerically stable softmax
   */
  void Softmax(const double* scores, int n, double* probs) const;

  /*! \brief Number of experts (K) */
  int num_experts_;

  /*! \brief Expert GBDTs */
  std::vector<std::unique_ptr<GBDT>> experts_;

  /*! \brief Gate GBDT (multiclass with K classes) */
  std::unique_ptr<GBDT> gate_;

  /*! \brief Config for experts */
  std::unique_ptr<Config> expert_config_;

  /*! \brief Config for gate */
  std::unique_ptr<Config> gate_config_;

  /*! \brief Original config */
  std::unique_ptr<Config> config_;

  /*! \brief Responsibilities r_ik (N x K) */
  std::vector<double> responsibilities_;

  /*! \brief Expert predictions (N x K) */
  std::vector<double> expert_pred_;

  /*! \brief Gate probabilities (N x K) */
  std::vector<double> gate_proba_;

  /*! \brief Combined prediction yhat (N) */
  std::vector<double> yhat_;

  /*! \brief Gradients for mixture (N) */
  std::vector<score_t> gradients_;

  /*! \brief Hessians for mixture (N) */
  std::vector<score_t> hessians_;

  /*! \brief Training data */
  const Dataset* train_data_;

  /*! \brief Objective function */
  const ObjectiveFunction* objective_function_;

  /*! \brief Training metrics */
  std::vector<const Metric*> training_metrics_;

  /*! \brief Number of data points */
  data_size_t num_data_;

  /*! \brief Current iteration */
  int iter_;

  /*! \brief Max feature index */
  int max_feature_idx_;

  /*! \brief Feature names */
  std::vector<std::string> feature_names_;

  /*! \brief Label index */
  int label_idx_;

  /*! \brief E-step loss type (l2, l1, quantile) */
  std::string e_step_loss_type_;

  /*! \brief Loaded parameter string for serialization */
  std::string loaded_parameter_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_BOOSTING_MIXTURE_GBDT_H_
