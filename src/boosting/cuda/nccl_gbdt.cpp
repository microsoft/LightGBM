/*!
 * Copyright (c) 2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "nccl_gbdt.hpp"
#include <LightGBM/metric.h>

#ifdef USE_CUDA

namespace LightGBM {

template <typename GBDT_T>
NCCLGBDT<GBDT_T>::NCCLGBDT(): GBDT_T() {}

template <typename GBDT_T>
NCCLGBDT<GBDT_T>::~NCCLGBDT() {}

template <typename GBDT_T>
void NCCLGBDT<GBDT_T>::Init(
  const Config* gbdt_config, const Dataset* train_data,
  const ObjectiveFunction* objective_function,
  const std::vector<const Metric*>& training_metrics) {
  GBDT_T::Init(gbdt_config, train_data, objective_function, training_metrics);
  int max_num_gpu = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDeviceCount(&max_num_gpu));
  num_gpu_ = this->config_->num_gpu;
  if (num_gpu_ > max_num_gpu) {
    Log::Warning("Specifying %d GPUs, but only %d available.", num_gpu_, max_num_gpu);
    num_gpu_ = max_num_gpu;
  }
  int gpu_device_id = this->config_->gpu_device_id;
  if (this->config_->gpu_device_list == std::string("")) {
    if (gpu_device_id < 0 || gpu_device_id >= num_gpu_) {
      Log::Warning("Master GPU Device ID %d is not in the valid range [%d, %d], will use GPU 0 as master.", gpu_device_id, 0, max_num_gpu);
      gpu_device_id = 0;
    }
  }
  master_gpu_device_id_ = gpu_device_id;
  master_gpu_index_ = master_gpu_device_id_;

  if (this->config_->gpu_device_list != std::string("")) {
    std::vector<std::string> gpu_list_str = Common::Split(this->config_->gpu_device_list.c_str(), ",");
    for (const auto& gpu_str : gpu_list_str) {
      int gpu_id = 0;
      Common::Atoi<int>(gpu_str.c_str(), &gpu_id);
      gpu_list_.emplace_back(gpu_id);
    }
    bool check_master_gpu = false;
    for (int i = 0; i < static_cast<int>(gpu_list_.size()); ++i) {
      const int gpu_id = gpu_list_[i];
      if (gpu_id == master_gpu_device_id_) {
        master_gpu_index_ = i;
        check_master_gpu = true;
      }
    }
    if (!check_master_gpu) {
      Log::Fatal("Master GPU ID %d is not in GPU ID list.", master_gpu_device_id_);
    }
  }

  const int num_threads = OMP_NUM_THREADS();
  if (num_gpu_ > num_threads) {
    Log::Fatal("Number of GPUs %d is greather than the number of threads %d. Please use more threads.", num_gpu_, num_threads);
  }

  InitNCCL();

  // partition data across GPUs
  const data_size_t num_data_per_gpu = (this->num_data_ + num_gpu_ - 1) / num_gpu_;
  std::vector<data_size_t> all_data_indices(this->num_data_, 0);
  #pragma omp parallel for schedule(static)
  for (data_size_t i = 0; i < this->num_data_; ++i) {
    all_data_indices[i] = i;
  }
  per_gpu_data_start_.resize(num_gpu_);
  per_gpu_data_end_.resize(num_gpu_);
  per_gpu_datasets_.resize(num_gpu_);
  for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
    SetCUDADevice(gpu_index);
    const data_size_t data_start = num_data_per_gpu * gpu_index;
    const data_size_t data_end = std::min(data_start + num_data_per_gpu, this->num_data_);
    const data_size_t num_data_in_gpu = data_end - data_start;
    per_gpu_data_start_[gpu_index] = data_start;
    per_gpu_data_end_[gpu_index] = data_end;
    per_gpu_datasets_[gpu_index].reset(new Dataset(num_data_in_gpu));
    per_gpu_datasets_[gpu_index]->ReSize(num_data_in_gpu);
    per_gpu_datasets_[gpu_index]->CopyFeatureMapperFrom(this->train_data_);
    per_gpu_datasets_[gpu_index]->CopySubrow(this->train_data_, all_data_indices.data() + data_start, num_data_in_gpu, true, data_start, data_end, GetCUDADevice(gpu_index));
  }

  // initialize per gpu objectives, training scores and tree learners
  per_gpu_objective_functions_.resize(num_gpu_);
  per_gpu_train_score_updater_.resize(num_gpu_);
  per_gpu_gradients_.resize(num_gpu_);
  per_gpu_hessians_.resize(num_gpu_);
  per_gpu_tree_learners_.resize(num_gpu_);
  for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
    SetCUDADevice(gpu_index);
    const data_size_t num_data_in_gpu = per_gpu_data_end_[gpu_index] - per_gpu_data_start_[gpu_index];
    per_gpu_objective_functions_[gpu_index].reset(ObjectiveFunction::CreateObjectiveFunction(this->config_->objective, *(this->config_.get())));
    per_gpu_objective_functions_[gpu_index]->Init(per_gpu_datasets_[gpu_index]->metadata(), per_gpu_datasets_[gpu_index]->num_data());
    per_gpu_objective_functions_[gpu_index]->SetNCCLComm(&nccl_communicators_[gpu_index]);
    per_gpu_train_score_updater_[gpu_index].reset(new CUDAScoreUpdater(per_gpu_datasets_[gpu_index].get(), this->num_tree_per_iteration_));
    per_gpu_gradients_[gpu_index].reset(new CUDAVector<score_t>(num_data_in_gpu));
    per_gpu_hessians_[gpu_index].reset(new CUDAVector<score_t>(num_data_in_gpu));
    per_gpu_tree_learners_[gpu_index].reset(TreeLearner::CreateTreeLearner(
      this->config_->tree_learner,
      this->config_->device_type,
      this->config_.get()));
    per_gpu_tree_learners_[gpu_index]->SetNCCL(&nccl_communicators_[gpu_index], nccl_gpu_rank_[gpu_index], GetCUDADevice(gpu_index), this->num_data_);
    per_gpu_tree_learners_[gpu_index]->Init(per_gpu_datasets_[gpu_index].get(), this->is_constant_hessian_);
  }

  // initialize host threads and thread data
  host_threads_.resize(num_gpu_);
  boosting_thread_data_.resize(num_gpu_);
  train_tree_learner_thread_data_.resize(num_gpu_);
  update_score_thread_data_.resize(num_gpu_);
  for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
    boosting_thread_data_[gpu_index].gpu_index = GetCUDADevice(gpu_index);
    boosting_thread_data_[gpu_index].gpu_objective_function = per_gpu_objective_functions_[gpu_index].get();
    boosting_thread_data_[gpu_index].gradients = per_gpu_gradients_[gpu_index]->RawData();
    boosting_thread_data_[gpu_index].hessians = per_gpu_hessians_[gpu_index]->RawData();
    boosting_thread_data_[gpu_index].score = per_gpu_train_score_updater_[gpu_index]->score();
    train_tree_learner_thread_data_[gpu_index].gpu_index = GetCUDADevice(gpu_index);
    train_tree_learner_thread_data_[gpu_index].gpu_tree_learner = per_gpu_tree_learners_[gpu_index].get();
    train_tree_learner_thread_data_[gpu_index].gradients = per_gpu_gradients_[gpu_index]->RawData();
    train_tree_learner_thread_data_[gpu_index].hessians = per_gpu_hessians_[gpu_index]->RawData();
    train_tree_learner_thread_data_[gpu_index].num_data_in_gpu = per_gpu_data_end_[gpu_index] - per_gpu_data_start_[gpu_index];
    update_score_thread_data_[gpu_index].gpu_index = GetCUDADevice(gpu_index);
    update_score_thread_data_[gpu_index].gpu_score_updater = per_gpu_train_score_updater_[gpu_index].get();
    update_score_thread_data_[gpu_index].gpu_tree_learner = per_gpu_tree_learners_[gpu_index].get();
  }

  // return to master gpu device
  CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
}

template <typename GBDT_T>
void NCCLGBDT<GBDT_T>::InitNCCL() {
  nccl_gpu_rank_.resize(num_gpu_, -1);
  nccl_communicators_.resize(num_gpu_);
  ncclUniqueId nccl_unique_id;
  if (Network::num_machines() == 1 || Network::rank() == 0) {
    NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  }
  if (Network::num_machines() > 1) {
    std::vector<ncclUniqueId> output_buffer(Network::num_machines());
    Network::Allgather(
      reinterpret_cast<char*>(&nccl_unique_id),
      sizeof(ncclUniqueId) / sizeof(char),
      reinterpret_cast<char*>(output_buffer.data()));
    if (Network::rank() > 0) {
      nccl_unique_id = output_buffer[0];
    }
  }

  if (Network::num_machines() > 1) {
    std::vector<int> num_gpus_per_machine(Network::num_machines() + 1, 0);
    Network::Allgather(
      reinterpret_cast<char*>(&num_gpu_),
      sizeof(int) / sizeof(char),
      reinterpret_cast<char*>(num_gpus_per_machine.data() + 1));
    for (int rank = 1; rank < Network::num_machines() + 1; ++rank) {
      num_gpus_per_machine[rank] += num_gpus_per_machine[rank - 1];
    }
    CHECK_EQ(num_gpus_per_machine[Network::rank() + 1] - num_gpus_per_machine[Network::rank()], num_gpu_);
    NCCLCHECK(ncclGroupStart());
    for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
      SetCUDADevice(gpu_index);
      nccl_gpu_rank_[gpu_index] = gpu_index + num_gpus_per_machine[Network::rank()];
      NCCLCHECK(ncclCommInitRank(&nccl_communicators_[gpu_index], num_gpus_per_machine.back(), nccl_unique_id, nccl_gpu_rank_[gpu_index]));
    }
    NCCLCHECK(ncclGroupEnd());
  } else {
    NCCLCHECK(ncclGroupStart());
    for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
      SetCUDADevice(gpu_index);
      nccl_gpu_rank_[gpu_index] = gpu_index;
      NCCLCHECK(ncclCommInitRank(&nccl_communicators_[gpu_index], num_gpu_, nccl_unique_id, gpu_index));
    }
    NCCLCHECK(ncclGroupEnd());
  }

  // return to master gpu device
  CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
}

template <typename GBDT_T>
void* NCCLGBDT<GBDT_T>::BoostingThread(void* thread_data) {
  const BoostingThreadData* boosting_thread_data = reinterpret_cast<BoostingThreadData*>(thread_data);
  const int gpu_index = boosting_thread_data->gpu_index;
  const ObjectiveFunction* objective_function = boosting_thread_data->gpu_objective_function;
  score_t* gradients = boosting_thread_data->gradients;
  score_t* hessians = boosting_thread_data->hessians;
  const double* score = boosting_thread_data->score;
  CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
  objective_function->GetGradients(score, gradients, hessians);
  return nullptr;
}

template <typename GBDT_T>
void NCCLGBDT<GBDT_T>::Boosting() {
  Common::FunctionTimer fun_timer("NCCLGBDT::Boosting", global_timer);
  if (this->objective_function_ == nullptr) {
    Log::Fatal("No object function provided");
  }
  for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
    if (pthread_create(&host_threads_[gpu_index], nullptr, BoostingThread,
      reinterpret_cast<void*>(&boosting_thread_data_[gpu_index]))) {
      Log::Fatal("Error in creating boosting threads.");
    }
  }
  for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
    if (pthread_join(host_threads_[gpu_index], nullptr)) {
      Log::Fatal("Error in joining boosting threads.");
    }
  }
}

template <typename GBDT_T>
double NCCLGBDT<GBDT_T>::BoostFromAverage(int class_id, bool update_scorer) {
  double init_score = GBDT_T::BoostFromAverage(class_id, update_scorer);
  for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
    SetCUDADevice(gpu_index);
    if (std::fabs(init_score) > kEpsilon && update_scorer) {
      per_gpu_train_score_updater_[gpu_index]->AddScore(init_score, class_id);
    }
  }

  // return to master gpu device
  CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
}

template <typename GBDT_T>
void* NCCLGBDT<GBDT_T>::TrainTreeLearnerThread(void* thread_data) {
  TrainTreeLearnerThreadData* tree_train_learner_thread_data = reinterpret_cast<TrainTreeLearnerThreadData*>(thread_data);
  const int gpu_index = tree_train_learner_thread_data->gpu_index;
  const int class_id = tree_train_learner_thread_data->class_id;
  const data_size_t num_data_in_gpu = tree_train_learner_thread_data->num_data_in_gpu;
  const score_t* gradients = tree_train_learner_thread_data->gradients + class_id * num_data_in_gpu;
  const score_t* hessians = tree_train_learner_thread_data->hessians + class_id * num_data_in_gpu;
  const bool is_first_tree = tree_train_learner_thread_data->is_first_time;
  CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
  tree_train_learner_thread_data->tree.reset(
    tree_train_learner_thread_data->gpu_tree_learner->Train(gradients, hessians, is_first_tree));
  return nullptr;
}

template <typename GBDT_T>
bool NCCLGBDT<GBDT_T>::TrainOneIter(const score_t* gradients, const score_t* hessians) {
  Common::FunctionTimer fun_timer("NCCLGBDT::TrainOneIter", global_timer);
  std::vector<double> init_scores(this->num_tree_per_iteration_, 0.0);
  // boosting first
  if (gradients == nullptr || hessians == nullptr) {
    for (int cur_tree_id = 0; cur_tree_id < this->num_tree_per_iteration_; ++cur_tree_id) {
      init_scores[cur_tree_id] = BoostFromAverage(cur_tree_id, true);
    }
    Boosting();
  } else {
    for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
      SetCUDADevice(gpu_index);
      const data_size_t gpu_data_start = per_gpu_data_start_[gpu_index];
      const data_size_t num_data_in_gpu = per_gpu_data_end_[gpu_index] - gpu_data_start;
      for (int class_id = 0; class_id < this->num_class_; ++class_id) {
        CopyFromHostToCUDADevice<score_t>(
          per_gpu_gradients_[gpu_index]->RawData() + class_id * num_data_in_gpu,
          gradients + class_id * this->num_data_ + gpu_data_start, num_data_in_gpu, __FILE__, __LINE__);
        CopyFromHostToCUDADevice<score_t>(
          per_gpu_hessians_[gpu_index]->RawData() + class_id * num_data_in_gpu,
          hessians + class_id * this->num_data_ + gpu_data_start, num_data_in_gpu, __FILE__, __LINE__);
      }
    }

    // return to master gpu device
    CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
  }

  bool should_continue = false;
  for (int cur_tree_id = 0; cur_tree_id < this->num_tree_per_iteration_; ++cur_tree_id) {
    std::vector<std::unique_ptr<Tree>> new_tree(num_gpu_);
    for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
      new_tree[gpu_index].reset(nullptr);
    }
    if (this->class_need_train_[cur_tree_id] && this->train_data_->num_features() > 0) {
      if (this->is_use_subset_ && this->bag_data_cnt_ < this->num_data_) {
        Log::Fatal("Bagging is not supported for NCCLGBDT");
      }
      bool is_first_tree = this->models_.size() < static_cast<size_t>(this->num_tree_per_iteration_);
      for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
        train_tree_learner_thread_data_[gpu_index].is_first_time = is_first_tree;
        train_tree_learner_thread_data_[gpu_index].class_id = cur_tree_id;
        if (pthread_create(&host_threads_[gpu_index], nullptr, TrainTreeLearnerThread,
          reinterpret_cast<void*>(&train_tree_learner_thread_data_[gpu_index]))) {
          Log::Fatal("Error in creating tree training threads.");
        }
      }
      for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
        if (pthread_join(host_threads_[gpu_index], nullptr)) {
          Log::Fatal("Error in joining tree training threads.");
        }
        new_tree[gpu_index].reset(train_tree_learner_thread_data_[gpu_index].tree.release());
      }
    }

    if (new_tree[master_gpu_index_]->num_leaves() > 1) {
      should_continue = true;
      if (this->objective_function_ != nullptr && this->objective_function_->IsRenewTreeOutput()) {
        Log::Fatal("Objective function with renewing is not supported for NCCLGBDT.");
      }
      // shrinkage by learning rate
      for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
        SetCUDADevice(gpu_index);
        new_tree[gpu_index]->Shrinkage(this->shrinkage_rate_);
      }
      CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
      // update score
      UpdateScore(new_tree, cur_tree_id);
      if (std::fabs(init_scores[cur_tree_id]) > kEpsilon) {
        for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
          SetCUDADevice(gpu_index);
          new_tree[gpu_index]->AddBias(init_scores[cur_tree_id]);
        }
        CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
      }
    } else {
      // only add default score one-time
      if (this->models_.size() < static_cast<size_t>(this->num_tree_per_iteration_)) {
        Log::Warning("Training stopped with no splits.");
      }
    }

    // add model
    this->models_.push_back(std::move(new_tree[master_gpu_index_]));

    for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
      if (gpu_index != master_gpu_index_) {
        SetCUDADevice(gpu_index);
        new_tree[gpu_index].reset(nullptr);
      }
    }
    CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
  }

  if (!should_continue) {
    Log::Warning("Stopped training because there are no more leaves that meet the split requirements");
    if (this->models_.size() > static_cast<size_t>(this->num_tree_per_iteration_)) {
      for (int cur_tree_id = 0; cur_tree_id < this->num_tree_per_iteration_; ++cur_tree_id) {
        this->models_.pop_back();
      }
    }
    return true;
  }

  ++this->iter_;
  return false;
}

template <typename GBDT_T>
void* NCCLGBDT<GBDT_T>::UpdateScoreThread(void* thread_data) {
  const UpdateScoreThreadData* update_score_thread_data = reinterpret_cast<const UpdateScoreThreadData*>(thread_data);
  const int gpu_index = update_score_thread_data->gpu_index;
  CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
  update_score_thread_data->gpu_score_updater->AddScore(
    update_score_thread_data->gpu_tree_learner,
    update_score_thread_data->tree,
    update_score_thread_data->cur_tree_id);
  return nullptr;
}

template <typename GBDT_T>
void NCCLGBDT<GBDT_T>::UpdateScore(const std::vector<std::unique_ptr<Tree>>& tree, const int cur_tree_id) {
  Common::FunctionTimer fun_timer("GBDT::UpdateScore", global_timer);
  // update training score
  if (!this->is_use_subset_) {
    if (this->num_data_ - this->bag_data_cnt_ > 0) {
      Log::Fatal("bagging is not supported for NCCLGBDT.");
    }
    for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
      update_score_thread_data_[gpu_index].tree = tree[gpu_index].get();
      update_score_thread_data_[gpu_index].cur_tree_id = cur_tree_id;
      if (pthread_create(&host_threads_[gpu_index], nullptr, UpdateScoreThread,
        reinterpret_cast<void*>(&update_score_thread_data_[gpu_index]))) {
        Log::Fatal("Error in creating update score threads.");
      }
    }
    for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
      if (pthread_join(host_threads_[gpu_index], nullptr)) {
        Log::Fatal("Error in joining tree training threads.");
      }
    }
  } else {
    Log::Fatal("bagging is not supported for NCCLGBDT.");
  }

  // update validation score
  for (auto& score_updater : this->valid_score_updater_) {
    score_updater->AddScore(tree[master_gpu_index_].get(), cur_tree_id);
  }
}

template <typename GBDT_T>
std::vector<double> NCCLGBDT<GBDT_T>::EvalOneMetric(const Metric* metric, const double* score, const data_size_t num_data) const {
  if (score == this->train_score_updater_->score()) {
    // delegate to per gpu train score updater
    std::vector<double> tmp_score(num_data * this->num_class_, 0.0f);
    for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
      const data_size_t data_start = per_gpu_data_start_[gpu_index];
      const data_size_t num_data_in_gpu = per_gpu_data_end_[gpu_index] - data_start;
      for (int class_id = 0; class_id < this->num_class_; ++class_id) {
        SetCUDADevice(gpu_index);
        CopyFromCUDADeviceToHost<double>(tmp_score.data() + class_id * this->num_data_ + data_start,
          per_gpu_train_score_updater_[gpu_index]->score() + class_id * num_data_in_gpu,
          static_cast<size_t>(num_data_in_gpu), __FILE__, __LINE__);
      }
      CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
      return metric->Eval(tmp_score.data(), this->objective_function_);
    }
  } else {
    return GBDT_T::EvalOneMetric(metric, score, num_data);
  }
}

template class NCCLGBDT<GBDT>;

}  // LightGBM

#endif  // USE_CUDA
