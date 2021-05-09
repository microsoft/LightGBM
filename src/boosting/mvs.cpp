//
// Created by archer on 11.04.2021.
//

#include "mvs.hpp"

#include <algorithm>

namespace LightGBM {

using ConstTreeIterator = std::vector<std::unique_ptr<Tree>>::const_iterator;

static double CalculateThresholdSequential(std::vector<score_t>* gradients, data_size_t begin, data_size_t end,
                                    const double sample_size) {
  double current_sum_small = 0.0;
  data_size_t big_grad_size = 0;

  while (begin != end) {
    data_size_t middle_begin=0, middle_end=0;
    ArrayArgs<score_t>::Partition(gradients, begin, end, &middle_begin, &middle_end);
    ++middle_begin; // for half intervals
    const data_size_t n_middle = middle_end - middle_begin;
    const data_size_t large_size = middle_begin - begin;

    const double sum_small = std::accumulate(gradients->begin() + middle_end, gradients->begin() + end, 0.0);
    const double sum_middle = (*gradients)[middle_begin] * n_middle;

    const double
        current_sampling_rate = (current_sum_small + sum_small) / (*gradients)[middle_begin] + big_grad_size + n_middle + large_size;

    if (current_sampling_rate > sample_size) {
      current_sum_small += sum_small + sum_middle;
      end = middle_begin;
    } else {
      big_grad_size += n_middle + large_size;
      begin = middle_end;
    }
  }

  return current_sum_small / (sample_size - big_grad_size + kEpsilon);
}

static double ComputeLeavesMeanSquaredValue(ConstTreeIterator begin, ConstTreeIterator end) {
  double sum_values = 0.0;
  data_size_t num_leaves = (*begin)->num_leaves();
#pragma omp parallel for schedule(static, 2048) reduction(+:sum_values)
  for (data_size_t leaf_idx = 0; leaf_idx < num_leaves; ++leaf_idx) {
    double leave_value = 0.0;
    for (ConstTreeIterator it = begin; it != end; ++it) {
      if (leaf_idx < (**it).num_leaves()) {
        const double value = (*it)->LeafOutput(leaf_idx);
        leave_value += value * value;
      }
    }
    sum_values += std::sqrt(leave_value);
  }
  return sum_values / num_leaves;
}

static double ComputeMeanGradValues(score_t *gradients,
                                    score_t *hessians,
                                    data_size_t size,
                                    data_size_t num_tree_per_iteration) {
  double sum = 0.0;
#pragma omp parallel for schedule(static, 1024) reduction(+:sum)
  for (data_size_t i = 0; i < size; ++i) {
    double local_hessians = 0.0, local_gradients = 0.0;
    for (data_size_t j = 0; j < num_tree_per_iteration; ++j) {
      size_t idx = static_cast<size_t>(size) * j + i;
      local_hessians += hessians[idx] * hessians[idx];
      local_gradients += gradients[idx] * gradients[idx];
    }
    sum += std::sqrt(local_gradients / local_hessians);
  }
  return sum / size;
}

double MVS::GetLambda() {
  if (!mvs_adaptive_) {
    return mvs_lambda_;
  }
  double lambda =
      (this->iter_ > 0) ? ComputeLeavesMeanSquaredValue(models_.cend() - num_tree_per_iteration_, models_.cend())
          / config_->learning_rate
                        : ComputeMeanGradValues(gradients_.data(),
                                                hessians_.data(),
                                                num_data_,
                                                num_tree_per_iteration_);

  return lambda;
}

void MVS::Bagging(int iter) {
  if (iter % config_->bagging_freq != 0 && !need_re_bagging_) {
    return;
  }

  bag_data_cnt_ = num_data_;
  mvs_lambda_ = GetLambda();

  if (num_data_ <= kMaxSequentialSize) {
    threshold_ = GetThreshold(0, num_data_);
  }

  auto left_cnt = bagging_runner_.Run<true>(
      num_data_,
      [=](int, data_size_t cur_start, data_size_t cur_cnt, data_size_t *left,
          data_size_t *) {
        data_size_t left_count = BaggingHelper(cur_start, cur_cnt, left);
        return left_count;
      },
      bag_data_indices_.data());

  bag_data_cnt_ = left_cnt;
  if (!is_use_subset_) {
    tree_learner_->SetBaggingData(nullptr, bag_data_indices_.data(), bag_data_cnt_);
  } else {
    tmp_subset_->ReSize(bag_data_cnt_);
    tmp_subset_->CopySubrow(train_data_, bag_data_indices_.data(),
                            bag_data_cnt_, false);
    tree_learner_->SetBaggingData(tmp_subset_.get(), bag_data_indices_.data(),
                                  bag_data_cnt_);
  }
  threshold_ = 0.0;
  Log::Debug("MVS Sample size %d %d", left_cnt, static_cast<data_size_t>(config_->bagging_fraction * num_data_));
}

data_size_t MVS::BaggingHelper(data_size_t start, data_size_t cnt, data_size_t *buffer) {
  if (cnt <= 0) {
    return 0;
  }

  const double threshold = GetThreshold(start, cnt);

  data_size_t left_cnt = 0;
  data_size_t right_pos = cnt;
  data_size_t big_weight_cnt = 0;
  for (data_size_t i = 0; i < cnt; ++i) {
    data_size_t position = start + i;

    double derivative = 0.0;
    for (data_size_t j = 0; j < num_tree_per_iteration_; ++j) {
      size_t idx = static_cast<size_t>(j) * num_data_ + position;
      derivative += gradients_[idx] * gradients_[idx] + mvs_lambda_ * hessians_[idx] * hessians_[idx];
    }
    derivative = std::sqrt(derivative);

    if (derivative >= threshold) {
      buffer[left_cnt++] = position;
      ++big_weight_cnt;
    } else {
      const double proba_threshold = derivative / threshold;
      const double proba = bagging_rands_[position / bagging_rand_block_].NextFloat();
      if (proba < proba_threshold) {
        buffer[left_cnt++] = position;
        for (data_size_t tree_id = 0; tree_id < num_tree_per_iteration_; ++tree_id) {
          size_t idx = static_cast<size_t>(num_data_) * tree_id + position;
          gradients_[idx] /= proba_threshold;
          hessians_[idx] /= proba_threshold;
        }
      } else {
        buffer[--right_pos] = position;
      }
    }
  }

  return left_cnt;
}

double MVS::GetThreshold(data_size_t begin, data_size_t cnt) {
  data_size_t n_blocks, block_size;
  Threading::BlockInfoForceSize<data_size_t>(num_data_, bagging_rand_block_, &n_blocks, &block_size);
  if (num_data_ < kMaxSequentialSize && block_size > 1 && threshold_ != 0.0) {
    return threshold_;
  }

  for (data_size_t i = begin; i < begin + cnt; ++i) {
    tmp_derivatives_[i] = 0.0f;
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + i;
      tmp_derivatives_[i] += gradients_[idx] * gradients_[idx] + mvs_lambda_ * hessians_[idx] * hessians_[idx];
    }
    tmp_derivatives_[i] = std::sqrt(tmp_derivatives_[i]);
  }

  double threshold = CalculateThresholdSequential(&tmp_derivatives_, begin, begin + cnt,
                                                  cnt * config_->bagging_fraction);
  return threshold;
}

void MVS::ResetMVS() {
  CHECK(config_->bagging_fraction > 0.0f && config_->bagging_fraction < 1.0f && config_->bagging_freq > 0);
  CHECK(config_->mvs_lambda >= 0.0f);
  CHECK(!balanced_bagging_);

  bag_data_indices_.resize(num_data_);
  tmp_derivatives_.resize(num_data_);
  Log::Info("Using MVS");
}

}  // namspace LightGBM