//
// Created by archer on 11.04.2021.
//

#include "mvs.hpp"

namespace LightGBM {

static score_t CalculateThreshold(std::vector<score_t> grad_values_copy, double sample_size, data_size_t* big_grad_cnt) {
  std::vector<score_t> *grad_values = &grad_values_copy;
  double sum_low = 0.;
  size_t n_high = 0;
  int begin = 0;
  int end = static_cast<int>(grad_values->size());

  while (begin != end) {
    // TODO do partition in parallel
    // TODO partition to three parts
    int middle_begin, middle_end;
    ArrayArgs<score_t>::Partition(grad_values, begin, end, &middle_begin, &middle_end);

    const size_t n_middle = middle_end - middle_begin;
    const size_t n_right = end - middle_end;
    const score_t pivot = (*grad_values)[middle_begin];

    // TODO do sum in parallel
    double cur_left_sum = std::accumulate(&grad_values->at(begin), &grad_values->at(middle_begin), 0.0);
    double sum_middle = n_middle * pivot;

    double cur_sampling_rate = (sum_low + cur_left_sum) / pivot + n_right + n_middle + n_high;

    if (cur_sampling_rate > sample_size) {
      sum_low += sum_middle + cur_left_sum;
      begin = middle_end;
    } else {
      n_high += n_right + n_middle;
      end = middle_begin;
    }
  }
  *big_grad_cnt = n_high;
  return sum_low / (sample_size - n_high + MVS::kMVSEps);
}

static double ComputeLeavesMeanSquaredValue(const Tree &tree) {
  // TODO sum over leaves are leave values one dimensional
  // TODO sum using openmp
  double sum_values = 0.0;
  for (int i = 0; i < tree.num_leaves(); ++i) {
    const auto output = tree.LeafOutput(i);
    sum_values += output * output;
  }
  return std::sqrt(sum_values / tree.num_leaves());
}

void MVS::ResetMVS() {
  CHECK(config_->bagging_fraction > 0.0f && config_->bagging_fraction < 1.0f && config_->bagging_freq > 0);
  CHECK(config_->mvs_lambda > 0.0f && config_->mvs_lambda < 1.0f);
  CHECK(!balanced_bagging_);
  const auto sample_size = static_cast<size_t>(config_->bagging_fraction * num_data_);
  CHECK_EQ(sample_size, bag_data_indices_.size());
  Log::Info("Using MVS");

}

double MVS::GetLambda() {
  double lambda = ComputeLeavesMeanSquaredValue(*models_.back());
  return lambda;
}

void MVS::Bagging(int iter) {
  bag_data_cnt_ = num_data_;
  if (mvs_adaptive_) {
    mvs_lambda_ = GetLambda();
  }

  auto left_cnt = bagging_runner_.Run<true>(
      num_data_,
      [=](int, data_size_t cur_start, data_size_t cur_cnt, data_size_t *left,
          data_size_t *) {
        data_size_t cur_left_cout = BaggingHelper(cur_start, cur_cnt, left);
        return cur_left_cout;
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
}

data_size_t MVS::BaggingHelper(data_size_t start, data_size_t cnt, data_size_t *buffer) {
  if (cnt <= 0) {
    return 0;
  }

  std::vector<score_t> tmp_derivatives(cnt, 0.0f);
  for (data_size_t i = 0; i < cnt; ++i) {
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + start * i;
      tmp_derivatives[i] += gradients_[idx] * gradients_[idx] + mvs_lambda_ * hessians_[idx] * hessians_[idx];
    }
    tmp_derivatives[i] = std::sqrt(tmp_derivatives[i]);
  }

  auto sample_rate = static_cast<data_size_t>(cnt * config_->bagging_fraction);
  data_size_t big_grad_cnt = 0;
  const auto threshold = CalculateThreshold(tmp_derivatives, static_cast<double>(sample_rate), &big_grad_cnt);
  data_size_t left_cnt = 0;
  data_size_t big_weight_cnt = 0;
  for (data_size_t i = 0; i < cnt; ++i) {
    auto position = start + i;
    if (tmp_derivatives[i] > threshold) {
      buffer[left_cnt++] = position;
      ++big_weight_cnt;
    } else {
      double proba_threshold = tmp_derivatives[i] / threshold;
      data_size_t sampled = left_cnt - big_weight_cnt;
      data_size_t  rest_needed = ;
    }
  }
}

}  // namspace LightGBM