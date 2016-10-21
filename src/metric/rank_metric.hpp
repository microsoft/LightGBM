#ifndef LIGHTGBM_METRIC_RANK_METRIC_HPP_
#define LIGHTGBM_METRIC_RANK_METRIC_HPP_

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <LightGBM/metric.h>

#include <omp.h>

#include <sstream>
#include <vector>

namespace LightGBM {

class NDCGMetric:public Metric {
public:
  explicit NDCGMetric(const MetricConfig& config) {
    early_stopping_round_ = config.early_stopping_round;
    output_freq_ = config.output_freq;
    the_bigger_the_better = true;
    // get eval position
    for (auto k : config.eval_at) {
      eval_at_.push_back(static_cast<data_size_t>(k));
    }
    // initialize DCG calculator
    DCGCalculator::Init(config.label_gain);
    // get number of threads
    #pragma omp parallel
    #pragma omp master
    {
      num_threads_ = omp_get_num_threads();
    }
  }

  ~NDCGMetric() {
  }
  void Init(const char* test_name, const Metadata& metadata, data_size_t num_data) override {
    name = test_name;
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get query boundaries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("For NDCG metric, there should be query information");
    }
    num_queries_ = metadata.num_queries();
    // get query weights
    query_weights_ = metadata.query_weights();
    if (query_weights_ == nullptr) {
      sum_query_weights_ = static_cast<double>(num_queries_);
    } else {
      sum_query_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_queries_; ++i) {
        sum_query_weights_ += query_weights_[i];
      }
    }
    // cache the inverse max DCG for all querys, used to calculate NDCG
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inverse_max_dcgs_.emplace_back(eval_at_.size(), 0.0);
      DCGCalculator::CalMaxDCG(eval_at_, label_ + query_boundaries_[i],
                               query_boundaries_[i + 1] - query_boundaries_[i],
                               &inverse_max_dcgs_[i]);
      for (size_t j = 0; j < inverse_max_dcgs_[i].size(); ++j) {
        if (inverse_max_dcgs_[i][j] > 0.0) {
          inverse_max_dcgs_[i][j] = 1.0 / inverse_max_dcgs_[i][j];
        }
        else {
          // marking negative for all negative querys.
          // if one meet this query, it's ndcg will be set as -1.
          inverse_max_dcgs_[i][j] = -1.0;
        }
      }
    }
  }

  void Print(int iter, const score_t* score, score_t& loss) const override {
    if (early_stopping_round_ > 0 || output_freq_ > 0 && iter % output_freq_ == 0) {
      // some buffers for multi-threading sum up
      std::vector<std::vector<double>> result_buffer_;
      for (int i = 0; i < num_threads_; ++i) {
        result_buffer_.emplace_back(eval_at_.size(), 0.0);
      }
      std::vector<double> tmp_dcg(eval_at_.size(), 0.0);
      if (query_weights_ == nullptr) {
        #pragma omp parallel for schedule(guided) firstprivate(tmp_dcg)
        for (data_size_t i = 0; i < num_queries_; ++i) {
          const int tid = omp_get_thread_num();
          // if all doc in this query are all negative, let its NDCG=1
          if (inverse_max_dcgs_[i][0] <= 0.0) {
            for (size_t j = 0; j < eval_at_.size(); ++j) {
              result_buffer_[tid][j] += 1.0;
            }
          } else {
            // calculate DCG
            DCGCalculator::CalDCG(eval_at_, label_ + query_boundaries_[i],
                                  score + query_boundaries_[i],
                                  query_boundaries_[i + 1] - query_boundaries_[i], &tmp_dcg);
            // calculate NDCG
            for (size_t j = 0; j < eval_at_.size(); ++j) {
              result_buffer_[tid][j] += tmp_dcg[j] * inverse_max_dcgs_[i][j];
            }
          }
        }
      } else {
        #pragma omp parallel for schedule(guided) firstprivate(tmp_dcg)
        for (data_size_t i = 0; i < num_queries_; ++i) {
          const int tid = omp_get_thread_num();
          // if all doc in this query are all negative, let its NDCG=1
          if (inverse_max_dcgs_[i][0] <= 0.0) {
            for (size_t j = 0; j < eval_at_.size(); ++j) {
              result_buffer_[tid][j] += 1.0;
            }
          } else {
            // calculate DCG
            DCGCalculator::CalDCG(eval_at_, label_ + query_boundaries_[i],
                                  score + query_boundaries_[i],
                                  query_boundaries_[i + 1] - query_boundaries_[i], &tmp_dcg);
            // calculate NDCG
            for (size_t j = 0; j < eval_at_.size(); ++j) {
              result_buffer_[tid][j] += tmp_dcg[j] * inverse_max_dcgs_[i][j] * query_weights_[i];
            }
          }
        }
      }
      // Get final average NDCG
      std::vector<double> result(eval_at_.size(), 0.0);
      std::stringstream result_ss;
      for (size_t j = 0; j < result.size(); ++j) {
        for (int i = 0; i < num_threads_; ++i) {
          result[j] += result_buffer_[i][j];
        }
        result[j] /= sum_query_weights_;
        result_ss << "NDCG@" << eval_at_[j] << ":" << result[j] << "\t";
      }
      loss = result[0];
      if (output_freq_ > 0 && iter % output_freq_ == 0){
        Log::Info("Iteration:%d, Test:%s, %s \n", iter, name, result_ss.str().c_str());
      }
    }
  }

private:
  /*! \brief Output frequently */
  int output_freq_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Name of test set */
  const char* name;
  /*! \brief Query boundaries information */
  const data_size_t* query_boundaries_;
  /*! \brief Number of queries */
  data_size_t num_queries_;
  /*! \brief Weights of queries */
  const float* query_weights_;
  /*! \brief Sum weights of queries */
  double sum_query_weights_;
  /*! \brief Evaluate position of NDCG */
  std::vector<data_size_t> eval_at_;
  /*! \brief Cache the inverse max dcg for all queries */
  std::vector<std::vector<double>> inverse_max_dcgs_;
  /*! \brief Number of threads */
  int num_threads_;
};

}  // namespace LightGBM

#endif   // LightGBM_METRIC_RANK_METRIC_HPP_
