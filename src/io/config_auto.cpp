#include<LightGBM/config.h>
namespace LightGBM {

std::unordered_map<std::string, std::string> Config::alias_table(
  {
    { "config", "config_file" },
  { "nthread", "num_threads" },
  { "num_thread", "num_threads" },
  { "random_seed", "seed" },
  { "boosting", "boosting" },
  { "boost", "boosting" },
  { "application", "objective" },
  { "app", "objective" },
  { "train_data", "data" },
  { "train", "data" },
  { "model_output", "output_model" },
  { "model_out", "output_model" },
  { "model_input", "input_model" },
  { "model_in", "input_model" },
  { "predict_result", "output_result" },
  { "prediction_result", "output_result" },
  { "valid", "valid_data" },
  { "test_data", "valid_data" },
  { "test", "valid_data" },
  { "is_sparse", "is_enable_sparse" },
  { "enable_sparse", "is_enable_sparse" },
  { "pre_partition", "pre_partition" },
  { "training_metric", "is_training_metric" },
  { "train_metric", "is_training_metric" },
  { "ndcg_at", "ndcg_eval_at" },
  { "eval_at", "ndcg_eval_at" },
  { "min_data_per_leaf", "min_data_in_leaf" },
  { "min_data", "min_data_in_leaf" },
  { "min_child_samples", "min_data_in_leaf" },
  { "min_sum_hessian_per_leaf", "min_sum_hessian_in_leaf" },
  { "min_sum_hessian", "min_sum_hessian_in_leaf" },
  { "min_hessian", "min_sum_hessian_in_leaf" },
  { "min_child_weight", "min_sum_hessian_in_leaf" },
  { "num_leaf", "num_leaves" },
  { "sub_feature", "feature_fraction" },
  { "colsample_bytree", "feature_fraction" },
  { "num_iteration", "num_iterations" },
  { "num_tree", "num_iterations" },
  { "num_round", "num_iterations" },
  { "num_trees", "num_iterations" },
  { "num_rounds", "num_iterations" },
  { "num_boost_round", "num_iterations" },
  { "n_estimators", "num_iterations" },
  { "sub_row", "bagging_fraction" },
  { "subsample", "bagging_fraction" },
  { "subsample_freq", "bagging_freq" },
  { "shrinkage_rate", "learning_rate" },
  { "tree", "tree_learner" },
  { "num_machine", "num_machines" },
  { "local_port", "local_listen_port" },
  { "two_round_loading", "two_round" },
  { "two_round", "two_round" },
  { "mlist", "machine_list_file" },
  { "is_save_binary", "save_binary" },
  { "save_binary", "save_binary" },
  { "early_stopping_rounds", "early_stopping_round" },
  { "early_stopping", "early_stopping_round" },
  { "verbosity", "verbose" },
  { "header", "header" },
  { "label", "label_column" },
  { "weight", "weight_column" },
  { "group", "group_column" },
  { "query", "group_column" },
  { "query_column", "group_column" },
  { "ignore_feature", "ignore_column" },
  { "blacklist", "ignore_column" },
  { "categorical_feature", "categorical_feature" },
  { "cat_column", "categorical_feature" },
  { "cat_feature", "categorical_feature" },
  { "predict_raw_score", "predict_raw_score" },
  { "raw_score", "predict_raw_score" },
  { "leaf_index", "predict_leaf_index" },
  { "predict_leaf_index", "predict_leaf_index" },
  { "contrib", "predict_contrib" },
  { "predict_contrib", "predict_contrib" },
  { "min_split_gain", "min_gain_to_split" },
  { "topk", "top_k" },
  { "reg_alpha", "lambda_l1" },
  { "reg_lambda", "lambda_l2" },
  { "num_classes", "num_class" },
  { "unbalanced_sets", "is_unbalance" },
  { "bagging_fraction_seed", "bagging_seed" },
  { "workers", "machines" },
  { "nodes", "machines" },
  { "subsample_for_bin", "bin_construct_sample_cnt" },
  { "metric_freq", "metric_freq" },
  { "mc", "monotone_constraints" },
  { "max_tree_output", "max_delta_step" },
  { "max_leaf_output", "max_delta_step" }
  });

std::unordered_set<std::string> Config::parameter_set({
  "config", "config_file", "task", "device",
  "num_threads", "seed", "boosting", "objective", "data",
  "output_model", "input_model", "output_result", "valid_data",
  "is_enable_sparse", "pre_partition", "is_training_metric",
  "ndcg_eval_at", "min_data_in_leaf", "min_sum_hessian_in_leaf",
  "num_leaves", "feature_fraction", "num_iterations",
  "bagging_fraction", "bagging_freq", "learning_rate", "tree_learner",
  "num_machines", "local_listen_port", "two_round",
  "machine_list_file", "save_binary", "early_stopping_round",
  "verbose", "header", "label_column", "weight_column", "group_column",
  "ignore_column", "categorical_feature", "predict_raw_score",
  "predict_leaf_index", "min_gain_to_split", "top_k",
  "lambda_l1", "lambda_l2", "num_class", "is_unbalance",
  "max_depth", "max_bin", "bagging_seed",
  "drop_rate", "skip_drop", "max_drop", "uniform_drop",
  "xgboost_dart_mode", "drop_seed", "top_rate", "other_rate",
  "min_data_in_bin", "data_random_seed", "bin_construct_sample_cnt",
  "num_iteration_predict", "pred_early_stop", "pred_early_stop_freq",
  "pred_early_stop_margin", "use_missing", "sigmoid",
  "fair_c", "poission_max_delta_step", "scale_pos_weight",
  "boost_from_average", "max_position", "label_gain",
  "metric", "metric_freq", "time_out",
  "gpu_platform_id", "gpu_device_id", "gpu_use_dp",
  "convert_model", "convert_model_language",
  "feature_fraction_seed", "enable_bundle", "data", "valid",
  "snapshot_freq", "verbosity", "sparse_threshold", "enable_load_from_binary_file",
  "max_conflict_rate", "poisson_max_delta_step",
  "histogram_pool_size", "is_provide_training_metric", "machine_list_filename", "machines",
  "zero_as_missing", "init_score_file", "valid_init_score_file", "predict_contrib",
  "max_cat_threshold",  "cat_smooth", "min_data_per_group", "cat_l2", "max_cat_to_onehot",
  "alpha", "reg_sqrt", "tweedie_variance_power", "monotone_constraints", "max_delta_step",
  "forced_splits" });

void Config::GetMembersFromString(const std::unordered_map<std::string, std::string>& params) {
  GetInt(params, "num_threads", &num_threads);
  GetString(params, "convert_model_language", &convert_model_language);
  // IO parameters
  GetInt(params, "max_bin", &max_bin);
  CHECK(max_bin > 0);
  GetInt(params, "num_class", &num_class);
  CHECK(num_class > 0);
  GetInt(params, "data_random_seed", &data_random_seed);
  GetString(params, "data", &data);
  GetString(params, "init_score_file", &initscore_filename);
  GetInt(params, "verbose", &verbosity);
  GetInt(params, "num_iteration_predict", &num_iteration_predict);
  GetInt(params, "bin_construct_sample_cnt", &bin_construct_sample_cnt);
  CHECK(bin_construct_sample_cnt > 0);
  GetBool(params, "pre_partition", &pre_partition);
  GetBool(params, "is_enable_sparse", &is_enable_sparse);
  GetDouble(params, "sparse_threshold", &sparse_threshold);
  GetBool(params, "two_round", &two_round);
  GetBool(params, "save_binary", &save_binary);
  GetBool(params, "enable_load_from_binary_file", &enable_load_from_binary_file);
  GetBool(params, "predict_raw_score", &predict_raw_score);
  GetBool(params, "predict_leaf_index", &predict_leaf_index);
  GetBool(params, "predict_contrib", &predict_contrib);
  GetInt(params, "snapshot_freq", &snapshot_freq);
  GetString(params, "output_model", &output_model);
  GetString(params, "input_model", &input_model);
  GetString(params, "convert_model", &convert_model);
  GetString(params, "output_result", &output_result);
  std::string tmp_str = "";

  CHECK(valid.size() == valid_data_initscores.size());
  GetBool(params, "header", &header);
  GetString(params, "label_column", &label_column);
  GetString(params, "weight_column", &weight_column);
  GetString(params, "group_column", &group_column);
  GetString(params, "ignore_column", &ignore_column);
  GetString(params, "categorical_feature", &categorical_feature);
  GetInt(params, "min_data_in_leaf", &min_data_in_leaf);
  GetInt(params, "min_data_in_bin", &min_data_in_bin);
  CHECK(min_data_in_bin > 0);
  CHECK(min_data_in_leaf >= 0);
  GetDouble(params, "max_conflict_rate", &max_conflict_rate);
  CHECK(max_conflict_rate >= 0);
  GetBool(params, "enable_bundle", &enable_bundle);
  GetBool(params, "pred_early_stop", &pred_early_stop);
  GetInt(params, "pred_early_stop_freq", &pred_early_stop_freq);
  GetDouble(params, "pred_early_stop_margin", &pred_early_stop_margin);
  GetBool(params, "use_missing", &use_missing);
  GetBool(params, "zero_as_missing", &zero_as_missing);

  // Objective parameters
  GetBool(params, "is_unbalance", &is_unbalance);
  GetDouble(params, "sigmoid", &sigmoid);
  CHECK(sigmoid > 0);
  GetDouble(params, "fair_c", &fair_c);
  CHECK(fair_c > 0);
  GetDouble(params, "poisson_max_delta_step", &poisson_max_delta_step);
  CHECK(poisson_max_delta_step > 0);
  GetInt(params, "max_position", &max_position);
  CHECK(max_position > 0);
  GetInt(params, "num_class", &num_class);
  CHECK(num_class > 0);
  GetDouble(params, "scale_pos_weight", &scale_pos_weight);
  CHECK(scale_pos_weight > 0);
  GetDouble(params, "alpha", &alpha);
  GetBool(params, "reg_sqrt", &reg_sqrt);
  GetDouble(params, "tweedie_variance_power", &tweedie_variance_power);
  CHECK(tweedie_variance_power >= 1 && tweedie_variance_power < 2);

  if (GetString(params, "monotone_constraints", &tmp_str)) {
    monotone_constraints = Common::StringToArray<int8_t>(tmp_str.c_str(), ',');
  }
  if (GetString(params, "valid_data", &tmp_str)) {
    valid = Common::Split(tmp_str.c_str(), ',');
  }
  if (GetString(params, "valid_init_score_file", &tmp_str)) {
    valid_data_initscores = Common::Split(tmp_str.c_str(), ',');
  } 

  if (GetString(params, "label_gain", &tmp_str)) {
    label_gain = Common::StringToArray<double>(tmp_str, ',');
    label_gain.shrink_to_fit();
  }

  // Metric parameters
  if (GetString(params, "ndcg_eval_at", &tmp_str)) {
    eval_at = Common::StringToArray<int>(tmp_str, ',');
    std::sort(eval_at.begin(), eval_at.end());
    for (size_t i = 0; i < eval_at.size(); ++i) {
      CHECK(eval_at[i] > 0);
    }
    eval_at.shrink_to_fit();
  }

  // Tree parameters
  GetDouble(params, "min_sum_hessian_in_leaf", &min_sum_hessian_in_leaf);
  CHECK(min_data_in_leaf > 0);
  CHECK(min_sum_hessian_in_leaf >= 0);
  GetDouble(params, "lambda_l1", &lambda_l1);
  CHECK(lambda_l1 >= 0.0f);
  GetDouble(params, "lambda_l2", &lambda_l2);
  CHECK(lambda_l2 >= 0.0f);
  GetDouble(params, "max_delta_step", &max_delta_step);
  GetDouble(params, "min_gain_to_split", &min_gain_to_split);
  CHECK(min_gain_to_split >= 0.0f);
  GetInt(params, "num_leaves", &num_leaves);
  CHECK(num_leaves > 1);
  GetInt(params, "feature_fraction_seed", &feature_fraction_seed);
  GetDouble(params, "feature_fraction", &feature_fraction);
  CHECK(feature_fraction > 0.0f && feature_fraction <= 1.0f);
  GetDouble(params, "histogram_pool_size", &histogram_pool_size);
  GetInt(params, "max_depth", &max_depth);
  GetInt(params, "top_k", &top_k);
  CHECK(top_k > 0);
  GetInt(params, "gpu_platform_id", &gpu_platform_id);
  GetInt(params, "gpu_device_id", &gpu_device_id);
  GetBool(params, "gpu_use_dp", &gpu_use_dp);
  GetInt(params, "max_cat_threshold", &max_cat_threshold);
  GetDouble(params, "cat_l2", &cat_l2);
  GetDouble(params, "cat_smooth", &cat_smooth);
  GetInt(params, "min_data_per_group", &min_data_per_group);
  GetInt(params, "max_cat_to_onehot", &max_cat_to_onehot);
  CHECK(max_cat_threshold > 0);
  CHECK(cat_l2 >= 0.0f);
  CHECK(cat_smooth >= 1);
  CHECK(min_data_per_group > 0);
  CHECK(max_cat_to_onehot > 0);
  // Boosting parameters
  GetInt(params, "num_iterations", &num_iterations);
  CHECK(num_iterations >= 0);
  GetInt(params, "bagging_seed", &bagging_seed);
  GetInt(params, "bagging_freq", &bagging_freq);
  GetDouble(params, "bagging_fraction", &bagging_fraction);
  CHECK(bagging_fraction > 0.0f && bagging_fraction <= 1.0f);
  GetDouble(params, "learning_rate", &learning_rate);
  CHECK(learning_rate > 0.0f);
  GetInt(params, "early_stopping_round", &early_stopping_round);
  CHECK(early_stopping_round >= 0);
  GetInt(params, "metric_freq", &metric_freq);
  CHECK(metric_freq >= 0);
  GetBool(params, "is_training_metric", &is_provide_training_metric);
  GetInt(params, "num_class", &num_class);
  CHECK(num_class > 0);
  GetInt(params, "drop_seed", &drop_seed);
  GetDouble(params, "drop_rate", &drop_rate);
  GetDouble(params, "skip_drop", &skip_drop);
  CHECK(drop_rate <= 1.0 && drop_rate >= 0.0);
  CHECK(skip_drop <= 1.0 && skip_drop >= 0.0);
  GetInt(params, "max_drop", &max_drop);
  GetBool(params, "xgboost_dart_mode", &xgboost_dart_mode);
  GetBool(params, "uniform_drop", &uniform_drop);
  GetDouble(params, "top_rate", &top_rate);
  GetDouble(params, "other_rate", &other_rate);
  CHECK(top_rate > 0);
  CHECK(other_rate > 0);
  CHECK(top_rate + other_rate <= 1.0);
  GetBool(params, "boost_from_average", &boost_from_average);
  GetString(params, "forced_splits", &forcedsplits_filename);

  // Network Parameters
  GetInt(params, "num_machines", &num_machines);
  CHECK(num_machines >= 1);
  GetInt(params, "local_listen_port", &local_listen_port);
  CHECK(local_listen_port > 0);
  GetInt(params, "time_out", &time_out);
  CHECK(time_out > 0);
  GetString(params, "machine_list_file", &machine_list_filename);
  GetString(params, "machines", &machines);
}

}