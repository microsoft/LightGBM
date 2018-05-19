#include<LightGBM/config.h>
namespace LightGBM {
std::unordered_map<std::string, std::string> Config::alias_table({
  {"config_file", "config"}, 
  {"task_type", "task"}, 
  {"application", "objective"}, 
  {"app", "objective"}, 
  {"objective_type", "objective"}, 
  {"boosting_type", "boosting"}, 
  {"boost", "boosting"}, 
  {"train", "data"}, 
  {"train_data", "data"}, 
  {"data_filename", "data"}, 
  {"test", "valid"}, 
  {"valid_data", "valid"}, 
  {"test_data", "valid"}, 
  {"valid_filenames", "valid"}, 
  {"num_iteration", "num_iterations"}, 
  {"num_tree", "num_iterations"}, 
  {"num_trees", "num_iterations"}, 
  {"num_round", "num_iterations"}, 
  {"num_rounds", "num_iterations"}, 
  {"num_boost_round", "num_iterations"}, 
  {"n_estimators", "num_iterations"}, 
  {"shrinkage_rate", "learning_rate"}, 
  {"num_leaf", "num_leaves"}, 
  {"tree", "tree_learner"}, 
  {"num_thread", "num_threads"}, 
  {"nthread", "num_threads"}, 
  {"min_data_per_leaf", "min_data_in_leaf"}, 
  {"min_data", "min_data_in_leaf"}, 
  {"min_child_samples", "min_data_in_leaf"}, 
  {"min_sum_hessian_per_leaf", "min_sum_hessian_in_leaf"}, 
  {"min_sum_hessian", "min_sum_hessian_in_leaf"}, 
  {"min_hessian", "min_sum_hessian_in_leaf"}, 
  {"min_child_weight", "min_sum_hessian_in_leaf"}, 
  {"sub_row", "bagging_fraction"}, 
  {"subsample", "bagging_fraction"}, 
  {"bagging", "bagging_fraction"}, 
  {"subsample_freq", "bagging_freq"}, 
  {"bagging_fraction_seed", "bagging_seed"}, 
  {"sub_feature", "feature_fraction"}, 
  {"colsample_bytree", "feature_fraction"}, 
  {"early_stopping_rounds", "early_stopping_round"}, 
  {"early_stopping", "early_stopping_round"}, 
  {"max_tree_output", "max_delta_step"}, 
  {"max_leaf_output", "max_delta_step"}, 
  {"reg_alpha", "lambda_l1"}, 
  {"reg_lambda", "lambda_l2"}, 
  {"min_gain_to_split", "min_gain_to_split"}, 
  {"min_split_gain", "min_gain_to_split"}, 
  {"topk", "top_k"}, 
  {"mc", "monotone_constraints"}, 
  {"model_output", "output_model"}, 
  {"model_out", "output_model"}, 
  {"model_input", "input_model"}, 
  {"model_in", "input_model"}, 
  {"predict_result", "output_result"}, 
  {"prediction_result", "output_result"}, 
  {"is_pre_partition", "pre_partition"}, 
  {"is_sparse", "is_enable_sparse"}, 
  {"enable_sparse", "is_enable_sparse"}, 
  {"two_round_loading", "two_round"}, 
  {"use_two_round_loading", "two_round"}, 
  {"is_save_binary", "save_binary"}, 
  {"is_save_binary_file", "save_binary"}, 
  {"verbose", "verbosity"}, 
  {"has_header", "header"}, 
  {"label", "label_column"}, 
  {"weight", "weight_column"}, 
  {"query_column", "group_column"}, 
  {"group", "group_column"}, 
  {"query", "group_column"}, 
  {"ignore_feature", "ignore_column"}, 
  {"blacklist", "ignore_column"}, 
  {"categorical_column", "categorical_feature"}, 
  {"cat_feature", "categorical_feature"}, 
  {"cat_column", "categorical_feature"}, 
  {"raw_score", "predict_raw_score"}, 
  {"is_predict_raw_score", "predict_raw_score"}, 
  {"predict_rawscore", "predict_raw_score"}, 
  {"leaf_index", "predict_leaf_index"}, 
  {"is_predict_leaf_index", "predict_leaf_index"}, 
  {"contrib", "predict_contrib"}, 
  {"is_predict_contrib", "predict_contrib"}, 
  {"subsample_for_bin", "bin_construct_sample_cnt"}, 
  {"num_classes", "num_class"}, 
  {"unbalanced_sets", "is_unbalance"}, 
  {"metric_types", "metric"}, 
  {"output_freq", "metric_freq"}, 
  {"training_metric", "is_provide_training_metric"}, 
  {"is_training_metric", "is_provide_training_metric"}, 
  {"train_metric", "is_provide_training_metric"}, 
  {"ndcg_eval_at", "eval_at"}, 
  {"ndcg_at", "eval_at"}, 
});

std::unordered_set<std::string> Config::parameter_set({
  "config", 
  "task", 
  "objective", 
  "boosting", 
  "data", 
  "valid", 
  "num_iterations", 
  "learning_rate", 
  "num_leaves", 
  "tree_learner", 
  "num_threads", 
  "device_type", 
  "seed", 
  "max_depth", 
  "min_data_in_leaf", 
  "min_sum_hessian_in_leaf", 
  "bagging_fraction", 
  "bagging_freq", 
  "bagging_seed", 
  "feature_fraction", 
  "feature_fraction_seed", 
  "early_stopping_round", 
  "max_delta_step", 
  "lambda_l1", 
  "lambda_l2", 
  "min_gain_to_split", 
  "drop_rate", 
  "max_drop", 
  "skip_drop", 
  "xgboost_dart_mode", 
  "uniform_drop", 
  "drop_seed", 
  "top_rate", 
  "other_rate", 
  "min_data_per_group", 
  "max_cat_threshold", 
  "cat_l2", 
  "cat_smooth", 
  "max_cat_to_onehot", 
  "top_k", 
  "monotone_constraints", 
  "forcedsplits_filename", 
  "max_bin", 
  "min_data_in_bin", 
  "data_random_seed", 
  "output_model", 
  "input_model", 
  "output_result", 
  "pre_partition", 
  "is_enable_sparse", 
  "sparse_threshold", 
  "two_round", 
  "save_binary", 
  "verbosity", 
  "header", 
  "label_column", 
  "weight_column", 
  "group_column", 
  "ignore_column", 
  "categorical_feature", 
  "predict_raw_score", 
  "predict_leaf_index", 
  "predict_contrib", 
  "num_iteration_predict", 
  "pred_early_stop", 
  "pred_early_stop_freq", 
  "pred_early_stop_margin", 
  "bin_construct_sample_cnt", 
  "use_missing", 
  "zero_as_missing", 
  "initscore_filename", 
  "valid_data_initscores", 
  "histogram_pool_size", 
  "num_class", 
  "enable_load_from_binary_file", 
  "enable_bundle", 
  "max_conflict_rate", 
  "snapshot_freq", 
  "convert_model_language", 
  "convert_model", 
  "sigmoid", 
  "alpha", 
  "fair_c", 
  "poisson_max_delta_step", 
  "boost_from_average", 
  "is_unbalance", 
  "scale_pos_weight", 
  "reg_sqrt", 
  "tweedie_variance_power", 
  "label_gain", 
  "max_position", 
  "metric", 
  "metric_freq", 
  "is_provide_training_metric", 
  "eval_at", 
  "gpu_platform_id", 
  "gpu_device_id", 
  "gpu_use_dp", 
});

void Config::GetMembersFromString(const std::unordered_map<std::string, std::string>& params) {
  std::string tmp_str = "";
  GetString(params, "data", &data);

  if (GetString(params, "valid", &tmp_str)) {
    valid = Common::Split(tmp_str.c_str(), ',');
  }

  GetInt(params, "num_iterations", &num_iterations);
  CHECK(num_iterations >=0);

  GetDouble(params, "learning_rate", &learning_rate);
  CHECK(learning_rate >0);

  GetInt(params, "num_leaves", &num_leaves);
  CHECK(num_leaves >1);

  GetInt(params, "num_threads", &num_threads);

  GetInt(params, "max_depth", &max_depth);

  GetInt(params, "min_data_in_leaf", &min_data_in_leaf);
  CHECK(min_data_in_leaf >=0);

  GetDouble(params, "min_sum_hessian_in_leaf", &min_sum_hessian_in_leaf);

  GetDouble(params, "bagging_fraction", &bagging_fraction);
  CHECK(bagging_fraction >0);
  CHECK(bagging_fraction <=1.0);

  GetInt(params, "bagging_freq", &bagging_freq);

  GetInt(params, "bagging_seed", &bagging_seed);

  GetDouble(params, "feature_fraction", &feature_fraction);
  CHECK(feature_fraction >0);
  CHECK(feature_fraction <=1.0);

  GetInt(params, "feature_fraction_seed", &feature_fraction_seed);

  GetInt(params, "early_stopping_round", &early_stopping_round);

  GetDouble(params, "max_delta_step", &max_delta_step);

  GetDouble(params, "lambda_l1", &lambda_l1);
  CHECK(lambda_l1 >=0);

  GetDouble(params, "lambda_l2", &lambda_l2);
  CHECK(lambda_l2 >=0);

  GetDouble(params, "min_gain_to_split", &min_gain_to_split);

  GetDouble(params, "drop_rate", &drop_rate);
  CHECK(drop_rate >=0);
  CHECK(drop_rate <=1.0);

  GetInt(params, "max_drop", &max_drop);

  GetDouble(params, "skip_drop", &skip_drop);
  CHECK(skip_drop >=0);
  CHECK(skip_drop <=1.0);

  GetBool(params, "xgboost_dart_mode", &xgboost_dart_mode);

  GetBool(params, "uniform_drop", &uniform_drop);

  GetInt(params, "drop_seed", &drop_seed);

  GetDouble(params, "top_rate", &top_rate);
  CHECK(top_rate >=0);
  CHECK(top_rate <=1.0);

  GetDouble(params, "other_rate", &other_rate);
  CHECK(other_rate >=0);
  CHECK(other_rate <=1.0);

  GetInt(params, "min_data_per_group", &min_data_per_group);
  CHECK(min_data_per_group >0);

  GetInt(params, "max_cat_threshold", &max_cat_threshold);
  CHECK(max_cat_threshold >0);

  GetDouble(params, "cat_l2", &cat_l2);
  CHECK(cat_l2 >=0);

  GetDouble(params, "cat_smooth", &cat_smooth);
  CHECK(cat_smooth >=0);

  GetInt(params, "max_cat_to_onehot", &max_cat_to_onehot);
  CHECK(max_cat_to_onehot >0);

  GetInt(params, "top_k", &top_k);

  if (GetString(params, "monotone_constraints", &tmp_str)) {
    monotone_constraints = Common::StringToArray<int8_t>(tmp_str, ',');
  }

  GetString(params, "forcedsplits_filename", &forcedsplits_filename);

  GetInt(params, "max_bin", &max_bin);
  CHECK(max_bin >1);

  GetInt(params, "min_data_in_bin", &min_data_in_bin);
  CHECK(min_data_in_bin >0);

  GetInt(params, "data_random_seed", &data_random_seed);

  GetString(params, "output_model", &output_model);

  GetString(params, "input_model", &input_model);

  GetString(params, "output_result", &output_result);

  GetBool(params, "pre_partition", &pre_partition);

  GetBool(params, "is_enable_sparse", &is_enable_sparse);

  GetDouble(params, "sparse_threshold", &sparse_threshold);
  CHECK(sparse_threshold >0);
  CHECK(sparse_threshold <=1);

  GetBool(params, "two_round", &two_round);

  GetBool(params, "save_binary", &save_binary);

  GetInt(params, "verbosity", &verbosity);

  GetBool(params, "header", &header);

  GetString(params, "label_column", &label_column);

  GetString(params, "weight_column", &weight_column);

  GetString(params, "group_column", &group_column);

  GetString(params, "ignore_column", &ignore_column);

  GetString(params, "categorical_feature", &categorical_feature);

  GetBool(params, "predict_raw_score", &predict_raw_score);

  GetBool(params, "predict_leaf_index", &predict_leaf_index);

  GetBool(params, "predict_contrib", &predict_contrib);

  GetInt(params, "num_iteration_predict", &num_iteration_predict);

  GetBool(params, "pred_early_stop", &pred_early_stop);

  GetInt(params, "pred_early_stop_freq", &pred_early_stop_freq);

  GetDouble(params, "pred_early_stop_margin", &pred_early_stop_margin);

  GetInt(params, "bin_construct_sample_cnt", &bin_construct_sample_cnt);
  CHECK(bin_construct_sample_cnt >0);

  GetBool(params, "use_missing", &use_missing);

  GetBool(params, "zero_as_missing", &zero_as_missing);

  GetString(params, "initscore_filename", &initscore_filename);

  if (GetString(params, "valid_data_initscores", &tmp_str)) {
    valid_data_initscores = Common::Split(tmp_str.c_str(), ',');
  }

  GetDouble(params, "histogram_pool_size", &histogram_pool_size);

  GetInt(params, "num_class", &num_class);

  GetBool(params, "enable_load_from_binary_file", &enable_load_from_binary_file);

  GetBool(params, "enable_bundle", &enable_bundle);

  GetDouble(params, "max_conflict_rate", &max_conflict_rate);
  CHECK(max_conflict_rate >=0);

  GetInt(params, "snapshot_freq", &snapshot_freq);

  GetString(params, "convert_model_language", &convert_model_language);

  GetString(params, "convert_model", &convert_model);

  GetDouble(params, "sigmoid", &sigmoid);

  GetDouble(params, "alpha", &alpha);

  GetDouble(params, "fair_c", &fair_c);

  GetDouble(params, "poisson_max_delta_step", &poisson_max_delta_step);

  GetBool(params, "boost_from_average", &boost_from_average);

  GetBool(params, "is_unbalance", &is_unbalance);

  GetDouble(params, "scale_pos_weight", &scale_pos_weight);
  CHECK(scale_pos_weight >0);

  GetBool(params, "reg_sqrt", &reg_sqrt);

  GetDouble(params, "tweedie_variance_power", &tweedie_variance_power);

  if (GetString(params, "label_gain", &tmp_str)) {
    label_gain = Common::StringToArray<double>(tmp_str, ',');
  }

  GetInt(params, "max_position", &max_position);
  CHECK(max_position >0);

  GetInt(params, "metric_freq", &metric_freq);
  CHECK(metric_freq >0);

  GetBool(params, "is_provide_training_metric", &is_provide_training_metric);

  if (GetString(params, "eval_at", &tmp_str)) {
    eval_at = Common::StringToArray<int>(tmp_str, ',');
  }

  GetInt(params, "gpu_platform_id", &gpu_platform_id);

  GetInt(params, "gpu_device_id", &gpu_device_id);

  GetBool(params, "gpu_use_dp", &gpu_use_dp);

}

std::string Config::SaveMembersToString() const {
  std::stringstream str_buf;
  str_buf << "data=" << data << "\n";
  str_buf << "valid=" << Common::Join(valid,",") << "\n";
  str_buf << "num_iterations=" << num_iterations << "\n";
  str_buf << "learning_rate=" << learning_rate << "\n";
  str_buf << "num_leaves=" << num_leaves << "\n";
  str_buf << "num_threads=" << num_threads << "\n";
  str_buf << "max_depth=" << max_depth << "\n";
  str_buf << "min_data_in_leaf=" << min_data_in_leaf << "\n";
  str_buf << "min_sum_hessian_in_leaf=" << min_sum_hessian_in_leaf << "\n";
  str_buf << "bagging_fraction=" << bagging_fraction << "\n";
  str_buf << "bagging_freq=" << bagging_freq << "\n";
  str_buf << "bagging_seed=" << bagging_seed << "\n";
  str_buf << "feature_fraction=" << feature_fraction << "\n";
  str_buf << "feature_fraction_seed=" << feature_fraction_seed << "\n";
  str_buf << "early_stopping_round=" << early_stopping_round << "\n";
  str_buf << "max_delta_step=" << max_delta_step << "\n";
  str_buf << "lambda_l1=" << lambda_l1 << "\n";
  str_buf << "lambda_l2=" << lambda_l2 << "\n";
  str_buf << "min_gain_to_split=" << min_gain_to_split << "\n";
  str_buf << "drop_rate=" << drop_rate << "\n";
  str_buf << "max_drop=" << max_drop << "\n";
  str_buf << "skip_drop=" << skip_drop << "\n";
  str_buf << "xgboost_dart_mode=" << xgboost_dart_mode << "\n";
  str_buf << "uniform_drop=" << uniform_drop << "\n";
  str_buf << "drop_seed=" << drop_seed << "\n";
  str_buf << "top_rate=" << top_rate << "\n";
  str_buf << "other_rate=" << other_rate << "\n";
  str_buf << "min_data_per_group=" << min_data_per_group << "\n";
  str_buf << "max_cat_threshold=" << max_cat_threshold << "\n";
  str_buf << "cat_l2=" << cat_l2 << "\n";
  str_buf << "cat_smooth=" << cat_smooth << "\n";
  str_buf << "max_cat_to_onehot=" << max_cat_to_onehot << "\n";
  str_buf << "top_k=" << top_k << "\n";
  str_buf << "monotone_constraints=" << Common::Join(Common::ArrayCast<int8_t, int>(monotone_constraints),",") << "\n";
  str_buf << "forcedsplits_filename=" << forcedsplits_filename << "\n";
  str_buf << "max_bin=" << max_bin << "\n";
  str_buf << "min_data_in_bin=" << min_data_in_bin << "\n";
  str_buf << "data_random_seed=" << data_random_seed << "\n";
  str_buf << "output_model=" << output_model << "\n";
  str_buf << "input_model=" << input_model << "\n";
  str_buf << "output_result=" << output_result << "\n";
  str_buf << "pre_partition=" << pre_partition << "\n";
  str_buf << "is_enable_sparse=" << is_enable_sparse << "\n";
  str_buf << "sparse_threshold=" << sparse_threshold << "\n";
  str_buf << "two_round=" << two_round << "\n";
  str_buf << "save_binary=" << save_binary << "\n";
  str_buf << "verbosity=" << verbosity << "\n";
  str_buf << "header=" << header << "\n";
  str_buf << "label_column=" << label_column << "\n";
  str_buf << "weight_column=" << weight_column << "\n";
  str_buf << "group_column=" << group_column << "\n";
  str_buf << "ignore_column=" << ignore_column << "\n";
  str_buf << "categorical_feature=" << categorical_feature << "\n";
  str_buf << "predict_raw_score=" << predict_raw_score << "\n";
  str_buf << "predict_leaf_index=" << predict_leaf_index << "\n";
  str_buf << "predict_contrib=" << predict_contrib << "\n";
  str_buf << "num_iteration_predict=" << num_iteration_predict << "\n";
  str_buf << "pred_early_stop=" << pred_early_stop << "\n";
  str_buf << "pred_early_stop_freq=" << pred_early_stop_freq << "\n";
  str_buf << "pred_early_stop_margin=" << pred_early_stop_margin << "\n";
  str_buf << "bin_construct_sample_cnt=" << bin_construct_sample_cnt << "\n";
  str_buf << "use_missing=" << use_missing << "\n";
  str_buf << "zero_as_missing=" << zero_as_missing << "\n";
  str_buf << "initscore_filename=" << initscore_filename << "\n";
  str_buf << "valid_data_initscores=" << Common::Join(valid_data_initscores,",") << "\n";
  str_buf << "histogram_pool_size=" << histogram_pool_size << "\n";
  str_buf << "num_class=" << num_class << "\n";
  str_buf << "enable_load_from_binary_file=" << enable_load_from_binary_file << "\n";
  str_buf << "enable_bundle=" << enable_bundle << "\n";
  str_buf << "max_conflict_rate=" << max_conflict_rate << "\n";
  str_buf << "snapshot_freq=" << snapshot_freq << "\n";
  str_buf << "convert_model_language=" << convert_model_language << "\n";
  str_buf << "convert_model=" << convert_model << "\n";
  str_buf << "sigmoid=" << sigmoid << "\n";
  str_buf << "alpha=" << alpha << "\n";
  str_buf << "fair_c=" << fair_c << "\n";
  str_buf << "poisson_max_delta_step=" << poisson_max_delta_step << "\n";
  str_buf << "boost_from_average=" << boost_from_average << "\n";
  str_buf << "is_unbalance=" << is_unbalance << "\n";
  str_buf << "scale_pos_weight=" << scale_pos_weight << "\n";
  str_buf << "reg_sqrt=" << reg_sqrt << "\n";
  str_buf << "tweedie_variance_power=" << tweedie_variance_power << "\n";
  str_buf << "label_gain=" << Common::Join(label_gain,",") << "\n";
  str_buf << "max_position=" << max_position << "\n";
  str_buf << "metric_freq=" << metric_freq << "\n";
  str_buf << "is_provide_training_metric=" << is_provide_training_metric << "\n";
  str_buf << "eval_at=" << Common::Join(eval_at,",") << "\n";
  str_buf << "gpu_platform_id=" << gpu_platform_id << "\n";
  str_buf << "gpu_device_id=" << gpu_device_id << "\n";
  str_buf << "gpu_use_dp=" << gpu_use_dp << "\n";
  return str_buf.str();
}

}
