#include "gbdt.h"

#include <LightGBM/utils/common.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>

#include <sstream>
#include <string>
#include <vector>

namespace LightGBM {

const std::string kModelVersion = "v2";

std::string GBDT::DumpModel(int num_iteration) const {
  std::stringstream str_buf;

  str_buf << "{";
  str_buf << "\"name\":\"" << SubModelName() << "\"," << std::endl;
  str_buf << "\"version\":\"" << kModelVersion << "\"," << std::endl;
  str_buf << "\"num_class\":" << num_class_ << "," << std::endl;
  str_buf << "\"num_tree_per_iteration\":" << num_tree_per_iteration_ << "," << std::endl;
  str_buf << "\"label_index\":" << label_idx_ << "," << std::endl;
  str_buf << "\"max_feature_idx\":" << max_feature_idx_ << "," << std::endl;

  str_buf << "\"feature_names\":[\""
    << Common::Join(feature_names_, "\",\"") << "\"],"
    << std::endl;

  str_buf << "\"tree_info\":[";
  int num_used_model = static_cast<int>(models_.size());
  if (num_iteration > 0) {
    num_used_model = std::min(num_iteration * num_tree_per_iteration_, num_used_model);
  }
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << ",";
    }
    str_buf << "{";
    str_buf << "\"tree_index\":" << i << ",";
    str_buf << models_[i]->ToJSON();
    str_buf << "}";
  }
  str_buf << "]" << std::endl;

  str_buf << "}" << std::endl;

  return str_buf.str();
}

std::string GBDT::ModelToIfElse(int num_iteration) const {
  std::stringstream str_buf;

  str_buf << "#include \"gbdt.h\"" << std::endl;
  str_buf << "#include <LightGBM/utils/common.h>" << std::endl;
  str_buf << "#include <LightGBM/objective_function.h>" << std::endl;
  str_buf << "#include <LightGBM/metric.h>" << std::endl;
  str_buf << "#include <LightGBM/prediction_early_stop.h>" << std::endl;
  str_buf << "#include <ctime>" << std::endl;
  str_buf << "#include <sstream>" << std::endl;
  str_buf << "#include <chrono>" << std::endl;
  str_buf << "#include <string>" << std::endl;
  str_buf << "#include <vector>" << std::endl;
  str_buf << "#include <utility>" << std::endl;
  str_buf << "namespace LightGBM {" << std::endl;

  int num_used_model = static_cast<int>(models_.size());
  if (num_iteration > 0) {
    num_used_model = std::min(num_iteration * num_tree_per_iteration_, num_used_model);
  }

  // PredictRaw
  for (int i = 0; i < num_used_model; ++i) {
    str_buf << models_[i]->ToIfElse(i, false) << std::endl;
  }

  str_buf << "double (*PredictTreePtr[])(const double*) = { ";
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << " , ";
    }
    str_buf << "PredictTree" << i;
  }
  str_buf << " };" << std::endl << std::endl;

  std::stringstream pred_str_buf;

  pred_str_buf << "\t" << "int early_stop_round_counter = 0;" << std::endl;
  pred_str_buf << "\t" << "std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);" << std::endl;
  pred_str_buf << "\t" << "for (int i = 0; i < num_iteration_for_pred_; ++i) {" << std::endl;
  pred_str_buf << "\t\t" << "for (int k = 0; k < num_tree_per_iteration_; ++k) {" << std::endl;
  pred_str_buf << "\t\t\t" << "output[k] += (*PredictTreePtr[i * num_tree_per_iteration_ + k])(features);" << std::endl;
  pred_str_buf << "\t\t" << "}" << std::endl;
  pred_str_buf << "\t\t" << "++early_stop_round_counter;" << std::endl;
  pred_str_buf << "\t\t" << "if (early_stop->round_period == early_stop_round_counter) {" << std::endl;
  pred_str_buf << "\t\t\t" << "if (early_stop->callback_function(output, num_tree_per_iteration_))" << std::endl;
  pred_str_buf << "\t\t\t\t" << "return;" << std::endl;
  pred_str_buf << "\t\t\t" << "early_stop_round_counter = 0;" << std::endl;
  pred_str_buf << "\t\t" << "}" << std::endl;
  pred_str_buf << "\t" << "}" << std::endl;

  str_buf << "void GBDT::PredictRaw(const double* features, double *output, const PredictionEarlyStopInstance* early_stop) const {" << std::endl;
  str_buf << pred_str_buf.str();
  str_buf << "}" << std::endl;
  str_buf << std::endl;

  // PredictRawByMap
  str_buf << "double (*PredictTreeByMapPtr[])(const std::unordered_map<int, double>&) = { ";
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << " , ";
    }
    str_buf << "PredictTree" << i << "ByMap";
  }
  str_buf << " };" << std::endl << std::endl;

  std::stringstream pred_str_buf_map;

  pred_str_buf_map << "\t" << "int early_stop_round_counter = 0;" << std::endl;
  pred_str_buf_map << "\t" << "std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);" << std::endl;
  pred_str_buf_map << "\t" << "for (int i = 0; i < num_iteration_for_pred_; ++i) {" << std::endl;
  pred_str_buf_map << "\t\t" << "for (int k = 0; k < num_tree_per_iteration_; ++k) {" << std::endl;
  pred_str_buf_map << "\t\t\t" << "output[k] += (*PredictTreeByMapPtr[i * num_tree_per_iteration_ + k])(features);" << std::endl;
  pred_str_buf_map << "\t\t" << "}" << std::endl;
  pred_str_buf_map << "\t\t" << "++early_stop_round_counter;" << std::endl;
  pred_str_buf_map << "\t\t" << "if (early_stop->round_period == early_stop_round_counter) {" << std::endl;
  pred_str_buf_map << "\t\t\t" << "if (early_stop->callback_function(output, num_tree_per_iteration_))" << std::endl;
  pred_str_buf_map << "\t\t\t\t" << "return;" << std::endl;
  pred_str_buf_map << "\t\t\t" << "early_stop_round_counter = 0;" << std::endl;
  pred_str_buf_map << "\t\t" << "}" << std::endl;
  pred_str_buf_map << "\t" << "}" << std::endl;

  str_buf << "void GBDT::PredictRawByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {" << std::endl;
  str_buf << pred_str_buf_map.str();
  str_buf << "}" << std::endl;
  str_buf << std::endl;

  // Predict
  str_buf << "void GBDT::Predict(const double* features, double *output, const PredictionEarlyStopInstance* early_stop) const {" << std::endl;
  str_buf << "\t" << "PredictRaw(features, output, early_stop);" << std::endl;
  str_buf << "\t" << "if (average_output_) {" << std::endl;
  str_buf << "\t\t" << "for (int k = 0; k < num_tree_per_iteration_; ++k) {" << std::endl;
  str_buf << "\t\t\t" << "output[k] /= num_iteration_for_pred_;" << std::endl;
  str_buf << "\t\t" << "}" << std::endl;
  str_buf << "\t" << "}" << std::endl;
  str_buf << "\t" << "else if (objective_function_ != nullptr) {" << std::endl;
  str_buf << "\t\t" << "objective_function_->ConvertOutput(output, output);" << std::endl;
  str_buf << "\t" << "}" << std::endl;
  str_buf << "}" << std::endl;
  str_buf << std::endl;

  // PredictByMap
  str_buf << "void GBDT::PredictByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {" << std::endl;
  str_buf << "\t" << "PredictRawByMap(features, output, early_stop);" << std::endl;
  str_buf << "\t" << "if (average_output_) {" << std::endl;
  str_buf << "\t\t" << "for (int k = 0; k < num_tree_per_iteration_; ++k) {" << std::endl;
  str_buf << "\t\t\t" << "output[k] /= num_iteration_for_pred_;" << std::endl;
  str_buf << "\t\t" << "}" << std::endl;
  str_buf << "\t" << "}" << std::endl;
  str_buf << "\t" << "else if (objective_function_ != nullptr) {" << std::endl;
  str_buf << "\t\t" << "objective_function_->ConvertOutput(output, output);" << std::endl;
  str_buf << "\t" << "}" << std::endl;
  str_buf << "}" << std::endl;
  str_buf << std::endl;


  // PredictLeafIndex
  for (int i = 0; i < num_used_model; ++i) {
    str_buf << models_[i]->ToIfElse(i, true) << std::endl;
  }

  str_buf << "double (*PredictTreeLeafPtr[])(const double*) = { ";
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << " , ";
    }
    str_buf << "PredictTree" << i << "Leaf";
  }
  str_buf << " };" << std::endl << std::endl;

  str_buf << "void GBDT::PredictLeafIndex(const double* features, double *output) const {" << std::endl;
  str_buf << "\t" << "int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;" << std::endl;
  str_buf << "\t" << "for (int i = 0; i < total_tree; ++i) {" << std::endl;
  str_buf << "\t\t" << "output[i] = (*PredictTreeLeafPtr[i])(features);" << std::endl;
  str_buf << "\t" << "}" << std::endl;
  str_buf << "}" << std::endl;

  //PredictLeafIndexByMap
  str_buf << "double (*PredictTreeLeafByMapPtr[])(const std::unordered_map<int, double>&) = { ";
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << " , ";
    }
    str_buf << "PredictTree" << i << "LeafByMap";
  }
  str_buf << " };" << std::endl << std::endl;

  str_buf << "void GBDT::PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const {" << std::endl;
  str_buf << "\t" << "int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;" << std::endl;
  str_buf << "\t" << "for (int i = 0; i < total_tree; ++i) {" << std::endl;
  str_buf << "\t\t" << "output[i] = (*PredictTreeLeafByMapPtr[i])(features);" << std::endl;
  str_buf << "\t" << "}" << std::endl;
  str_buf << "}" << std::endl;

  str_buf << "}  // namespace LightGBM" << std::endl;

  return str_buf.str();
}

bool GBDT::SaveModelToIfElse(int num_iteration, const char* filename) const {
  /*! \brief File to write models */
  std::ofstream output_file;
  std::ifstream ifs(filename);
  if (ifs.good()) {
    std::string origin((std::istreambuf_iterator<char>(ifs)),
      (std::istreambuf_iterator<char>()));
    output_file.open(filename);
    output_file << "#define USE_HARD_CODE 0" << std::endl;
    output_file << "#ifndef USE_HARD_CODE" << std::endl;
    output_file << origin << std::endl;
    output_file << "#else" << std::endl;
    output_file << ModelToIfElse(num_iteration);
    output_file << "#endif" << std::endl;
  } else {
    output_file.open(filename);
    output_file << ModelToIfElse(num_iteration);
  }

  ifs.close();
  output_file.close();

  return (bool)output_file;
}

std::string GBDT::SaveModelToString(int num_iteration) const {
  std::stringstream ss;

  // output model type
  ss << SubModelName() << std::endl;
  ss << "version=" << kModelVersion << std::endl;
  // output number of class
  ss << "num_class=" << num_class_ << std::endl;
  ss << "num_tree_per_iteration=" << num_tree_per_iteration_ << std::endl;
  // output label index
  ss << "label_index=" << label_idx_ << std::endl;
  // output max_feature_idx
  ss << "max_feature_idx=" << max_feature_idx_ << std::endl;
  // output objective
  if (objective_function_ != nullptr) {
    ss << "objective=" << objective_function_->ToString() << std::endl;
  }

  if (average_output_) {
    ss << "average_output" << std::endl;
  }

  ss << "feature_names=" << Common::Join(feature_names_, " ") << std::endl;

  ss << "feature_infos=" << Common::Join(feature_infos_, " ") << std::endl;

  std::vector<double> feature_importances = FeatureImportance(num_iteration, 0);

  ss << std::endl;
  int num_used_model = static_cast<int>(models_.size());
  if (num_iteration > 0) {
    num_used_model = std::min(num_iteration * num_tree_per_iteration_, num_used_model);
  }
  // output tree models
  for (int i = 0; i < num_used_model; ++i) {
    ss << "Tree=" << i << std::endl;
    ss << models_[i]->ToString() << std::endl;
  }

  // store the importance first
  std::vector<std::pair<size_t, std::string>> pairs;
  for (size_t i = 0; i < feature_importances.size(); ++i) {
    size_t feature_importances_int = static_cast<size_t>(feature_importances[i]);
    if (feature_importances_int > 0) {
      pairs.emplace_back(feature_importances_int, feature_names_[i]);
    }
  }
  // sort the importance
  std::sort(pairs.begin(), pairs.end(),
            [](const std::pair<size_t, std::string>& lhs,
               const std::pair<size_t, std::string>& rhs) {
    return lhs.first > rhs.first;
  });
  ss << std::endl << "feature importances:" << std::endl;
  for (size_t i = 0; i < pairs.size(); ++i) {
    ss << pairs[i].second << "=" << std::to_string(pairs[i].first) << std::endl;
  }

  return ss.str();
}

bool GBDT::SaveModelToFile(int num_iteration, const char* filename) const {
  /*! \brief File to write models */
  std::ofstream output_file;
  output_file.open(filename);

  output_file << SaveModelToString(num_iteration);

  output_file.close();

  return (bool)output_file;
}

bool GBDT::LoadModelFromString(const std::string& model_str) {
  // use serialized string to restore this object
  models_.clear();
  auto c_str = model_str.c_str();
  auto p = c_str;
  std::unordered_map<std::string, std::string> key_vals;
  while (*p != '\0') {
    auto line_len = Common::GetLine(p);
    std::string cur_line(p, line_len);
    if (!Common::StartsWith(cur_line, "Tree=")) {
      auto strs = Common::Split(cur_line.c_str(), '=');
      if (strs.size() == 1) {
        key_vals[strs[0]] = "";
      } else if (strs.size() == 2) {
        key_vals[strs[0]] = strs[1];
      } else if (strs.size() > 2) {
        Log::Fatal("Wrong line at model file: %s", cur_line.c_str());
      }
    } else {
      break;
    }
    p += line_len;
    p = Common::SkipNewLine(p);
  }

  // get number of classes
  if (key_vals.count("num_class")) {
    Common::Atoi(key_vals["num_class"].c_str(), &num_class_);
  } else {
    Log::Fatal("Model file doesn't specify the number of classes");
    return false;
  }

  if (key_vals.count("num_tree_per_iteration")) {
    Common::Atoi(key_vals["num_tree_per_iteration"].c_str(), &num_tree_per_iteration_);
  } else {
    num_tree_per_iteration_ = num_class_;
  }

  // get index of label
  if (key_vals.count("label_index")) {
    Common::Atoi(key_vals["label_index"].c_str(), &label_idx_);
  } else {
    Log::Fatal("Model file doesn't specify the label index");
    return false;
  }

  // get max_feature_idx first
  if (key_vals.count("max_feature_idx")) {
    Common::Atoi(key_vals["max_feature_idx"].c_str(), &max_feature_idx_);
  } else {
    Log::Fatal("Model file doesn't specify max_feature_idx");
    return false;
  }

  // get average_output
  if (key_vals.count("average_output")) {
    average_output_ = true;
  }

  // get feature names
  if (key_vals.count("feature_names")) {
    feature_names_ = Common::Split(key_vals["feature_names"].c_str(), ' ');
    if (feature_names_.size() != static_cast<size_t>(max_feature_idx_ + 1)) {
      Log::Fatal("Wrong size of feature_names");
      return false;
    }
  } else {
    Log::Fatal("Model file doesn't contain feature names");
    return false;
  }

  if (key_vals.count("feature_infos")) {
    feature_infos_ = Common::Split(key_vals["feature_infos"].c_str(), ' ');
    if (feature_infos_.size() != static_cast<size_t>(max_feature_idx_ + 1)) {
      Log::Fatal("Wrong size of feature_infos");
      return false;
    }
  } else {
    Log::Fatal("Model file doesn't contain feature infos");
    return false;
  }

  if (key_vals.count("objective")) {
    auto str = key_vals["objective"];
    loaded_objective_.reset(ObjectiveFunction::CreateObjectiveFunction(str));
    objective_function_ = loaded_objective_.get();
  }

  while (*p != '\0') {
    auto line_len = Common::GetLine(p);
    std::string cur_line(p, line_len);
    if (Common::StartsWith(cur_line, "Tree=")) {
      p += line_len;
      p = Common::SkipNewLine(p);
      size_t used_len = 0;
      models_.emplace_back(new Tree(p, &used_len));
      p += used_len;
    } else {
      break;
    }
    p = Common::SkipNewLine(p);
  }

  Log::Info("Finished loading %d models", models_.size());
  num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_tree_per_iteration_;
  num_init_iteration_ = num_iteration_for_pred_;
  iter_ = 0;

  return true;
}

std::vector<double> GBDT::FeatureImportance(int num_iteration, int importance_type) const {

  int num_used_model = static_cast<int>(models_.size());
  if (num_iteration > 0) {
    num_iteration += 0;
    num_used_model = std::min(num_iteration * num_tree_per_iteration_, num_used_model);
  }

  std::vector<double> feature_importances(max_feature_idx_ + 1, 0.0);
  if (importance_type == 0) {
    for (int iter = 0; iter < num_used_model; ++iter) {
      for (int split_idx = 0; split_idx < models_[iter]->num_leaves() - 1; ++split_idx) {
        if (models_[iter]->split_gain(split_idx) > 0) {
          feature_importances[models_[iter]->split_feature(split_idx)] += 1.0;
        }
      }
    }
  } else if (importance_type == 1) {
    for (int iter = 0; iter < num_used_model; ++iter) {
      for (int split_idx = 0; split_idx < models_[iter]->num_leaves() - 1; ++split_idx) {
        if (models_[iter]->split_gain(split_idx) > 0) {
          feature_importances[models_[iter]->split_feature(split_idx)] += models_[iter]->split_gain(split_idx);
        }
      }
    }
  } else {
    Log::Fatal("Unknown importance type: only support split=0 and gain=1.");
  }
  return feature_importances;
}

}  // namespace LightGBM
