#include <LightGBM/tree.h>

#include <LightGBM/utils/threading.h>
#include <LightGBM/utils/common.h>

#include <LightGBM/dataset.h>

#include <sstream>
#include <unordered_map>
#include <functional>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>

namespace LightGBM {

Tree::Tree(int max_leaves)
  :max_leaves_(max_leaves) {

  num_leaves_ = 0;
  left_child_.resize(max_leaves_ - 1);
  right_child_.resize(max_leaves_ - 1);
  split_feature_inner_.resize(max_leaves_ - 1);
  split_feature_.resize(max_leaves_ - 1);
  threshold_in_bin_.resize(max_leaves_ - 1);
  threshold_.resize(max_leaves_ - 1);
  decision_type_.resize(max_leaves_ - 1, 0);
  split_gain_.resize(max_leaves_ - 1);
  leaf_parent_.resize(max_leaves_);
  leaf_value_.resize(max_leaves_);
  leaf_count_.resize(max_leaves_);
  internal_value_.resize(max_leaves_ - 1);
  internal_count_.resize(max_leaves_ - 1);
  leaf_depth_.resize(max_leaves_);
  // root is in the depth 0
  leaf_depth_[0] = 0;
  num_leaves_ = 1;
  leaf_parent_[0] = -1;
  shrinkage_ = 1.0f;
  num_cat_ = 0;
  cat_boundaries_.push_back(0);
  cat_boundaries_inner_.push_back(0);
}

Tree::~Tree() {

}

int Tree::Split(int leaf, int feature, int real_feature, uint32_t threshold_bin,
                double threshold_double, double left_value, double right_value,
                data_size_t left_cnt, data_size_t right_cnt, double gain, MissingType missing_type, bool default_left) {
  Split(leaf, feature, real_feature, left_value, right_value, left_cnt, right_cnt, gain);
  int new_node_idx = num_leaves_ - 1;
  decision_type_[new_node_idx] = 0;
  SetDecisionType(&decision_type_[new_node_idx], false, kCategoricalMask);
  SetDecisionType(&decision_type_[new_node_idx], default_left, kDefaultLeftMask);
  if (missing_type == MissingType::None) {
    SetMissingType(&decision_type_[new_node_idx], 0);
  } else if (missing_type == MissingType::Zero) {
    SetMissingType(&decision_type_[new_node_idx], 1);
  } else if (missing_type == MissingType::NaN) {
    SetMissingType(&decision_type_[new_node_idx], 2);
  }
  threshold_in_bin_[new_node_idx] = threshold_bin;
  threshold_[new_node_idx] = Common::AvoidInf(threshold_double);
  ++num_leaves_;
  return num_leaves_ - 1;
}

int Tree::SplitCategorical(int leaf, int feature, int real_feature, const uint32_t* threshold_bin, int num_threshold_bin,
                           const uint32_t* threshold, int num_threshold, double left_value, double right_value,
                           data_size_t left_cnt, data_size_t right_cnt, double gain, MissingType missing_type) {
  Split(leaf, feature, real_feature, left_value, right_value, left_cnt, right_cnt, gain);
  int new_node_idx = num_leaves_ - 1;
  decision_type_[new_node_idx] = 0;
  SetDecisionType(&decision_type_[new_node_idx], true, kCategoricalMask);
  if (missing_type == MissingType::None) {
    SetMissingType(&decision_type_[new_node_idx], 0);
  } else if (missing_type == MissingType::Zero) {
    SetMissingType(&decision_type_[new_node_idx], 1);
  } else if (missing_type == MissingType::NaN) {
    SetMissingType(&decision_type_[new_node_idx], 2);
  }
  threshold_in_bin_[new_node_idx] = num_cat_;
  threshold_[new_node_idx] = num_cat_;
  ++num_cat_;
  cat_boundaries_.push_back(cat_boundaries_.back() + num_threshold);
  for (int i = 0; i < num_threshold; ++i) {
    cat_threshold_.push_back(threshold[i]);
  }
  cat_boundaries_inner_.push_back(cat_boundaries_inner_.back() + num_threshold_bin);
  for (int i = 0; i < num_threshold_bin; ++i) {
    cat_threshold_inner_.push_back(threshold_bin[i]);
  }
  ++num_leaves_;
  return num_leaves_ - 1;
}

#define PredictionFun(niter, fidx_in_iter, start_pos, decision_fun, iter_idx, data_idx) \
std::vector<std::unique_ptr<BinIterator>> iter((niter)); \
for (int i = 0; i < (niter); ++i) { \
  iter[i].reset(data->FeatureIterator((fidx_in_iter))); \
  iter[i]->Reset((start_pos)); \
}\
for (data_size_t i = start; i < end; ++i) {\
  int node = 0;\
  while (node >= 0) {\
    node = decision_fun(iter[(iter_idx)]->Get((data_idx)), node, default_bins[node], max_bins[node]);\
  }\
  score[(data_idx)] += static_cast<double>(leaf_value_[~node]);\
}\

void Tree::AddPredictionToScore(const Dataset* data, data_size_t num_data, double* score) const {
  if (num_leaves_ <= 1) { return; }
  std::vector<uint32_t> default_bins(num_leaves_ - 1);
  std::vector<uint32_t> max_bins(num_leaves_ - 1);
  for (int i = 0; i < num_leaves_ - 1; ++i) {
    const int fidx = split_feature_inner_[i];
    auto bin_mapper = data->FeatureBinMapper(fidx);
    default_bins[i] = bin_mapper->GetDefaultBin();
    max_bins[i] = bin_mapper->num_bin() - 1;
  }
  if (num_cat_ > 0) {
    if (data->num_features() > num_leaves_ - 1) {
      Threading::For<data_size_t>(0, num_data, [this, &data, score, &default_bins, &max_bins]
      (int, data_size_t start, data_size_t end) {
        PredictionFun(num_leaves_ - 1, split_feature_inner_[i], start, DecisionInner, node, i);
      });
    } else {
      Threading::For<data_size_t>(0, num_data, [this, &data, score, &default_bins, &max_bins]
      (int, data_size_t start, data_size_t end) {
        PredictionFun(data->num_features(), i, start, DecisionInner, split_feature_inner_[node], i);
      });
    }
  } else {
    if (data->num_features() > num_leaves_ - 1) {
      Threading::For<data_size_t>(0, num_data, [this, &data, score, &default_bins, &max_bins]
      (int, data_size_t start, data_size_t end) {
        PredictionFun(num_leaves_ - 1, split_feature_inner_[i], start, NumericalDecisionInner, node, i);
      });
    } else {
      Threading::For<data_size_t>(0, num_data, [this, &data, score, &default_bins, &max_bins]
      (int, data_size_t start, data_size_t end) {
        PredictionFun(data->num_features(), i, start, NumericalDecisionInner, split_feature_inner_[node], i);
      });
    }
  }
}

void Tree::AddPredictionToScore(const Dataset* data,
  const data_size_t* used_data_indices,
  data_size_t num_data, double* score) const {
  if (num_leaves_ <= 1) { return; }
  std::vector<uint32_t> default_bins(num_leaves_ - 1);
  std::vector<uint32_t> max_bins(num_leaves_ - 1);
  for (int i = 0; i < num_leaves_ - 1; ++i) {
    const int fidx = split_feature_inner_[i];
    auto bin_mapper = data->FeatureBinMapper(fidx);
    default_bins[i] = bin_mapper->GetDefaultBin();
    max_bins[i] = bin_mapper->num_bin() - 1;
  }
  if (num_cat_ > 0) {
    if (data->num_features() > num_leaves_ - 1) {
      Threading::For<data_size_t>(0, num_data, [this, &data, score, used_data_indices, &default_bins, &max_bins]
      (int, data_size_t start, data_size_t end) {
        PredictionFun(num_leaves_ - 1, split_feature_inner_[i], used_data_indices[start], DecisionInner, node, used_data_indices[i]);
      });
    } else {
      Threading::For<data_size_t>(0, num_data, [this, &data, score, used_data_indices, &default_bins, &max_bins]
      (int, data_size_t start, data_size_t end) {
        PredictionFun(data->num_features(), i, used_data_indices[start], DecisionInner, split_feature_inner_[node], used_data_indices[i]);
      });
    }
  } else {
    if (data->num_features() > num_leaves_ - 1) {
      Threading::For<data_size_t>(0, num_data, [this, &data, score, used_data_indices, &default_bins, &max_bins]
      (int, data_size_t start, data_size_t end) {
        PredictionFun(num_leaves_ - 1, split_feature_inner_[i], used_data_indices[start], NumericalDecisionInner, node, used_data_indices[i]);
      });
    } else {
      Threading::For<data_size_t>(0, num_data, [this, &data, score, used_data_indices, &default_bins, &max_bins]
      (int, data_size_t start, data_size_t end) {
        PredictionFun(data->num_features(), i, used_data_indices[start], NumericalDecisionInner, split_feature_inner_[node], used_data_indices[i]);
      });
    }
  }
}

#undef PredictionFun

std::string Tree::ToString() {
  std::stringstream str_buf;
  str_buf << "num_leaves=" << num_leaves_ << std::endl;
  str_buf << "num_cat=" << num_cat_ << std::endl;
  str_buf << "split_feature="
    << Common::ArrayToString<int>(split_feature_, num_leaves_ - 1, ' ') << std::endl;
  str_buf << "split_gain="
    << Common::ArrayToString<double>(split_gain_, num_leaves_ - 1, ' ') << std::endl;
  str_buf << "threshold="
    << Common::ArrayToString<double>(threshold_, num_leaves_ - 1, ' ') << std::endl;
  str_buf << "decision_type="
    << Common::ArrayToString<int>(Common::ArrayCast<int8_t, int>(decision_type_), num_leaves_ - 1, ' ') << std::endl;
  str_buf << "left_child="
    << Common::ArrayToString<int>(left_child_, num_leaves_ - 1, ' ') << std::endl;
  str_buf << "right_child="
    << Common::ArrayToString<int>(right_child_, num_leaves_ - 1, ' ') << std::endl;
  str_buf << "leaf_parent="
    << Common::ArrayToString<int>(leaf_parent_, num_leaves_, ' ') << std::endl;
  str_buf << "leaf_value="
    << Common::ArrayToString<double>(leaf_value_, num_leaves_, ' ') << std::endl;
  str_buf << "leaf_count="
    << Common::ArrayToString<data_size_t>(leaf_count_, num_leaves_, ' ') << std::endl;
  str_buf << "internal_value="
    << Common::ArrayToString<double>(internal_value_, num_leaves_ - 1, ' ') << std::endl;
  str_buf << "internal_count="
    << Common::ArrayToString<data_size_t>(internal_count_, num_leaves_ - 1, ' ') << std::endl;
  if (num_cat_ > 0) {
    str_buf << "cat_boundaries="
      << Common::ArrayToString<int>(cat_boundaries_, num_cat_ + 1, ' ') << std::endl;
    str_buf << "cat_threshold="
      << Common::ArrayToString<uint32_t>(cat_threshold_, cat_threshold_.size(), ' ') << std::endl;
  }
  str_buf << "shrinkage=" << shrinkage_ << std::endl;
  str_buf << std::endl;
  return str_buf.str();
}

std::string Tree::ToJSON() {
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << "\"num_leaves\":" << num_leaves_ << "," << std::endl;
  str_buf << "\"num_cat\":" << num_cat_ << "," << std::endl;
  str_buf << "\"shrinkage\":" << shrinkage_ << "," << std::endl;
  if (num_leaves_ == 1) {
    str_buf << "\"tree_structure\":" << NodeToJSON(-1) << std::endl;
  } else {
    str_buf << "\"tree_structure\":" << NodeToJSON(0) << std::endl;
  }

  return str_buf.str();
}

std::string Tree::NodeToJSON(int index) {
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  if (index >= 0) {
    // non-leaf
    str_buf << "{" << std::endl;
    str_buf << "\"split_index\":" << index << "," << std::endl;
    str_buf << "\"split_feature\":" << split_feature_[index] << "," << std::endl;
    str_buf << "\"split_gain\":" << split_gain_[index] << "," << std::endl;
    if (GetDecisionType(decision_type_[index], kCategoricalMask)) {
      int cat_idx = static_cast<int>(threshold_[index]);
      std::vector<int> cats;
      for (int i = cat_boundaries_[cat_idx]; i < cat_boundaries_[cat_idx + 1]; ++i) {
        for (int j = 0; j < 32; ++j) {
          int cat = (i - cat_boundaries_[cat_idx]) * 32 + j;
          if (Common::FindInBitset(cat_threshold_.data() + cat_boundaries_[cat_idx],
                                   cat_boundaries_[cat_idx + 1] - cat_boundaries_[cat_idx], cat)) {
            cats.push_back(cat);
          }
        }
      }
      str_buf << "\"threshold\":\"" << Common::Join(cats, "||") << "\"," << std::endl;
      str_buf << "\"decision_type\":\"==\"," << std::endl;
    } else {
      str_buf << "\"threshold\":" << Common::AvoidInf(threshold_[index]) << "," << std::endl;
      str_buf << "\"decision_type\":\"<=\"," << std::endl;
    }
    if (GetDecisionType(decision_type_[index], kDefaultLeftMask)) {
      str_buf << "\"default_left\":true," << std::endl;
    } else {
      str_buf << "\"default_left\":false," << std::endl;
    }
    uint8_t missing_type = GetMissingType(decision_type_[index]);
    if (missing_type == 0) {
      str_buf << "\"missing_type\":\"None\"," << std::endl;
    } else if (missing_type == 1) {
      str_buf << "\"missing_type\":\"Zero\"," << std::endl;
    } else {
      str_buf << "\"missing_type\":\"NaN\"," << std::endl;
    }
    str_buf << "\"internal_value\":" << internal_value_[index] << "," << std::endl;
    str_buf << "\"internal_count\":" << internal_count_[index] << "," << std::endl;
    str_buf << "\"left_child\":" << NodeToJSON(left_child_[index]) << "," << std::endl;
    str_buf << "\"right_child\":" << NodeToJSON(right_child_[index]) << std::endl;
    str_buf << "}";
  } else {
    // leaf
    index = ~index;
    str_buf << "{" << std::endl;
    str_buf << "\"leaf_index\":" << index << "," << std::endl;
    str_buf << "\"leaf_parent\":" << leaf_parent_[index] << "," << std::endl;
    str_buf << "\"leaf_value\":" << leaf_value_[index] << "," << std::endl;
    str_buf << "\"leaf_count\":" << leaf_count_[index] << std::endl;
    str_buf << "}";
  }

  return str_buf.str();
}

std::string Tree::ToIfElse(int index, bool is_predict_leaf_index) {
  std::stringstream str_buf;
  str_buf << "double PredictTree" << index;
  if (is_predict_leaf_index) {
    str_buf << "Leaf";
  }
  str_buf << "(const double* arr) { ";
  if (num_leaves_ == 1) {
    str_buf << "return 0";
  } else {
    str_buf << "const std::vector<uint32_t> cat_threshold = {";
    for (size_t i = 0; i < cat_threshold_.size(); ++i) {
      if (i != 0) {
        str_buf << ",";
      }
      str_buf << cat_threshold_[i];
    }
    str_buf << "};";
    // use this for the missing value conversion
    str_buf << "double fval = 0.0f; ";
    if (num_cat_ > 0) {
      str_buf << "int int_fval = 0; ";
    }
    str_buf << NodeToIfElse(0, is_predict_leaf_index);
  }
  str_buf << " }" << std::endl;
  return str_buf.str();
}

std::string Tree::NodeToIfElse(int index, bool is_predict_leaf_index) {
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  if (index >= 0) {
    // non-leaf
    str_buf << "fval = arr[" << split_feature_[index] << "];";
    if (GetDecisionType(decision_type_[index], kCategoricalMask) == 0) {
      str_buf << NumericalDecisionIfElse(index);
    } else {
      str_buf << CategoricalDecisionIfElse(index);
    }
    // left subtree
    str_buf << NodeToIfElse(left_child_[index], is_predict_leaf_index);
    str_buf << " } else { ";
    // right subtree
    str_buf << NodeToIfElse(right_child_[index], is_predict_leaf_index);
    str_buf << " }";
  } else {
    // leaf
    str_buf << "return ";
    if (is_predict_leaf_index) {
      str_buf << ~index;
    } else {
      str_buf << leaf_value_[~index];
    }
    str_buf << ";";
  }

  return str_buf.str();
}

Tree::Tree(const std::string& str) {
  std::vector<std::string> lines = Common::SplitLines(str.c_str());
  std::unordered_map<std::string, std::string> key_vals;
  for (const std::string& line : lines) {
    std::vector<std::string> tmp_strs = Common::Split(line.c_str(), '=');
    if (tmp_strs.size() == 2) {
      std::string key = Common::Trim(tmp_strs[0]);
      std::string val = Common::Trim(tmp_strs[1]);
      if (key.size() > 0 && val.size() > 0) {
        key_vals[key] = val;
      }
    }
  }
  if (key_vals.count("num_leaves") <= 0) {
    Log::Fatal("Tree model should contain num_leaves field.");
  }

  Common::Atoi(key_vals["num_leaves"].c_str(), &num_leaves_);

  if (key_vals.count("num_cat") <= 0) {
    Log::Fatal("Tree model should contain num_cat field.");
  }

  Common::Atoi(key_vals["num_cat"].c_str(), &num_cat_);

  if (num_leaves_ <= 1) { return; }

  if (key_vals.count("left_child")) {
    left_child_ = Common::StringToArray<int>(key_vals["left_child"], ' ', num_leaves_ - 1);
  } else {
    Log::Fatal("Tree model string format error, should contain left_child field");
  }

  if (key_vals.count("right_child")) {
    right_child_ = Common::StringToArray<int>(key_vals["right_child"], ' ', num_leaves_ - 1);
  } else {
    Log::Fatal("Tree model string format error, should contain right_child field");
  }

  if (key_vals.count("split_feature")) {
    split_feature_ = Common::StringToArray<int>(key_vals["split_feature"], ' ', num_leaves_ - 1);
  } else {
    Log::Fatal("Tree model string format error, should contain split_feature field");
  }

  if (key_vals.count("threshold")) {
    threshold_ = Common::StringToArray<double>(key_vals["threshold"], ' ', num_leaves_ - 1);
  } else {
    Log::Fatal("Tree model string format error, should contain threshold field");
  }

  if (key_vals.count("leaf_value")) {
    leaf_value_ = Common::StringToArray<double>(key_vals["leaf_value"], ' ', num_leaves_);
  } else {
    Log::Fatal("Tree model string format error, should contain leaf_value field");
  }

  if (key_vals.count("split_gain")) {
    split_gain_ = Common::StringToArray<double>(key_vals["split_gain"], ' ', num_leaves_ - 1);
  } else {
    split_gain_.resize(num_leaves_ - 1);
  }

  if (key_vals.count("internal_count")) {
    internal_count_ = Common::StringToArray<data_size_t>(key_vals["internal_count"], ' ', num_leaves_ - 1);
  } else {
    internal_count_.resize(num_leaves_ - 1);
  }

  if (key_vals.count("internal_value")) {
    internal_value_ = Common::StringToArray<double>(key_vals["internal_value"], ' ', num_leaves_ - 1);
  } else {
    internal_value_.resize(num_leaves_ - 1);
  }

  if (key_vals.count("leaf_count")) {
    leaf_count_ = Common::StringToArray<data_size_t>(key_vals["leaf_count"], ' ', num_leaves_);
  } else {
    leaf_count_.resize(num_leaves_);
  }

  if (key_vals.count("leaf_parent")) {
    leaf_parent_ = Common::StringToArray<int>(key_vals["leaf_parent"], ' ', num_leaves_);
  } else {
    leaf_parent_.resize(num_leaves_);
  }

  if (key_vals.count("decision_type")) {
    decision_type_ = Common::StringToArray<int8_t>(key_vals["decision_type"], ' ', num_leaves_ - 1);
  } else {
    decision_type_ = std::vector<int8_t>(num_leaves_ - 1, 0);
  }

  if (num_cat_ > 0) {
    if (key_vals.count("cat_boundaries")) {
      cat_boundaries_ = Common::StringToArray<int>(key_vals["cat_boundaries"], ' ', num_cat_ + 1);
    } else {
      Log::Fatal("Tree model should contain cat_boundaries field.");
    }

    if (key_vals.count("cat_threshold")) {
      cat_threshold_ = Common::StringToArray<uint32_t>(key_vals["cat_threshold"], ' ', cat_boundaries_.back());
    } else {
      Log::Fatal("Tree model should contain cat_threshold field.");
    }
  }

  if (key_vals.count("shrinkage")) {
    Common::Atof(key_vals["shrinkage"].c_str(), &shrinkage_);
  } else {
    shrinkage_ = 1.0f;
  }
}

}  // namespace LightGBM
