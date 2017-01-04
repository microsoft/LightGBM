#include <LightGBM/tree.h>

#include <LightGBM/utils/threading.h>
#include <LightGBM/utils/common.h>

#include <LightGBM/dataset.h>
#include <LightGBM/feature.h>

#include <sstream>
#include <unordered_map>
#include <functional>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>

namespace LightGBM {

std::vector<std::function<bool(unsigned int, unsigned int)>> Tree::inner_decision_funs = 
          {Tree::NumericalDecision<unsigned int>, Tree::CategoricalDecision<unsigned int> };
std::vector<std::function<bool(double, double)>> Tree::decision_funs = 
          { Tree::NumericalDecision<double>, Tree::CategoricalDecision<double> };


Tree::Tree(int max_leaves)
  :max_leaves_(max_leaves) {

  num_leaves_ = 0;
  left_child_ = std::vector<int>(max_leaves_ - 1);
  right_child_ = std::vector<int>(max_leaves_ - 1);
  split_feature_ = std::vector<int>(max_leaves_ - 1);
  split_feature_real_ = std::vector<int>(max_leaves_ - 1);
  threshold_in_bin_ = std::vector<unsigned int>(max_leaves_ - 1);
  threshold_ = std::vector<double>(max_leaves_ - 1);
  decision_type_ = std::vector<int8_t>(max_leaves_ - 1);
  split_gain_ = std::vector<double>(max_leaves_ - 1);
  leaf_parent_ = std::vector<int>(max_leaves_);
  leaf_value_ = std::vector<double>(max_leaves_);
  leaf_count_ = std::vector<data_size_t>(max_leaves_);
  internal_value_ = std::vector<double>(max_leaves_ - 1);
  internal_count_ = std::vector<data_size_t>(max_leaves_ - 1);
  leaf_depth_ = std::vector<int>(max_leaves_);
  // root is in the depth 0
  leaf_depth_[0] = 0;
  num_leaves_ = 1;
  leaf_parent_[0] = -1;
}
Tree::~Tree() {

}

int Tree::Split(int leaf, int feature, BinType bin_type, unsigned int threshold_bin, int real_feature,
    double threshold_double, double left_value,
    double right_value, data_size_t left_cnt, data_size_t right_cnt, double gain) {
  int new_node_idx = num_leaves_ - 1;
  // update parent info
  int parent = leaf_parent_[leaf];
  if (parent >= 0) {
    // if cur node is left child
    if (left_child_[parent] == ~leaf) {
      left_child_[parent] = new_node_idx;
    } else {
      right_child_[parent] = new_node_idx;
    }
  }
  // add new node
  split_feature_[new_node_idx] = feature;
  split_feature_real_[new_node_idx] = real_feature;
  threshold_in_bin_[new_node_idx] = threshold_bin;
  threshold_[new_node_idx] = threshold_double;
  if (bin_type == BinType::NumericalBin) {
    decision_type_[new_node_idx] = 0;
  } else {
    decision_type_[new_node_idx] = 1;
  }
  split_gain_[new_node_idx] = gain;
  // add two new leaves
  left_child_[new_node_idx] = ~leaf;
  right_child_[new_node_idx] = ~num_leaves_;
  // update new leaves
  leaf_parent_[leaf] = new_node_idx;
  leaf_parent_[num_leaves_] = new_node_idx;
  // save current leaf value to internal node before change
  internal_value_[new_node_idx] = leaf_value_[leaf];
  internal_count_[new_node_idx] = left_cnt + right_cnt;
  leaf_value_[leaf] = left_value;
  leaf_count_[leaf] = left_cnt;
  leaf_value_[num_leaves_] = right_value;
  leaf_count_[num_leaves_] = right_cnt;
  // update leaf depth
  leaf_depth_[num_leaves_] = leaf_depth_[leaf] + 1;
  leaf_depth_[leaf]++;

  ++num_leaves_;
  return num_leaves_ - 1;
}

void Tree::AddPredictionToScore(const Dataset* data, data_size_t num_data, score_t* score) const {
  Threading::For<data_size_t>(0, num_data, [this, data, score](int, data_size_t start, data_size_t end) {
    std::vector<std::unique_ptr<BinIterator>> iterators(data->num_features());
    for (int i = 0; i < data->num_features(); ++i) {
      iterators[i].reset(data->FeatureAt(i)->bin_data()->GetIterator(start));
    }
    for (data_size_t i = start; i < end; ++i) {
      score[i] += static_cast<score_t>(leaf_value_[GetLeaf(iterators, i)]);
    }
  });
}

void Tree::AddPredictionToScore(const Dataset* data, const data_size_t* used_data_indices,
                                             data_size_t num_data, score_t* score) const {
  Threading::For<data_size_t>(0, num_data,
      [this, data, used_data_indices, score](int, data_size_t start, data_size_t end) {
    std::vector<std::unique_ptr<BinIterator>> iterators(data->num_features());
    for (int i = 0; i < data->num_features(); ++i) {
      iterators[i].reset(data->FeatureAt(i)->bin_data()->GetIterator(used_data_indices[start]));
    }
    for (data_size_t i = start; i < end; ++i) {
      score[used_data_indices[i]] += static_cast<score_t>(leaf_value_[GetLeaf(iterators, used_data_indices[i])]);
    }
  });
}

std::string Tree::ToString() {
  std::stringstream str_buf;
  str_buf << "num_leaves=" << num_leaves_ << std::endl;
  str_buf << "split_feature="
    << Common::ArrayToString<int>(split_feature_real_, num_leaves_ - 1, ' ') << std::endl;
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
  str_buf << std::endl;
  return str_buf.str();
}

std::string Tree::ToJSON() {
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << "\"num_leaves\":" << num_leaves_ << "," << std::endl;

  str_buf << "\"tree_structure\":" << NodeToJSON(0) << std::endl;

  return str_buf.str();
}

std::string Tree::NodeToJSON(int index) {
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  if (index >= 0) {
    // non-leaf
    str_buf << "{" << std::endl;
    str_buf << "\"split_index\":" << index << "," << std::endl;
    str_buf << "\"split_feature\":" << split_feature_real_[index] << "," << std::endl;
    str_buf << "\"split_gain\":" << split_gain_[index] << "," << std::endl;
    str_buf << "\"threshold\":" << threshold_[index] << "," << std::endl;
    str_buf << "\"decision_type\":\"" << Tree::GetDecisionTypeName(decision_type_[index]) << "\"," << std::endl;
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

Tree::Tree(const std::string& str) {
  std::vector<std::string> lines = Common::Split(str.c_str(), '\n');
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
  if (key_vals.count("num_leaves") <= 0 || key_vals.count("split_feature") <= 0
    || key_vals.count("split_gain") <= 0 || key_vals.count("threshold") <= 0
    || key_vals.count("left_child") <= 0 || key_vals.count("right_child") <= 0
    || key_vals.count("leaf_parent") <= 0 || key_vals.count("leaf_value") <= 0
    || key_vals.count("internal_value") <= 0 || key_vals.count("internal_count") <= 0
    || key_vals.count("leaf_count") <= 0 || key_vals.count("decision_type") <= 0
    ) {
    Log::Fatal("Tree model string format error");
  }

  Common::Atoi(key_vals["num_leaves"].c_str(), &num_leaves_);

  left_child_ = Common::StringToArray<int>(key_vals["left_child"], ' ', num_leaves_ - 1);
  right_child_ = Common::StringToArray<int>(key_vals["right_child"], ' ', num_leaves_ - 1);
  split_feature_real_ = Common::StringToArray<int>(key_vals["split_feature"], ' ', num_leaves_ - 1);
  threshold_ = Common::StringToArray<double>(key_vals["threshold"], ' ', num_leaves_ - 1);
  split_gain_ = Common::StringToArray<double>(key_vals["split_gain"], ' ', num_leaves_ - 1);
  internal_count_ = Common::StringToArray<data_size_t>(key_vals["internal_count"], ' ', num_leaves_ - 1);
  internal_value_ = Common::StringToArray<double>(key_vals["internal_value"], ' ', num_leaves_ - 1);
  decision_type_ = Common::StringToArray<int8_t>(key_vals["decision_type"], ' ', num_leaves_ - 1);

  leaf_count_ = Common::StringToArray<data_size_t>(key_vals["leaf_count"], ' ', num_leaves_);
  leaf_parent_ = Common::StringToArray<int>(key_vals["leaf_parent"], ' ', num_leaves_);
  leaf_value_ = Common::StringToArray<double>(key_vals["leaf_value"], ' ', num_leaves_);

}

}  // namespace LightGBM
