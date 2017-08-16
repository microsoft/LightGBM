#ifndef LIGHTGBM_TREE_H_
#define LIGHTGBM_TREE_H_

#include <LightGBM/meta.h>
#include <LightGBM/dataset.h>

#include <string>
#include <vector>
#include <memory>

namespace LightGBM {

#define kMaxTreeOutput (100)
#define kCategoricalMask (1)
#define kDefaultLeftMask (2)

/*!
* \brief Tree model
*/
class Tree {
public:
  /*!
  * \brief Constructor
  * \param max_leaves The number of max leaves
  */
  explicit Tree(int max_leaves);

  /*!
  * \brief Construtor, from a string
  * \param str Model string
  */
  explicit Tree(const std::string& str);

  ~Tree();

  /*!
  * \brief Performing a split on tree leaves.
  * \param leaf Index of leaf to be split
  * \param feature Index of feature; the converted index after removing useless features
  * \param real_feature Index of feature, the original index on data
  * \param threshold_bin Threshold(bin) of split
  * \param threshold_double Threshold on feature value
  * \param left_value Model Left child output
  * \param right_value Model Right child output
  * \param left_cnt Count of left child
  * \param right_cnt Count of right child
  * \param gain Split gain
  * \param missing_type missing type
  * \param default_left default direction for missing value
  * \return The index of new leaf.
  */
  int Split(int leaf, int feature, int real_feature, uint32_t threshold_bin,
            double threshold_double, double left_value, double right_value,
            data_size_t left_cnt, data_size_t right_cnt, double gain, MissingType missing_type, bool default_left);

  /*!
  * \brief Performing a split on tree leaves, with categorical feature
  * \param leaf Index of leaf to be split
  * \param feature Index of feature; the converted index after removing useless features
  * \param real_feature Index of feature, the original index on data
  * \param threshold_bin Threshold(bin) of split, use bitset to represent
  * \param num_threshold_bin size of threshold_bin
  * \param threshold Thresholds of real feature value, use bitset to represent
  * \param num_threshold size of threshold
  * \param left_value Model Left child output
  * \param right_value Model Right child output
  * \param left_cnt Count of left child
  * \param right_cnt Count of right child
  * \param gain Split gain
  * \return The index of new leaf.
  */
  int SplitCategorical(int leaf, int feature, int real_feature, const uint32_t* threshold_bin, int num_threshold_bin,
                       const uint32_t* threshold, int num_threshold, double left_value, double right_value,
                       data_size_t left_cnt, data_size_t right_cnt, double gain, MissingType missing_type);

  /*! \brief Get the output of one leaf */
  inline double LeafOutput(int leaf) const { return leaf_value_[leaf]; }

  /*! \brief Set the output of one leaf */
  inline void SetLeafOutput(int leaf, double output) {
    leaf_value_[leaf] = output;
  }

  /*!
  * \brief Adding prediction value of this tree model to scores
  * \param data The dataset
  * \param num_data Number of total data
  * \param score Will add prediction to score
  */
  void AddPredictionToScore(const Dataset* data,
                            data_size_t num_data,
                            double* score) const;

  /*!
  * \brief Adding prediction value of this tree model to scorese
  * \param data The dataset
  * \param used_data_indices Indices of used data
  * \param num_data Number of total data
  * \param score Will add prediction to score
  */
  void AddPredictionToScore(const Dataset* data,
                            const data_size_t* used_data_indices,
                            data_size_t num_data, double* score) const;

  /*!
  * \brief Prediction on one record
  * \param feature_values Feature value of this record
  * \return Prediction result
  */
  inline double Predict(const double* feature_values) const;

  inline int PredictLeafIndex(const double* feature_values) const;

  inline void PredictContrib(const double* feature_values, int num_features, double* output) const;

  inline double ExpectedValue(int node = 0) const;

  inline int MaxDepth() const;

  /*!
  * \brief Used by TreeSHAP for data we keep about our decision path
  */
  struct PathElement {
    int feature_index;
    double zero_fraction;
    double one_fraction;

    // note that pweight is included for convenience and is not tied with the other attributes,
    // the pweight of the i'th path element is the permuation weight of paths with i-1 ones in them
    double pweight;

    PathElement() {}
    PathElement(int i, double z, double o, double w) : feature_index(i), zero_fraction(z), one_fraction(o), pweight(w) {}
  };

  /*! \brief Polynomial time algorithm for SHAP values (https://arxiv.org/abs/1706.06060) */
  inline void TreeSHAP(const double *feature_values, double *phi,
                       int node, int unique_depth,
                       PathElement *parent_unique_path, double parent_zero_fraction,
                       double parent_one_fraction, int parent_feature_index) const;

  /*! \brief Get Number of leaves*/
  inline int num_leaves() const { return num_leaves_; }

  /*! \brief Get depth of specific leaf*/
  inline int leaf_depth(int leaf_idx) const { return leaf_depth_[leaf_idx]; }

  /*! \brief Get feature of specific split*/
  inline int split_feature(int split_idx) const { return split_feature_[split_idx]; }

  inline double split_gain(int split_idx) const { return split_gain_[split_idx]; }

  /*! \brief Get the number of data points that fall at or below this node*/
  inline int data_count(int node = 0) const { return node >= 0 ? internal_count_[node] : leaf_count_[~node]; }

  /*!
  * \brief Shrinkage for the tree's output
  *        shrinkage rate (a.k.a learning rate) is used to tune the traning process
  * \param rate The factor of shrinkage
  */
  inline void Shrinkage(double rate) {
    #pragma omp parallel for schedule(static, 1024) if (num_leaves_ >= 2048)
    for (int i = 0; i < num_leaves_; ++i) {
      leaf_value_[i] *= rate;
      if (leaf_value_[i] > kMaxTreeOutput) { leaf_value_[i] = kMaxTreeOutput; }
      else if (leaf_value_[i] < -kMaxTreeOutput) { leaf_value_[i] = -kMaxTreeOutput; }
    }
    shrinkage_ *= rate;
  }

  /*! \brief Serialize this object to string*/
  std::string ToString();

  /*! \brief Serialize this object to json*/
  std::string ToJSON();

  /*! \brief Serialize this object to if-else statement*/
  std::string ToIfElse(int index, bool is_predict_leaf_index);

  inline static bool IsZero(double fval) {
    if (fval > -kZeroAsMissingValueRange && fval <= kZeroAsMissingValueRange) {
      return true;
    } else {
      return false;
    }
  }

  inline static bool GetDecisionType(int8_t decision_type, int8_t mask) {
    return (decision_type & mask) > 0;
  }

  inline static void SetDecisionType(int8_t* decision_type, bool input, int8_t mask) {
    if (input) {
      (*decision_type) |= mask;
    } else {
      (*decision_type) &= (127 - mask);
    }
  }

  inline static int8_t GetMissingType(int8_t decision_type) {
    return (decision_type >> 2) & 3;
  }

  inline static void SetMissingType(int8_t* decision_type, int8_t input) {
    (*decision_type) &= 3;
    (*decision_type) |= (input << 2);
  }

private:

  inline std::string NumericalDecisionIfElse(int node) {
    std::stringstream str_buf;
    uint8_t missing_type = GetMissingType(decision_type_[node]);
    bool default_left = GetDecisionType(decision_type_[node], kDefaultLeftMask);
    if (missing_type == 0 || (missing_type == 1 && default_left && kZeroAsMissingValueRange < threshold_[node])) {
      str_buf << "if (fval <= " << threshold_[node] << ") {";
    } else if (missing_type == 1) {
      if (default_left) {
        str_buf << "if (fval <= " << threshold_[node] << " || Tree::IsZero(fval)" << " || std::isnan(fval)) {";
      } else {
        str_buf << "if (fval <= " << threshold_[node] << " && !Tree::IsZero(fval)" << " && !std::isnan(fval)) {";
      }
    } else {
      if (default_left) {
        str_buf << "if (fval <= " << threshold_[node] << " || std::isnan(fval)) {";
      } else {
        str_buf << "if (fval <= " << threshold_[node] << " && !std::isnan(fval)) {";
      }
    }
    return str_buf.str();
  }

  inline std::string CategoricalDecisionIfElse(int node) const {
    uint8_t missing_type = GetMissingType(decision_type_[node]);
    std::stringstream str_buf;
    if (missing_type == 2) {
      str_buf << "if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }";
    } else {
      str_buf << "if (std::isnan(fval)) { int_fval = 0; } else { int_fval = static_cast<int>(fval); }";
    }
    int cat_idx = int(threshold_[node]);
    str_buf << "if (int_fval >= 0 && int_fval < 32 * (";
    str_buf << cat_boundaries_[cat_idx + 1] - cat_boundaries_[cat_idx];
    str_buf << ") && (((cat_threshold[" << cat_boundaries_[cat_idx];
    str_buf << " + int_fval / 32] >> (int_fval & 31)) & 1))) {";
    return str_buf.str();
  }

  inline int NumericalDecision(double fval, int node) const {
    uint8_t missing_type = GetMissingType(decision_type_[node]);
    if (std::isnan(fval)) {
      if (missing_type != 2) {
        fval = 0.0f;
      }
    }
    if ((missing_type == 1 && IsZero(fval))
        || (missing_type == 2 && std::isnan(fval))) {
      if (GetDecisionType(decision_type_[node], kDefaultLeftMask)) {
        return left_child_[node];
      } else {
        return right_child_[node];
      }
    }
    if (fval <= threshold_[node]) {
      return left_child_[node];
    } else {
      return right_child_[node];
    }
  }

  inline int NumericalDecisionInner(uint32_t fval, int node, uint32_t default_bin, uint32_t max_bin) const {
    uint8_t missing_type = GetMissingType(decision_type_[node]);
    if ((missing_type == 1 && fval == default_bin)
        || (missing_type == 2 && fval == max_bin)) {
      if (GetDecisionType(decision_type_[node], kDefaultLeftMask)) {
        return left_child_[node];
      } else {
        return right_child_[node];
      }
    }
    if (fval <= threshold_in_bin_[node]) {
      return left_child_[node];
    } else {
      return right_child_[node];
    }
  }

  inline int CategoricalDecision(double fval, int node) const {
    uint8_t missing_type = GetMissingType(decision_type_[node]);
    int int_fval = static_cast<int>(fval);
    if (int_fval < 0) {
      return right_child_[node];;
    } else if (std::isnan(fval)) {
      // NaN is always in the right
      if (missing_type == 2) {
        return right_child_[node];
      }
      int_fval = 0;
    }
    int cat_idx = int(threshold_[node]);
    if (Common::FindInBitset(cat_threshold_.data() + cat_boundaries_[cat_idx],
                             cat_boundaries_[cat_idx + 1] - cat_boundaries_[cat_idx], int_fval)) {
      return left_child_[node];
    }
    return right_child_[node];
  }

  inline int CategoricalDecisionInner(uint32_t fval, int node) const {
    int cat_idx = int(threshold_in_bin_[node]);
    if (Common::FindInBitset(cat_threshold_inner_.data() + cat_boundaries_inner_[cat_idx],
                             cat_boundaries_inner_[cat_idx + 1] - cat_boundaries_inner_[cat_idx], fval)) {
      return left_child_[node];
    }
    return right_child_[node];
  }

  inline int Decision(double fval, int node) const {
    if (GetDecisionType(decision_type_[node], kCategoricalMask)) {
      return CategoricalDecision(fval, node);
    } else {
      return NumericalDecision(fval, node);
    }
  }

  inline int DecisionInner(uint32_t fval, int node, uint32_t default_bin, uint32_t max_bin) const {
    if (GetDecisionType(decision_type_[node], kCategoricalMask)) {
      return CategoricalDecisionInner(fval, node);
    } else {
      return NumericalDecisionInner(fval, node, default_bin, max_bin);
    }
  }

  inline void Split(int leaf, int feature, int real_feature,
                    double left_value, double right_value, data_size_t left_cnt, data_size_t right_cnt, double gain);
  /*!
  * \brief Find leaf index of which record belongs by features
  * \param feature_values Feature value of this record
  * \return Leaf index
  */
  inline int GetLeaf(const double* feature_values) const;

  /*! \brief Serialize one node to json*/
  inline std::string NodeToJSON(int index);

  /*! \brief Serialize one node to if-else statement*/
  inline std::string NodeToIfElse(int index, bool is_predict_leaf_index);

  /*! \brief Extend our decision path with a fraction of one and zero extensions for TreeSHAP*/
  inline static void ExtendPath(PathElement *unique_path, int unique_depth,
                                double zero_fraction, double one_fraction, int feature_index);

  /*! \brief Undo a previous extension of the decision path for TreeSHAP*/
  inline static void UnwindPath(PathElement *unique_path, int unique_depth, int path_index);

  /*! determine what the total permuation weight would be if we unwound a previous extension in the decision path*/
  inline static double UnwoundPathSum(const PathElement *unique_path, int unique_depth, int path_index);

  /*! \brief Number of max leaves*/
  int max_leaves_;
  /*! \brief Number of current levas*/
  int num_leaves_;
  // following values used for non-leaf node
  /*! \brief A non-leaf node's left child */
  std::vector<int> left_child_;
  /*! \brief A non-leaf node's right child */
  std::vector<int> right_child_;
  /*! \brief A non-leaf node's split feature */
  std::vector<int> split_feature_inner_;
  /*! \brief A non-leaf node's split feature, the original index */
  std::vector<int> split_feature_;
  /*! \brief A non-leaf node's split threshold in bin */
  std::vector<uint32_t> threshold_in_bin_;
  /*! \brief A non-leaf node's split threshold in feature value */
  std::vector<double> threshold_;
  int num_cat_;
  std::vector<int> cat_boundaries_inner_;
  std::vector<uint32_t> cat_threshold_inner_;
  std::vector<int> cat_boundaries_;
  std::vector<uint32_t> cat_threshold_;
  /*! \brief Store the information for categorical feature handle and mising value handle. */
  std::vector<int8_t> decision_type_;
  /*! \brief A non-leaf node's split gain */
  std::vector<double> split_gain_;
  // used for leaf node
  /*! \brief The parent of leaf */
  std::vector<int> leaf_parent_;
  /*! \brief Output of leaves */
  std::vector<double> leaf_value_;
  /*! \brief DataCount of leaves */
  std::vector<data_size_t> leaf_count_;
  /*! \brief Output of non-leaf nodes */
  std::vector<double> internal_value_;
  /*! \brief DataCount of non-leaf nodes */
  std::vector<data_size_t> internal_count_;
  /*! \brief Depth for leaves */
  std::vector<int> leaf_depth_;
  double shrinkage_;
};

inline void Tree::Split(int leaf, int feature, int real_feature,
                        double left_value, double right_value, data_size_t left_cnt, data_size_t right_cnt, double gain) {
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
  split_feature_inner_[new_node_idx] = feature;
  split_feature_[new_node_idx] = real_feature;

  split_gain_[new_node_idx] = Common::AvoidInf(gain);
  // add two new leaves
  left_child_[new_node_idx] = ~leaf;
  right_child_[new_node_idx] = ~num_leaves_;
  // update new leaves
  leaf_parent_[leaf] = new_node_idx;
  leaf_parent_[num_leaves_] = new_node_idx;
  // save current leaf value to internal node before change
  internal_value_[new_node_idx] = leaf_value_[leaf];
  internal_count_[new_node_idx] = left_cnt + right_cnt;
  leaf_value_[leaf] = std::isnan(left_value) ? 0.0f : left_value;
  leaf_count_[leaf] = left_cnt;
  leaf_value_[num_leaves_] = std::isnan(right_value) ? 0.0f : right_value;
  leaf_count_[num_leaves_] = right_cnt;
  // update leaf depth
  leaf_depth_[num_leaves_] = leaf_depth_[leaf] + 1;
  leaf_depth_[leaf]++;
}

inline double Tree::Predict(const double* feature_values) const {
  if (num_leaves_ > 1) {
    int leaf = GetLeaf(feature_values);
    return LeafOutput(leaf);
  } else {
    return 0.0f;
  }
}

inline int Tree::PredictLeafIndex(const double* feature_values) const {
  if (num_leaves_ > 1) {
    int leaf = GetLeaf(feature_values);
    return leaf;
  } else {
    return 0;
  }
}

inline void Tree::ExtendPath(PathElement *unique_path, int unique_depth,
                                    double zero_fraction, double one_fraction, int feature_index) {
  unique_path[unique_depth].feature_index = feature_index;
  unique_path[unique_depth].zero_fraction = zero_fraction;
  unique_path[unique_depth].one_fraction = one_fraction;
  unique_path[unique_depth].pweight = (unique_depth == 0 ? 1 : 0);
  for (int i = unique_depth-1; i >= 0; i--) {
    unique_path[i+1].pweight += one_fraction*unique_path[i].pweight*(i+1)
                                / static_cast<double>(unique_depth+1);
    unique_path[i].pweight = zero_fraction*unique_path[i].pweight*(unique_depth-i)
                             / static_cast<double>(unique_depth+1);
  }
}

inline void Tree::UnwindPath(PathElement *unique_path, int unique_depth, int path_index) {
  const double one_fraction = unique_path[path_index].one_fraction;
  const double zero_fraction = unique_path[path_index].zero_fraction;
  double next_one_portion = unique_path[unique_depth].pweight;

  for (int i = unique_depth-1; i >= 0; --i) {
    if (one_fraction != 0) {
      const double tmp = unique_path[i].pweight;
      unique_path[i].pweight = next_one_portion*(unique_depth+1)
                               / static_cast<double>((i+1)*one_fraction);
      next_one_portion = tmp - unique_path[i].pweight*zero_fraction*(unique_depth-i)
                               / static_cast<double>(unique_depth+1);
    } else {
      unique_path[i].pweight = (unique_path[i].pweight*(unique_depth+1))
                               / static_cast<double>(zero_fraction*(unique_depth-i));
    }
  }

  for (int i = path_index; i < unique_depth; ++i) {
    unique_path[i].feature_index = unique_path[i+1].feature_index;
    unique_path[i].zero_fraction = unique_path[i+1].zero_fraction;
    unique_path[i].one_fraction = unique_path[i+1].one_fraction;
  }
}

inline double Tree::UnwoundPathSum(const PathElement *unique_path, int unique_depth, int path_index) {
  const double one_fraction = unique_path[path_index].one_fraction;
  const double zero_fraction = unique_path[path_index].zero_fraction;
  double next_one_portion = unique_path[unique_depth].pweight;
  double total = 0;
  for (int i = unique_depth-1; i >= 0; --i) {
    if (one_fraction != 0) {
      const double tmp = next_one_portion*(unique_depth+1)
                            / static_cast<double>((i+1)*one_fraction);
      total += tmp;
      next_one_portion = unique_path[i].pweight - tmp*zero_fraction*((unique_depth-i)
                         / static_cast<double>(unique_depth+1));
    } else {
      total += (unique_path[i].pweight/zero_fraction)/((unique_depth-i)
               / static_cast<double>(unique_depth+1));
    }
  }
  return total;
}

// recursive computation of SHAP values for a decision tree
inline void Tree::TreeSHAP(const double *feature_values, double *phi,
                           int node, int unique_depth,
                           PathElement *parent_unique_path, double parent_zero_fraction,
                           double parent_one_fraction, int parent_feature_index) const {

  // extend the unique path
  PathElement *unique_path = parent_unique_path + unique_depth;
  if (unique_depth > 0) std::copy(parent_unique_path, parent_unique_path+unique_depth, unique_path);
  ExtendPath(unique_path, unique_depth, parent_zero_fraction,
             parent_one_fraction, parent_feature_index);
  const int split_index = split_feature_[node];

  // leaf node
  if (node < 0) {
    for (int i = 1; i <= unique_depth; ++i) {
      const double w = UnwoundPathSum(unique_path, unique_depth, i);
      const PathElement &el = unique_path[i];
      phi[el.feature_index] += w*(el.one_fraction-el.zero_fraction)*leaf_value_[~node];
    }

  // internal node
  } else {
    const int hot_index = Decision(feature_values[split_index], node);
    const int cold_index = (hot_index == left_child_[node] ? right_child_[node] : left_child_[node]);
    const double w = data_count(node);
    const double hot_zero_fraction = data_count(hot_index)/w;
    const double cold_zero_fraction = data_count(cold_index)/w;
    double incoming_zero_fraction = 1;
    double incoming_one_fraction = 1;

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    int path_index = 0;
    for (; path_index <= unique_depth; ++path_index) {
      if (unique_path[path_index].feature_index == split_index) break;
    }
    if (path_index != unique_depth+1) {
      incoming_zero_fraction = unique_path[path_index].zero_fraction;
      incoming_one_fraction = unique_path[path_index].one_fraction;
      UnwindPath(unique_path, unique_depth, path_index);
      unique_depth -= 1;
    }

    TreeSHAP(feature_values, phi, hot_index, unique_depth+1, unique_path,
             hot_zero_fraction*incoming_zero_fraction, incoming_one_fraction, split_index);

    TreeSHAP(feature_values, phi, cold_index, unique_depth+1, unique_path,
             cold_zero_fraction*incoming_zero_fraction, 0, split_index);
  }
}

inline void Tree::PredictContrib(const double* feature_values, int num_features, double *output) const {
  output[num_features] += ExpectedValue();

  // Run the recursion with preallocated space for the unique path data
  const int max_path_len = MaxDepth()+1;
  PathElement *unique_path_data = new PathElement[(max_path_len*(max_path_len+1))/2];
  TreeSHAP(feature_values, output, 0, 0, unique_path_data, 1, 1, -1);
  delete[] unique_path_data;
}

inline double Tree::ExpectedValue(int node) const {
  if (node >= 0) {
    const int l = left_child_[node];
    const int r = right_child_[node];
    return (data_count(l)*ExpectedValue(l) + data_count(r)*ExpectedValue(r))/data_count(node);
  } else {
    return LeafOutput(~node);
  }
}

inline int Tree::MaxDepth() const {
  int max_depth = 0;
  for (int i = 0; i < num_leaves(); ++i) {
    if (max_depth < leaf_depth_[i]) max_depth = leaf_depth_[i];
  }
  return max_depth;
}

inline int Tree::GetLeaf(const double* feature_values) const {
  int node = 0;
  if (num_cat_ > 0) {
    while (node >= 0) {
      node = Decision(feature_values[split_feature_[node]], node);
    }
  } else {
    while (node >= 0) {
      node = NumericalDecision(feature_values[split_feature_[node]], node);
    }
  }
  return ~node;
}

}  // namespace LightGBM

#endif   // LightGBM_TREE_H_
