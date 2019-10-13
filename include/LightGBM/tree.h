/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREE_H_
#define LIGHTGBM_TREE_H_

#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>

#include <string>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace LightGBM {

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
  * \brief Constructor, from a string
  * \param str Model string
  * \param used_len used count of str
  */
  Tree(const char* str, size_t* used_len);

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
  * \param left_weight Weight of left child
  * \param right_weight Weight of right child
  * \param gain Split gain
  * \param missing_type missing type
  * \param default_left default direction for missing value
  * \return The index of new leaf.
  */
  int Split(int leaf, int feature, int real_feature, uint32_t threshold_bin,
            double threshold_double, double left_value, double right_value,
            int left_cnt, int right_cnt, double left_weight, double right_weight,
            float gain, MissingType missing_type, bool default_left);

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
  * \param left_weight Weight of left child
  * \param right_weight Weight of right child
  * \param gain Split gain
  * \return The index of new leaf.
  */
  int SplitCategorical(int leaf, int feature, int real_feature, const uint32_t* threshold_bin, int num_threshold_bin,
                       const uint32_t* threshold, int num_threshold, double left_value, double right_value,
                       int left_cnt, int right_cnt, double left_weight, double right_weight, float gain, MissingType missing_type);

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
  * \brief Adding prediction value of this tree model to scores
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
  inline double PredictByMap(const std::unordered_map<int, double>& feature_values) const;

  inline int PredictLeafIndex(const double* feature_values) const;
  inline int PredictLeafIndexByMap(const std::unordered_map<int, double>& feature_values) const;


  inline void PredictContrib(const double* feature_values, int num_features, double* output);

  /*! \brief Get Number of leaves*/
  inline int num_leaves() const { return num_leaves_; }

  /*! \brief Get depth of specific leaf*/
  inline int leaf_depth(int leaf_idx) const { return leaf_depth_[leaf_idx]; }

  /*! \brief Get feature of specific split*/
  inline int split_feature(int split_idx) const { return split_feature_[split_idx]; }

  inline double split_gain(int split_idx) const { return split_gain_[split_idx]; }

  /*! \brief Get the number of data points that fall at or below this node*/
  inline int data_count(int node) const { return node >= 0 ? internal_count_[node] : leaf_count_[~node]; }

  /*!
  * \brief Shrinkage for the tree's output
  *        shrinkage rate (a.k.a learning rate) is used to tune the training process
  * \param rate The factor of shrinkage
  */
  inline void Shrinkage(double rate) {
    #pragma omp parallel for schedule(static, 1024) if (num_leaves_ >= 2048)
    for (int i = 0; i < num_leaves_; ++i) {
      leaf_value_[i] *= rate;
    }
    shrinkage_ *= rate;
  }

  inline double shrinkage() const {
    return shrinkage_;
  }

  inline void AddBias(double val) {
    #pragma omp parallel for schedule(static, 1024) if (num_leaves_ >= 2048)
    for (int i = 0; i < num_leaves_; ++i) {
      leaf_value_[i] = val + leaf_value_[i];
    }
    // force to 1.0
    shrinkage_ = 1.0f;
  }

  inline void AsConstantTree(double val) {
    num_leaves_ = 1;
    shrinkage_ = 1.0f;
    leaf_value_[0] = val;
  }

  /*! \brief Serialize this object to string*/
  std::string ToString() const;

  /*! \brief Serialize this object to json*/
  std::string ToJSON() const;

  /*! \brief Serialize this object to if-else statement*/
  std::string ToIfElse(int index, bool predict_leaf_index) const;

  inline static bool IsZero(double fval) {
    if (fval > -kZeroThreshold && fval <= kZeroThreshold) {
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

  void RecomputeMaxDepth();

 private:
  std::string NumericalDecisionIfElse(int node) const;

  std::string CategoricalDecisionIfElse(int node) const;

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
    int cat_idx = static_cast<int>(threshold_[node]);
    if (Common::FindInBitset(cat_threshold_.data() + cat_boundaries_[cat_idx],
                             cat_boundaries_[cat_idx + 1] - cat_boundaries_[cat_idx], int_fval)) {
      return left_child_[node];
    }
    return right_child_[node];
  }

  inline int CategoricalDecisionInner(uint32_t fval, int node) const {
    int cat_idx = static_cast<int>(threshold_in_bin_[node]);
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

  inline void Split(int leaf, int feature, int real_feature, double left_value, double right_value, int left_cnt, int right_cnt,
                    double left_weight, double right_weight, float gain);
  /*!
  * \brief Find leaf index of which record belongs by features
  * \param feature_values Feature value of this record
  * \return Leaf index
  */
  inline int GetLeaf(const double* feature_values) const;
  inline int GetLeafByMap(const std::unordered_map<int, double>& feature_values) const;

  /*! \brief Serialize one node to json*/
  std::string NodeToJSON(int index) const;

  /*! \brief Serialize one node to if-else statement*/
  std::string NodeToIfElse(int index, bool predict_leaf_index) const;

  std::string NodeToIfElseByMap(int index, bool predict_leaf_index) const;

  double ExpectedValue() const;

  /*! \brief This is used fill in leaf_depth_ after reloading a model*/
  inline void RecomputeLeafDepths(int node = 0, int depth = 0);

  /*!
  * \brief Used by TreeSHAP for data we keep about our decision path
  */
  struct PathElement {
    int feature_index;
    double zero_fraction;
    double one_fraction;

    // note that pweight is included for convenience and is not tied with the other attributes,
    // the pweight of the i'th path element is the permutation weight of paths with i-1 ones in them
    double pweight;

    PathElement() {}
    PathElement(int i, double z, double o, double w) : feature_index(i), zero_fraction(z), one_fraction(o), pweight(w) {}
  };

  /*! \brief Polynomial time algorithm for SHAP values (arXiv:1706.06060)*/
  void TreeSHAP(const double *feature_values, double *phi,
                int node, int unique_depth,
                PathElement *parent_unique_path, double parent_zero_fraction,
                double parent_one_fraction, int parent_feature_index) const;

  /*! \brief Extend our decision path with a fraction of one and zero extensions for TreeSHAP*/
  static void ExtendPath(PathElement *unique_path, int unique_depth,
                         double zero_fraction, double one_fraction, int feature_index);

  /*! \brief Undo a previous extension of the decision path for TreeSHAP*/
  static void UnwindPath(PathElement *unique_path, int unique_depth, int path_index);

  /*! determine what the total permutation weight would be if we unwound a previous extension in the decision path*/
  static double UnwoundPathSum(const PathElement *unique_path, int unique_depth, int path_index);

  /*! \brief Number of max leaves*/
  int max_leaves_;
  /*! \brief Number of current leaves*/
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
  /*! \brief Store the information for categorical feature handle and missing value handle. */
  std::vector<int8_t> decision_type_;
  /*! \brief A non-leaf node's split gain */
  std::vector<float> split_gain_;
  // used for leaf node
  /*! \brief The parent of leaf */
  std::vector<int> leaf_parent_;
  /*! \brief Output of leaves */
  std::vector<double> leaf_value_;
  /*! \brief weight of leaves */
  std::vector<double> leaf_weight_;
  /*! \brief DataCount of leaves */
  std::vector<int> leaf_count_;
  /*! \brief Output of non-leaf nodes */
  std::vector<double> internal_value_;
  /*! \brief weight of non-leaf nodes */
  std::vector<double> internal_weight_;
  /*! \brief DataCount of non-leaf nodes */
  std::vector<int> internal_count_;
  /*! \brief Depth for leaves */
  std::vector<int> leaf_depth_;
  double shrinkage_;
  int max_depth_;
};

inline void Tree::Split(int leaf, int feature, int real_feature,
                        double left_value, double right_value, int left_cnt, int right_cnt,
                        double left_weight, double right_weight, float gain) {
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

  split_gain_[new_node_idx] = gain;
  // add two new leaves
  left_child_[new_node_idx] = ~leaf;
  right_child_[new_node_idx] = ~num_leaves_;
  // update new leaves
  leaf_parent_[leaf] = new_node_idx;
  leaf_parent_[num_leaves_] = new_node_idx;
  // save current leaf value to internal node before change
  internal_weight_[new_node_idx] = leaf_weight_[leaf];
  internal_value_[new_node_idx] = leaf_value_[leaf];
  internal_count_[new_node_idx] = left_cnt + right_cnt;
  leaf_value_[leaf] = std::isnan(left_value) ? 0.0f : left_value;
  leaf_weight_[leaf] = left_weight;
  leaf_count_[leaf] = left_cnt;
  leaf_value_[num_leaves_] = std::isnan(right_value) ? 0.0f : right_value;
  leaf_weight_[num_leaves_] = right_weight;
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
    return leaf_value_[0];
  }
}

inline double Tree::PredictByMap(const std::unordered_map<int, double>& feature_values) const {
  if (num_leaves_ > 1) {
    int leaf = GetLeafByMap(feature_values);
    return LeafOutput(leaf);
  } else {
    return leaf_value_[0];
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

inline int Tree::PredictLeafIndexByMap(const std::unordered_map<int, double>& feature_values) const {
  if (num_leaves_ > 1) {
    int leaf = GetLeafByMap(feature_values);
    return leaf;
  } else {
    return 0;
  }
}

inline void Tree::PredictContrib(const double* feature_values, int num_features, double* output) {
  output[num_features] += ExpectedValue();
  // Run the recursion with preallocated space for the unique path data
  if (num_leaves_ > 1) {
    CHECK(max_depth_ >= 0);
    const int max_path_len = max_depth_ + 1;
    std::vector<PathElement> unique_path_data(max_path_len*(max_path_len + 1) / 2);
    TreeSHAP(feature_values, output, 0, 0, unique_path_data.data(), 1, 1, -1);
  }
}

inline void Tree::RecomputeLeafDepths(int node, int depth) {
  if (node == 0) leaf_depth_.resize(num_leaves());
  if (node < 0) {
    leaf_depth_[~node] = depth;
  } else {
    RecomputeLeafDepths(left_child_[node], depth + 1);
    RecomputeLeafDepths(right_child_[node], depth + 1);
  }
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

inline int Tree::GetLeafByMap(const std::unordered_map<int, double>& feature_values) const {
  int node = 0;
  if (num_cat_ > 0) {
    while (node >= 0) {
      node = Decision(feature_values.count(split_feature_[node]) > 0 ? feature_values.at(split_feature_[node]) : 0.0f, node);
    }
  } else {
    while (node >= 0) {
      node = NumericalDecision(feature_values.count(split_feature_[node]) > 0 ? feature_values.at(split_feature_[node]) : 0.0f, node);
    }
  }
  return ~node;
}


}  // namespace LightGBM

#endif   // LightGBM_TREE_H_
