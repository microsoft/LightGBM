#ifndef LIGHTGBM_TREE_H_
#define LIGHTGBM_TREE_H_

#include <LightGBM/meta.h>
#include <LightGBM/feature.h>
#include <LightGBM/dataset.h>

#include <string>
#include <vector>
#include <memory>

namespace LightGBM {


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
  * \param bin_type type of this feature, numerical or categorical
  * \param threshold Threshold(bin) of split
  * \param real_feature Index of feature, the original index on data
  * \param threshold_double Threshold on feature value
  * \param left_value Model Left child output
  * \param right_value Model Right child output
  * \param left_cnt Count of left child
  * \param right_cnt Count of right child
  * \param gain Split gain
  * \return The index of new leaf.
  */
  int Split(int leaf, int feature, BinType bin_type, unsigned int threshold, int real_feature,
    double threshold_double, double left_value,
    double right_value, data_size_t left_cnt, data_size_t right_cnt, double gain);

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
  void AddPredictionToScore(const Dataset* data, data_size_t num_data,
                                                       score_t* score) const;

  /*!
  * \brief Adding prediction value of this tree model to scorese
  * \param data The dataset
  * \param used_data_indices Indices of used data
  * \param num_data Number of total data
  * \param score Will add prediction to score
  */
  void AddPredictionToScore(const Dataset* data,
                            const data_size_t* used_data_indices,
                            data_size_t num_data, score_t* score) const;

  /*!
  * \brief Prediction on one record 
  * \param feature_values Feature value of this record
  * \return Prediction result
  */
  inline double Predict(const double* feature_values) const;
  inline int PredictLeafIndex(const double* feature_values) const;

  /*! \brief Get Number of leaves*/
  inline int num_leaves() const { return num_leaves_; }

  /*! \brief Get depth of specific leaf*/
  inline int leaf_depth(int leaf_idx) const { return leaf_depth_[leaf_idx]; }

  /*! \brief Get feature of specific split*/
  inline int split_feature_real(int split_idx) const { return split_feature_real_[split_idx]; }

  /*!
  * \brief Shrinkage for the tree's output
  *        shrinkage rate (a.k.a learning rate) is used to tune the traning process
  * \param rate The factor of shrinkage
  */
  inline void Shrinkage(double rate) {
    for (int i = 0; i < num_leaves_; ++i) {
      leaf_value_[i] = leaf_value_[i] * rate;
    }
  }

  /*! \brief Serialize this object to string*/
  std::string ToString();

  /*! \brief Serialize this object to json*/
  std::string ToJSON();

  template<typename T>
  static bool CategoricalDecision(T fval, T threshold) {
    if (static_cast<int>(fval) == static_cast<int>(threshold)) {
      return true;
    } else {
      return false;
    }
  }

  template<typename T>
  static bool NumericalDecision(T fval, T threshold) {
    if (fval <= threshold) {
      return true;
    } else {
      return false;
    }
  }

  static const char* GetDecisionTypeName(int8_t type) {
    if (type == 0) {
      return "no_greater";
    } else {
      return "is";
    }
  }

  static std::vector<std::function<bool(unsigned int, unsigned int)>> inner_decision_funs;
  static std::vector<std::function<bool(double, double)>> decision_funs;

private:
  /*!
  * \brief Find leaf index of which record belongs by data
  * \param data The dataset
  * \param data_idx Index of record
  * \return Leaf index
  */
  inline int GetLeaf(const std::vector<std::unique_ptr<BinIterator>>& iterators,
                                           data_size_t data_idx) const;

  /*!
  * \brief Find leaf index of which record belongs by features
  * \param feature_values Feature value of this record
  * \return Leaf index
  */
  inline int GetLeaf(const double* feature_values) const;

  /*! \brief Serialize one node to json*/
  inline std::string NodeToJSON(int index);

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
  std::vector<int> split_feature_;
  /*! \brief A non-leaf node's split feature, the original index */
  std::vector<int> split_feature_real_;
  /*! \brief A non-leaf node's split threshold in bin */
  std::vector<unsigned int> threshold_in_bin_;
  /*! \brief A non-leaf node's split threshold in feature value */
  std::vector<double> threshold_;
  /*! \brief Decision type, 0 for '<='(numerical feature), 1 for 'is'(categorical feature) */
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
};


inline double Tree::Predict(const double* feature_values) const {
  int leaf = GetLeaf(feature_values);
  return LeafOutput(leaf);
}

inline int Tree::PredictLeafIndex(const double* feature_values) const {
  int leaf = GetLeaf(feature_values);
  return leaf;
}

inline int Tree::GetLeaf(const std::vector<std::unique_ptr<BinIterator>>& iterators,
                                       data_size_t data_idx) const {
  int node = 0;
  while (node >= 0) {
    if (inner_decision_funs[decision_type_[node]](
        iterators[split_feature_[node]]->Get(data_idx),
        threshold_in_bin_[node])) {
      node = left_child_[node];
    } else {
      node = right_child_[node];
    }
  }
  return ~node;
}

inline int Tree::GetLeaf(const double* feature_values) const {
  int node = 0;
  while (node >= 0) {
    if (decision_funs[decision_type_[node]](
        feature_values[split_feature_real_[node]],
        threshold_[node])) {
      node = left_child_[node];
    } else {
      node = right_child_[node];
    }
  }
  return ~node;
}

}  // namespace LightGBM

#endif   // LightGBM_TREE_H_
