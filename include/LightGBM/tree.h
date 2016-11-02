#ifndef LIGHTGBM_TREE_H_
#define LIGHTGBM_TREE_H_

#include <LightGBM/meta.h>
#include <LightGBM/feature.h>
#include <LightGBM/dataset.h>

#include <string>
#include <vector>

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
  * \param threshold Threshold(bin) of split
  * \param real_feature Index of feature, the original index on data
  * \param threshold_double Threshold on feature value
  * \param left_value Model Left child output
  * \param right_value Model Right child output
  * \param gain Split gain
  * \return The index of new leaf.
  */
  int Split(int leaf, int feature, unsigned int threshold, int real_feature,
    double threshold_double, double left_value,
    double right_value, double gain);

  /*! \brief Get the output of one leave */
  inline double LeafOutput(int leaf) const { return leaf_value_[leaf]; }

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

  /*! \brief Serialize this object by string*/
  std::string ToString();

  /*! \brief Disable copy */
  Tree& operator=(const Tree&) = delete;
  /*! \brief Disable copy */
  Tree(const Tree&) = delete;
private:
  /*!
  * \brief Find leaf index of which record belongs by data
  * \param data The dataset
  * \param data_idx Index of record
  * \return Leaf index
  */
  inline int GetLeaf(const std::vector<BinIterator*>& iterators,
                                           data_size_t data_idx) const;

  /*!
  * \brief Find leaf index of which record belongs by features
  * \param feature_values Feature value of this record
  * \return Leaf index
  */
  inline int GetLeaf(const double* feature_values) const;

  /*! \brief Number of max leaves*/
  int max_leaves_;
  /*! \brief Number of current levas*/
  int num_leaves_;
  // following values used for non-leaf node
  /*! \brief A non-leaf node's left child */
  int* left_child_;
  /*! \brief A non-leaf node's right child */
  int* right_child_;
  /*! \brief A non-leaf node's split feature */
  int* split_feature_;
  /*! \brief A non-leaf node's split feature, the original index */
  int* split_feature_real_;
  /*! \brief A non-leaf node's split threshold in bin */
  unsigned int* threshold_in_bin_;
  /*! \brief A non-leaf node's split threshold in feature value */
  double* threshold_;
  /*! \brief A non-leaf node's split gain */
  double* split_gain_;
  // used for leaf node
  /*! \brief The parent of leaf */
  int* leaf_parent_;
  /*! \brief Output of leaves */
  double* leaf_value_;
  /*! \brief Depth for leaves */
  int* leaf_depth_;
};


inline double Tree::Predict(const double* feature_values) const {
  int leaf = GetLeaf(feature_values);
  return LeafOutput(leaf);
}

inline int Tree::PredictLeafIndex(const double* feature_values) const {
  int leaf = GetLeaf(feature_values);
  return leaf;
}

inline int Tree::GetLeaf(const std::vector<BinIterator*>& iterators,
                                       data_size_t data_idx) const {
  int node = 0;
  while (node >= 0) {
    if (iterators[split_feature_[node]]->Get(data_idx) <=
                                  threshold_in_bin_[node]) {
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
    if (feature_values[split_feature_real_[node]] <= threshold_[node]) {
      node = left_child_[node];
    } else {
      node = right_child_[node];
    }
  }
  return ~node;
}

}  // namespace LightGBM

#endif   // LightGBM_TREE_H_
