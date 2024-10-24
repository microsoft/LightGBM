/*!
 * Copyright (c) 2019 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_MEMORY_RESTRICTED_FOREST_H
#define LIGHTGBM_MEMORY_RESTRICTED_FOREST_H

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/utils/log.h>

#include <vector>

namespace LightGBM {
  struct consumed_memory {
    int bytes;
    bool new_threshold;
    bool new_feature;
    int tindex;
    int findex;
  };

  struct threshold_info {
    double threshold;
    uint32_t feature;
    double leftmost;
    double rightmost;
    bool used;

    bool operator==(const threshold_info &other) const {
      return threshold == other.threshold && leftmost == other.leftmost && rightmost == other.rightmost;
    }
  };

  inline std::ostream &operator<<(std::ostream &os, const threshold_info &thres_info) {
    os << "Used: " << thres_info.threshold << ", Min: " << thres_info.leftmost << "Max: " << thres_info.rightmost;
    return os;
  }

  class MemoryRestrictedForest {
  public:
    explicit MemoryRestrictedForest(const SerialTreeLearner *tree_learner)
      : init_(false), tree_learner_(tree_learner) {
    }
    void InsertLeafInformation(double leaf_value) {
      double n_leave_value = CalculateAndInsertThresholdVariability(leaf_value);
      if (n_leave_value == leaf_value) {
        // value in the threshold table.
        est_leftover_memory -= sizeof(float);
        thresholds_used_global_.push_back(leaf_value);
      }
      // reference in the "tree" table.
      est_leftover_memory -= sizeof(short);
      //Log::Debug("Leaf value to be inserted: %.3f", n_leave_value);
    }
    void InsertLeavesInformation(std::vector<double> leaf_value_) {
      Log::Debug("Inserting leaf information");
      for (double leaf_value : leaf_value_) {
        InsertLeafInformation(leaf_value);
      }
    }
    void UpdateMemoryForTree(Tree* tree) {
      // Tree weight.
      // TODO Delete Not needed
      est_leftover_memory -= sizeof(float);
      tree_size_.push_back(tree->getNumberNodes());
    }
    void InsertSplitInfo(const Tree *tree, const Dataset *train_data_) {
      // ID of last node is the number of leaves - 2, as tree has num_leaves - 1 nodes and ids start with 0.
      // We need a split that might not have been inserted
      const int last_node_id = tree->num_leaves_ - 2;

      // TODO: merge following lines, but depends on if/how we implement rounding.
      const double threshold = tree->threshold_[last_node_id];
      const uint32_t feature = tree->split_feature_[last_node_id];
      const BinMapper *bin_mapper = train_data_->FeatureBinMapper(feature);

      InsertThresholdFeatureInfo(threshold, feature, bin_mapper);

      consumed_memory con_mem = {};
      CalculateSplitMemoryConsumption(con_mem, threshold, feature);
      CalculateEffectQuantization(&con_mem, tree);
      // How do we know the order?
      if (con_mem.new_feature) {
        // If yes, only one int value is added.
        features_used_global_.push_back(feature);
      }
      if (con_mem.new_threshold) {
        // If yes, only one double value is added.
        thresholds_used_global_.push_back(threshold);
      }
      // Always the predict value adds to one double.
      est_leftover_memory -= con_mem.bytes;
    }

    void CalculateEffectQuantization(consumed_memory *con_mem, const Tree * tree) {
      // Every time we cross a power of two we require a new bit.
      size_t fug_size_ = features_used_global_.size();
      size_t next_power_of_two = static_cast<size_t>(std::pow(2, std::ceil(std::log2(fug_size_ + 1))));
      if (fug_size_ + 1 > next_power_of_two) {
        f_needed_bits++;
      }

      size_t tug_size = thresholds_used_global_.size();
      next_power_of_two = static_cast<size_t>(std::pow(2, std::ceil(std::log2(tug_size + 1))));
      if (tug_size + 1 > next_power_of_two) {
        t_needed_bits++;
      }
      // Until now all trees could use the less bit representation.
      // Reduce the current tree and increase the bits for the following trees.
      est_leftover_memory -= f_needed_bits + t_needed_bits * tree->getNumberNodes();
    }
    double CalculateSplitMemoryConsumption(consumed_memory &con_mem, double threshold, uint32_t feature) {
      // Insert the memory consumption of the two global tables.
      std::vector<uint32_t>::iterator feature_it;
      std::vector<double>::iterator threshold_it = std::find(thresholds_used_global_.begin(),
                                                            thresholds_used_global_.end(), threshold);
      feature_it = std::find(features_used_global_.begin(), features_used_global_.end(), feature);
      if (feature_it == features_used_global_.end()) {
        con_mem.bytes += f_needed_bits;
        con_mem.new_feature = true;
      } else {
        con_mem.findex = std::distance(features_used_global_.begin(), feature_it);
      }
      // If the threshold is not present and cannot be adjusted to a close by threshold.
      if (threshold_it == thresholds_used_global_.end()) {
        double possible_thres = CalculateAndInsertThresholdVariability(threshold);
        if (possible_thres == threshold) {
          con_mem.bytes += sizeof(float);
          con_mem.new_threshold = true;
        } else {
          return possible_thres;
        }
      } else {
        con_mem.tindex = std::distance(thresholds_used_global_.begin(), threshold_it);
      }
      // TODO: change needed bits as we are using int and float values.
      con_mem.bytes += f_needed_bits + t_needed_bits;
      return threshold;
    }

    double CalculateAndInsertThresholdVariability(double threshold) {
      double epsilon = this->precision; // precision;
      double best_sofar = threshold;
      for (const threshold_info &elem: threshold_feature_info) {
        // In case the threshold is close to an already inserted threshold take it.
        if (std::fabs(threshold - elem.threshold) < epsilon) {
          if (best_sofar > std::fabs(threshold - elem.threshold)) {
            best_sofar = elem.threshold;
          }
        }
      }
      // TODO check for min and max?
      return best_sofar;
    }

    void InsertThresholdFeatureInfo(double threshold, uint32_t featureidx, const BinMapper *bin_mapper) {
      threshold_info info{};
      info.threshold = threshold;
      info.feature = featureidx;
      // TODO CLEARUP MINMAX
      // MinMax minmax = bin_mapper->getMinAndMax(threshold);
      // info.leftmost = minmax.getMin();
      // info.rightmost = minmax.getMax();
      info.used = true;
      if (std::find(threshold_feature_info.begin(), threshold_feature_info.end(), info) == threshold_feature_info.end()) {
        threshold_feature_info.push_back(info);
      }
    }

    static bool IsEnable(const Config *config) {
      if (config->tinygbdt_forestsize == 0) {
        Log::Debug("MemoryRestrictedForest disabled");
        return false;
      }
      if (config->num_iterations != 100 && config->max_depth > 0) {
        // TODO TinyGBT do we automatically want to set values if those are not set?
        // TODO I guess having one set is the easiest as we can eventually scale in the other direction.
      } else if (config->num_iterations != 100) {
      } else if (config->max_depth > 0) {
        // Assuming we have a fully covered binary tree get the maximum nodes in a single tree.
        // int max_nodes = static_cast<int>(pow(2, config->max_depth + 1) - 1);
      } else {
        // TODO TinyGBT none set we could either set a value assuming an average memory consumption for one if both but estimation will lead to a loss in accuracy.
      }
      return true;
    }

    void Init(const int treesize, const double precision) {
      est_leftover_memory = treesize;
      this->precision = precision;
    }

    bool init_;
    int est_leftover_memory{};
    // In the unrealistic best case we have 2 features -> 1 bit. But this would require a optimization of the header files, ...
    // so right now we assume 4 bytes for float and 2 bytes for int.
    int f_needed_bits = 16;
    int t_needed_bits = 32;
    double precision;
    const SerialTreeLearner *tree_learner_;
    std::vector<threshold_info> threshold_feature_info;
    /*! \brief count feature use; */
    std::vector<u_int32_t> features_used_global_;
    /*! \brief record thresholds used for split; */
    std::vector<double> thresholds_used_global_;
    std::vector<int> tree_size_;
  };
}
#endif //LIGHTGBM_MEMORY_RESTRICTED_FOREST_H
