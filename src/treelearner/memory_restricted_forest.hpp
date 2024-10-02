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
  };

  struct threshold_info {
    float threshold;
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
    void InsertLeavesInformation(std::vector<double> leaf_value_) {
      for (double leaf_value : leaf_value_) {
        InsertLeafInformation(static_cast<float>(leaf_value));
      }
    }
    void InsertLeafInformation(float leaf_value) {
      auto threshold_it = std::find(thresholds_used_global_.begin(),
                                    thresholds_used_global_.end(), leaf_value);
      Log::Debug("Leaf value inserted: %f", leaf_value);
      // If the threshold is not present and cannot be adjusted to a close by threshold.
      if (threshold_it == thresholds_used_global_.end()) {
        est_leftover_memory -= sizeof(float);
        thresholds_used_global_.push_back(leaf_value);
      }
      est_leftover_memory -= sizeof(short);
    }

    void InsertSplitInfo(const SplitInfo &best_split_info, const Tree *tree, int precision,
                         const Dataset *train_data_) {
      // ID of last node is the number of leaves - 2, as tree has num_leaves - 1 nodes and ids start with 0.
      // We need a split that might not have been inserted
      const int last_node_id = tree->num_leaves_ - 2;

      // TODO: merge following lines, but depends on if/how we implement rounding.
      const float threshold = tree->threshold_[last_node_id];
      //float threshold_rounded = (int)(threshold*pow(10, precision) + 0.5) / pow(10, precision);
      //threshold = threshold_rounded;
      const uint32_t feature = tree->split_feature_[last_node_id];
      const BinMapper *bin_mapper = train_data_->FeatureBinMapper(feature);

      InsertThresholdFeatureInfo(threshold, feature, bin_mapper);

      consumed_memory con_mem = {};
      CalculateSplitMemoryConsumption(con_mem, threshold, feature);

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

    float CalculateSplitMemoryConsumption(consumed_memory &con_mem, float threshold, uint32_t feature) {
      // Two integers to save the id of split and threshold in the overall structure and a float for the predict value.
      // TODO dependent on the size of the tree those could be encoded as chars. ( 0 -255 for unsigned chars) (unsigned short 65535)
      con_mem.bytes = 2 * sizeof(short) + sizeof(float);

      std::vector<uint32_t>::iterator feature_it;

      std::vector<float>::iterator threshold_it = std::find(thresholds_used_global_.begin(),
                                                            thresholds_used_global_.end(), threshold);
      feature_it = std::find(features_used_global_.begin(), features_used_global_.end(), feature);
      if (feature_it == features_used_global_.end()) {
        con_mem.bytes += sizeof(short);
        con_mem.new_feature = true;
      }
      // If the threshold is not present and cannot be adjusted to a close by threshold.
      if (threshold_it == thresholds_used_global_.end()) {
        float possible_thres = CalculateAndInsertThresholdVariability(threshold);
        if (possible_thres == threshold) {
          con_mem.bytes += sizeof(float);
          con_mem.new_threshold = true;
        } else {
          return possible_thres;
        }
      }
      return threshold;
    }

    float CalculateAndInsertThresholdVariability(float threshold) {
      float epsilon = 0.0f; // precision;
      float best_sofar = threshold;
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

    void InsertThresholdFeatureInfo(float threshold, uint32_t featureidx, const BinMapper *bin_mapper) {
      threshold_info info{};
      info.threshold = threshold;
      info.feature = featureidx;
      MinMax minmax = bin_mapper->getMinAndMax(threshold);
      info.leftmost = minmax.getMin();
      info.rightmost = minmax.getMax();
      info.used = true;
      if (std::find(threshold_feature_info.begin(), threshold_feature_info.end(), info) == threshold_feature_info.end()) {
        threshold_feature_info.push_back(info);
      }
    }

    static bool IsEnable(const Config *config) {
      if (config->tinygbdt_forestsize == 0.0f) {
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

    void Init(const double treesize, const double precision) {
      est_leftover_memory = (int) treesize;
      this->precision = precision;
      Log::Debug("MemoryRestrictedForest Init");
    }

    bool init_;
    int est_leftover_memory{};
    double precision;
    const SerialTreeLearner *tree_learner_;
    std::vector<threshold_info> threshold_feature_info;

    // TODO change this to short? or even smaller?
    /*! \brief count feature use; TODO: possible to use fewer bits? */
    std::vector<u_int32_t> features_used_global_;
    /*! \brief record thresholds used for split; TODO: round values to avoid dissimilarity of (almost) same values (-> quantization?) */
    std::vector<float> thresholds_used_global_;
  };
}
#endif //LIGHTGBM_MEMORY_RESTRICTED_FOREST_H
