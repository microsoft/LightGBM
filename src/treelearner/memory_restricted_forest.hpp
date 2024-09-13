/*!
 * Copyright (c) 2019 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_MEMORY_RESTRICTED_FOREST_H
#define LIGHTGBM_MEMORY_RESTRICTED_FOREST_H

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/threading.h>

#include <vector>

#include "data_partition.hpp"
#include "serial_tree_learner.h"
#include "split_info.hpp"

namespace LightGBM {
    class MemoryRestrictedForest {
    public:
        explicit MemoryRestrictedForest(const SerialTreeLearner* tree_learner)
                : init_(false), tree_learner_(tree_learner) {}

        // TODO Nina appropriate place to store features used? Rather create an own struct that only gets created if required?
        struct threshold_info {
            uint32_t threshold_used;
            uint32_t leftmost_value;
            uint32_t rightmost_value;
            bool operator==(const threshold_info& other) const {
              return threshold_used == other.threshold_used && leftmost_value == other.leftmost_value && rightmost_value == other.rightmost_value;
            }
        };
        void InsertSplitInfo(SplitInfo best_split_info, Tree* tree, int precision){
          // TODO Nina as we have the Histograms we can not really get the left and right value. I mean theoretically we could looking when the count value changes but that consumes time and I guess would be a no go for merging into LightGBM.
          InsertThresholdInfo(best_split_info.threshold, best_split_info.left_count, best_split_info.right_count);
          // TODO Jan/Nina merging with Method from Jan either storing result here or in the tree.
          // tree->toArrayPointer();
          // For the overall memory consumption it is assumed that each split that uses a feature not yet used increases the memory by two double values (split and predict value).
          // Was feature already used?
          estimated_consumed_memory += 2 * sizeof(int);
  
          // id of last node is the number of leaves - 2, as tree has num_leaves - 1 nodes and ids start with 0
          int last_node_id = tree->num_leaves_ - 2;

          // TODO: merge following lines, but depends on if/how we implement rounding 
          float threshold = tree->threshold_[last_node_id];
          float threshold_rounded = (int)(threshold*pow(10, precision) + 0.5) / pow(10, precision);
          threshold = threshold_rounded;

          uint32_t feature = tree->split_feature_[last_node_id];
          Log::Debug("threshold: %f, node_id: %i, feature: %u", threshold, last_node_id, feature);

          std::vector<float>::iterator threshold_it;
          std::vector<u_int32_t>::iterator feature_it;
          
          threshold_it = std::find(thresholds_used_global_.begin(), thresholds_used_global_.end(), threshold);
          feature_it = std::find(features_used_global_.begin(), features_used_global_.end(), feature);
          if (feature_it == features_used_global_.end()) {
            // If yes, only one double value is added.
            estimated_consumed_memory += sizeof(double);
            features_used_global_.push_back(feature);
          }
          if (threshold_it == thresholds_used_global_.end()) {
            // If yes, only one double value is added.
            estimated_consumed_memory += sizeof(double);
            thresholds_used_global_.push_back(threshold);
          }
          Log::Debug("Estimated consumed memory: %f", (estimated_consumed_memory));
        }
        void InsertThresholdInfo(uint32_t threshold, uint32_t left, uint32_t right) {
          threshold_info info;
          info.threshold_used = threshold;
          info.leftmost_value = left;
          info.rightmost_value = right;
          if (std::find(threshold_used.begin(), threshold_used.end(), info) == threshold_used.end()) {
            // If not found, add it to the vector
            threshold_used.push_back(info);
          }
        }
        static bool IsEnable(const Config* config) {
            if (config->tinygbdt_forestsize == 0.0f) {
                Log::Debug("MemoryRestrictedForest disabled");
                return false;
            } else {
                if (config->num_iterations != 100 && config->max_depth > 0) {
                    Log::Debug("num_tree and max_depth set");
                  // TODO TinyGBT do we automatically want to set values if those are not set?
                  // TODO I guess having one set is the easiest as we can eventually scale in the other direction.
                } else if (config->num_iterations != 100 ) {
                    Log::Debug("num_tree set");
                } else if (config->max_depth > 0) {
                    Log::Debug("max_depth set");
                    // Assuming we have a fully covered binary tree get the maximum nodes in a single tree.
                    // int max_nodes = static_cast<int>(pow(2, config->max_depth + 1) - 1);
                } else {
                  // TODO TinyGBT none set we could either set a value assuming an average memory consumption for one if both but estimation will lead to a loss in accuracy.
                    Log::Debug("Non set");
                }
                return true;
            }
        }
        void Init() {
          Log::Debug("MemoryRestrictedForest Init");
        }

        bool init_;
        float estimated_consumed_memory;
        const SerialTreeLearner* tree_learner_;
        std::vector<threshold_info> threshold_used;
        
        /*! \brief count feature use; TODO: possible to use fewer bits? */
        std::vector<u_int32_t> features_used_global_;
        /*! \brief record thresholds used for split; TODO: round values to avoid dissimilarity of (almost) same values (-> quantization?) */
        std::vector<float> thresholds_used_global_;

    };
}
#endif //LIGHTGBM_MEMORY_RESTRICTED_FOREST_H
