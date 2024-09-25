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
  struct consumed_memory{
    int bytes;
    bool new_threshold;
    bool new_feature;
  };
  struct threshold_info {
      float threshold;
      uint32_t feature;
      uint32_t leftmost;
      uint32_t rightmost;
      bool used;
      bool operator==(const threshold_info& other) const {
        return threshold == other.threshold && leftmost == other.leftmost && rightmost == other.rightmost;
      }
      
  };
  std::ostream& operator<<(std::ostream& os, const threshold_info& thres_info) {
    os << "Used: " << thres_info.threshold << ", Min: " << thres_info.leftmost << "Max: " << thres_info.rightmost;
    return os;
  }
    class MemoryRestrictedForest {
    public:
        explicit MemoryRestrictedForest(const SerialTreeLearner* tree_learner)
                : init_(false), tree_learner_(tree_learner) {}
        void InsertSplitInfo(SplitInfo best_split_info, Tree* tree, int precision){
          // ID of last node is the number of leaves - 2, as tree has num_leaves - 1 nodes and ids start with 0.
          // We need a split that might not have been inserted
          int last_node_id = tree->num_leaves_ - 2;

          // TODO: merge following lines, but depends on if/how we implement rounding.
          float threshold = tree->threshold_[last_node_id];
          //float threshold_rounded = (int)(threshold*pow(10, precision) + 0.5) / pow(10, precision);
          //threshold = threshold_rounded;
          uint32_t feature = tree->split_feature_[last_node_id];

          UpdateThresholdInfo(threshold, feature);

          consumed_memory con_mem = {};
          CalculateSplitMemoryConsumption(/*precision,*/ con_mem, threshold, feature);

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
          Log::Debug("Estimated consumed memory: %d", (est_leftover_memory));
        }
        void CalculateSplitMemoryConsumption(/*int precision,*/ consumed_memory &con_mem, float threshold, uint32_t feature){
          // Two integers to save the id of split and threshold in the overall structure. 
          // TODO dependent on the size of the tree those could be encoded as chars. ( 0 -255 for unsigned chars) (unsigned short 65535)
          con_mem.bytes = 2 * sizeof(short);
        
          std::vector<float>::iterator threshold_it;
          std::vector<u_int32_t>::iterator feature_it;
          
          threshold_it = std::find(thresholds_used_global_.begin(), thresholds_used_global_.end(), threshold);
          feature_it = std::find(features_used_global_.begin(), features_used_global_.end(), feature);
          if (feature_it == features_used_global_.end()) {
            con_mem.bytes += sizeof(short);
            con_mem.new_feature = true;

          }
          if (threshold_it == thresholds_used_global_.end()) {
            con_mem.bytes += sizeof(float);
            con_mem.new_threshold = true;
          }
          // Always the predict value adds to one double. 
          con_mem.bytes += sizeof(float);
        }
        void UpdateThresholdInfo(float threshold, uint32_t feature) {
          // threshold_feature_info
          std::vector<threshold_info>::iterator it;
          // Iterate through the vector update the inserted threshold as used.
          for (it = threshold_feature_info.begin(); it != threshold_feature_info.end(); ++it) {
              if (it->feature == feature && it->threshold == threshold) {
                it->used = true;
                return;
              }
          }
          // This should not happen error handling.
        }
        void CalculateAndInsertThresholdVariability(const Dataset* train_data, const BinMapper* binmapper, int featureidx, float threshold){
          // Gives the min and max value - scan data yourself?
          const Bin* bin = train_data->FeatureGroupBin(featureidx); 
          // Somehow the bin does not have min and max values seems like it is not the right bin.
          printf("Test Reinterpret Cast Min %f Max %f ...\n", bin->min, bin->max);
        }
        void InsertThresholdFeatureInfo(uint32_t threshold, uint32_t featureidx) {
          threshold_info info;
          info.threshold = threshold;
          // info.leftmost = left;
          // info.rightmost = right;
          if (std::find(threshold_feature_info.begin(), threshold_feature_info.end(), info) == threshold_feature_info.end()) {
            // If not found, add it to the vector.
            threshold_feature_info.push_back(info);
          }
        }
        void CheckThresholdVariability(float min, float max) {
          for (const threshold_info& elem : threshold_feature_info) {
            // TODO: Check if there is a threshold "close by"
          }  
        }
        static bool IsEnable(const Config* config) {
            if (config->tinygbdt_forestsize == 0.0f) {
                Log::Debug("MemoryRestrictedForest disabled");
                return false;
            } else {
                if (config->num_iterations != 100 && config->max_depth > 0) {
                  // TODO TinyGBT do we automatically want to set values if those are not set?
                  // TODO I guess having one set is the easiest as we can eventually scale in the other direction.
                } else if (config->num_iterations != 100 ) {
                } else if (config->max_depth > 0) {
                    // Assuming we have a fully covered binary tree get the maximum nodes in a single tree.
                    // int max_nodes = static_cast<int>(pow(2, config->max_depth + 1) - 1);
                } else {
                  // TODO TinyGBT none set we could either set a value assuming an average memory consumption for one if both but estimation will lead to a loss in accuracy.
                }
                return true;
            }
        }
        void Init(const double treesize) {
          est_leftover_memory = (int) treesize;
          Log::Debug("MemoryRestrictedForest Init");
        }

        bool init_;
        int est_leftover_memory;
        const SerialTreeLearner* tree_learner_;
        std::vector<threshold_info> threshold_feature_info;
        
        // TODO change this to short? or even smaller?
        /*! \brief count feature use; TODO: possible to use fewer bits? */
        std::vector<u_int32_t> features_used_global_;
        /*! \brief record thresholds used for split; TODO: round values to avoid dissimilarity of (almost) same values (-> quantization?) */
        std::vector<float> thresholds_used_global_;

    };
}
#endif //LIGHTGBM_MEMORY_RESTRICTED_FOREST_H
