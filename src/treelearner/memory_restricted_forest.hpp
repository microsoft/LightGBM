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
          // ID of last node is the number of leaves - 2, as tree has num_leaves - 1 nodes and ids start with 0.
          // TODO we need a split that might not have been inserted
          int last_node_id = tree->num_leaves_ - 2;

          // TODO: merge following lines, but depends on if/how we implement rounding.
          float threshold = tree->threshold_[last_node_id];
          //float threshold_rounded = (int)(threshold*pow(10, precision) + 0.5) / pow(10, precision);
          //threshold = threshold_rounded;
          uint32_t feature = tree->split_feature_[last_node_id];

          // TODO Nina as we have the Histograms we can not really get the left and right value. I mean theoretically we could looking when the count 
          // ... value changes but that consumes time and I guess would be a no go for merging into LightGBM.
          // NVM I think we can.
          InsertThresholdInfo(threshold, feature, best_split_info.left_count, best_split_info.right_count);

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
        void CalculateThresholdVariability(const BinMapper* binmapper, const Dataset* train_data, int featureidx, uint32_t bin_idx, FeatureHistogram* histogram_array_){
          // Gives the min and max value - scan data yourself?
          uint8_t bit_type = 0;
          bool is_sparse = false;
          BinIterator* bin_iterator = nullptr;
          const Bin* bin = train_data->FeatureGroupBin(featureidx); 
          // This does not seem to work.
          const void * data = bin->GetColWiseData(&bit_type, &is_sparse, &bin_iterator);
          // std::cout << "The value at the pointer address is: " << *(static_cast<const float*>(data)) << std::endl;
        }
        void InsertThresholdInfo(uint32_t threshold, uint32_t featureidx, uint32_t left, uint32_t right) {
          threshold_info info;
          info.threshold_used = threshold;
          info.leftmost_value = left;
          info.rightmost_value = right;
          if (std::find(threshold_used_.begin(), threshold_used_.end(), info) == threshold_used_.end()) {
            // If not found, add it to the vector.
            threshold_used_.push_back(info);
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
        std::vector<threshold_info> threshold_used_;
        
        // TODO change this to short? or even smaller?
        /*! \brief count feature use; TODO: possible to use fewer bits? */
        std::vector<u_int32_t> features_used_global_;
        /*! \brief record thresholds used for split; TODO: round values to avoid dissimilarity of (almost) same values (-> quantization?) */
        std::vector<float> thresholds_used_global_;

    };
}
#endif //LIGHTGBM_MEMORY_RESTRICTED_FOREST_H
