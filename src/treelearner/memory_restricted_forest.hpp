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
        void InsertSplitInfo(SplitInfo best_split_info, Tree* tree){
          // TODO Nina as we have the Histograms we can not really get the left and right value. I mean theoretically we could looking when the count value changes but that consumes time and I guess would be a no go for merging into LightGBM.
          InsertThresholdInfo(best_split_info.threshold, best_split_info.left_count, best_split_info.right_count);
          // TODO Jan/Nina merging with Method from Jan either storing result here or in the tree.
          // tree->toArrayPointer();
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
                return false;
            } else {
                if (config->num_iterations && config->max_depth) {
                    Log::Debug("num_tree and max_depth set");
                  // TODO TinyGBT do we automatically want to set values if those are not set?
                  // TODO I guess having one set is the easiest as we can eventually scale in the other direction.
                } else if (config->num_iterations) {
                    Log::Debug("num_tree set");
                } else if (config->max_depth) {
                    Log::Debug("max_depth set");
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

    };
}
#endif //LIGHTGBM_MEMORY_RESTRICTED_FOREST_H
