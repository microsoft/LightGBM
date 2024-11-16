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
    int bits;
    bool new_threshold;
    bool new_feature;
    int tindex;
    int findex;
    int feature_bits;
  };

  struct ref_tree {
    int tree_id;
    std::vector<int> feature_ids;
    std::vector<double> thresholds;
  };
  std::ostream & operator << (std::ostream & outs, const ref_tree & ref_t) {
    outs << ref_t.tree_id << " -> ";
    for (double feature : ref_t.feature_ids) {
      outs << feature << " ";
    }
    return outs;
  }

  struct threshold_info {
    std::vector<double> thresholds_;
    uint32_t feature;
    bool used;
  };

  class MemoryRestrictedForest {
  public:
    explicit MemoryRestrictedForest(const SerialTreeLearner *tree_learner)
      : init_(false), tree_learner_(tree_learner) {
    }
    void InsertLeafInformation(double leaf_value) {
      est_leftover_memory -= sizeof(float);
#pragma omp critical
      thresholds_used_global_.push_back(leaf_value);
    }

    void UpdateMemoryForTree(Tree* tree) {
#pragma omp critical
      tree_size_.push_back(tree->getNumberNodes());
    }
    void InsertSplitInfo(const Tree *tree, const Dataset *train_data_) {
      const int last_node_id = tree->num_leaves_ - 2;
      const double threshold = RoundDecimals(tree->threshold_[last_node_id], this->precision);
      const uint32_t feature = tree->split_feature_[last_node_id];
      const BinMapper *bin_mapper = train_data_->FeatureBinMapper(feature);
      consumed_memory con_mem = {};
      CalculateSplitMemoryConsumption(con_mem, threshold, feature);
      if (con_mem.new_feature) {
        features_used_global_[fcounter] = (feature);
        fcounter++;
      }
      if (con_mem.new_threshold) {
#pragma omp critical
        thresholds_used_global_.push_back(threshold);
      }
      // Always the predict value adds to one double.
      est_leftover_memory -= con_mem.bits;
    }

    bool isAllBool(const std::vector<double>& column) {
      for (const auto& value : column) {
        if (value != 0.0 && value != 1.0) {
          return false;
        }
      }
      return true;
    }
    bool isAllInteger(const std::vector<double>& column) {
      bool isInteger = true;
      for (const auto& value : column) {
        if (std::floor(value) != value) { // Check if value is not an integer
          return false;
        }
      }
      return isInteger;
    }

    void CalculateSplitMemoryConsumption(consumed_memory &con_mem, double threshold, uint32_t feature) {
      int size = thresholds_used_global_.size();
      bool foundthrehold = false;
      for (int i = 0; i < size; i++) {
        if (threshold == thresholds_used_global_[i]) {
          foundthrehold = true;
        }
      }
      //auto itf = std::find(features_used_global_.begin(), features_used_global_.end(), feature);
      int sizef = features_used_global_.size();
      bool foundfeature = false;
      for (int i = 0; i < sizef; i++) {
        if (feature == features_used_global_[i]) {
          foundfeature = true;
        }
      }
      // In case the feature is not used 8 bits are added for representing a bits_single and bits_ref.
      if (!foundfeature) {
        con_mem.bits += 4;
        con_mem.new_feature = true;
      }
      if (!foundthrehold) {
        con_mem.bits += 16;
        con_mem.new_threshold = true;
      }
    }

    double RoundDecimals(double number, double decimals) {
      double rounded = ((double)((int)(number * pow(10.0, decimals) + .5))) / pow(10.0, decimals);
      return rounded;
    }

    static bool IsEnable(const Config *config) {
      if (config->tinygbdt_forestsize == 0) {
        Log::Info("MemoryRestrictedForest disabled");
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

    void Init(const int treesize, const double precision, int max_depth_) {
      max_depth = max_depth_;
      ref_trees_.push_back({});
      ref_trees_[treecounter].tree_id = treecounter;
      est_leftover_memory = treesize;
      this->precision = precision;
      auto train_data = tree_learner_->train_data_;
      features_used_global_.resize(train_data->num_features());
    }
    void printForest() {
      std::stringstream out;
      out << "Leftover memory: " << est_leftover_memory;
      out << "\n";
      std::cout << out.str();
    }
    bool init_;
    int est_leftover_memory, max_depth;
    double precision;
    const SerialTreeLearner *tree_learner_;
    /*! \brief count feature use; */
    /*! \brief record thresholds used for split; */
    std::vector<double> thresholds_used_global_;
    std::vector<int> tree_size_;
    std::vector<uint32_t> features_used_global_;
    int fcounter = 0;
    std::vector<ref_tree> ref_trees_;
    int treecounter = 0;
  };
}
#endif //LIGHTGBM_MEMORY_RESTRICTED_FOREST_H
