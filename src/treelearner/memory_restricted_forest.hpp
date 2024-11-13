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

  struct feature_info {
    int feature_id;
    int bits_needed; // Example given bool needs 1 bit, float 8 bits etc., int with max 16 needs 4 bits
    int stored_rel_thresholds=0;
    std::vector<double> thresholds;
    feature_info(int feature_id_, int bits_needed_, int stored_rel_thresholds_)
      : feature_id(feature_id_), bits_needed(bits_needed_), stored_rel_thresholds(stored_rel_thresholds_) {
    }
  };

  std::ostream & operator << (std::ostream & out, const feature_info & feature) {
    out << feature.feature_id << " ";
    out << feature.bits_needed << " ";
    out << feature.stored_rel_thresholds << " ";
    for (double threshold : feature.thresholds) {
      out << threshold << " ";
    }
    return out;
  }
  struct feature_thresholds {
    int feature_id;
    std::vector<double> thresholds;
  };
  class MemoryRestrictedForest {
  public:
    explicit MemoryRestrictedForest(const SerialTreeLearner *tree_learner)
      : init_(false), tree_learner_(tree_learner) {
    }
    void InsertLeafInformation(double leaf_value) {
      bool present = false;
      for (feature_info feature : featureinfo_used_global_) {
        if (feature.feature_id == -1) {
          for (double threshold : feature.thresholds) {
            if (leaf_value == threshold) {
              present = true;
              break;
      }}}}
      if (!present) {
        // value in the threshold table.
        est_leftover_memory -= 8;
        thresholds_used_global_.push_back(leaf_value);
        int fug_size_ = 0;
        for (feature_info feature : featureinfo_used_global_) {
          if (feature.feature_id == -1) {
            feature.thresholds.push_back(leaf_value);
            fug_size_ = feature.thresholds.size();
          }
        }
        size_t next_power_of_two = static_cast<size_t>(std::pow(2, std::ceil(std::log2(fug_size_ + 1))));
        if (fug_size_ + 1 > next_power_of_two) {
          est_leftover_memory -= static_cast<int>(std::pow(2, max_depth));
        }
      }
    }

    void UpdateMemoryForTree(Tree* tree) {
      Log::Info("Update Memory For Tree %d %d", treecounter, est_leftover_memory);
      treecounter++;
      ref_trees_.push_back({});
      ref_trees_[treecounter].tree_id = treecounter;
      tree_size_.push_back(tree->getNumberNodes());
    }
    void InsertSplitInfo(const Tree *tree, const Dataset *train_data_) {
      const int last_node_id = tree->num_leaves_ - 2;
      // TODO: merge following lines, but depends on if/how we implement rounding.
      const double threshold = RoundDecimals(tree->threshold_[last_node_id], this->precision);
      // printf("original threshold: %f, rounded threshold: %f \n", tree->threshold_[last_node_id], threshold);
      const uint32_t feature = tree->split_feature_[last_node_id];
      const BinMapper *bin_mapper = train_data_->FeatureBinMapper(feature);
      consumed_memory con_mem = {};
      CalculateSplitMemoryConsumption(con_mem, threshold, feature, bin_mapper);
      // Trees have duplicates so we can count how often it is used.
      ref_trees_[treecounter].feature_ids.push_back(con_mem.findex);
      ref_trees_[treecounter].thresholds.push_back(threshold);

      if (con_mem.new_feature) {
        // If we have a new feature it has a single threshold it references.
        feature_info feature_struct = {(int)feature, con_mem.feature_bits, 1};
        features_used_global_.push_back(feature);
        featureinfo_used_global_.push_back(feature_struct);
      }
      if (con_mem.new_threshold) {
        // If yes, only one double value is added.
        thresholds_used_global_.push_back(threshold);
        for (feature_thresholds feature_thresholds_info : thresholds_per_feature_) {
          if (feature_thresholds_info.feature_id == feature) {
            feature_thresholds_info.thresholds.push_back(threshold);
          }
        }
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
    double CalculateSplitMemoryConsumption(consumed_memory &con_mem, double threshold, uint32_t feature, const BinMapper *bin_mapper) {
      // Insert the memory consumption of the two global tables.
      std::vector<uint32_t>::iterator feature_it;
      // threshold = RoundDecimals(threshold, this->precision);
      std::vector<double>::iterator threshold_it = std::find(thresholds_used_global_.begin(),
                                                            thresholds_used_global_.end(), threshold);
      feature_it = std::find(features_used_global_.begin(), features_used_global_.end(), feature);
      // In case the feature is not used 8 bits are added for representing a bits_single and bits_ref.
      int bits_needed = 0;
      if (feature_it == features_used_global_.end()) {
        std::vector<double> bin_bounds = bin_mapper->getBinUpperBound();
        if (isAllBool(bin_bounds)) {
          bits_needed = 1;
        }
        if (isAllBool(bin_bounds)) {
          bits_needed = 1;
        } else if (isAllInteger(bin_bounds)) {
          int range = bin_mapper->getMaxVal() - bin_mapper->getMinVal();
          bits_needed = static_cast<int>(std::ceil(std::log2(std::abs(range) + 1)));
        } else {
          bits_needed = 8;
        }
        con_mem.bits += bits_needed;
        con_mem.new_feature = true;
        // If the new feature results in an increase of a 2 power we need more bits to reference them in each tree
        size_t fug_size_ = featureinfo_used_global_.size();
        size_t next_power_of_two = static_cast<size_t>(std::pow(2, std::ceil(std::log2(fug_size_ + 1))));
        if (fug_size_ + 2 > next_power_of_two) {
          int feature_counter = 0;
          for (ref_tree tree : ref_trees_) {
            for (int feature_id : tree.feature_ids) {
              if (feature_id == feature) {
                feature_counter++;
          }}}
          con_mem.bits += feature_counter;
          // Add one bit for every time a feature is referenced in a tree.
        }
        con_mem.feature_bits = bits_needed;
      } else {
        con_mem.findex = std::distance(features_used_global_.begin(), feature_it);
      }
      if (threshold_it == thresholds_used_global_.end()) {
        bool found_for_feature = false;
        int reference_size = 0;
        for (feature_thresholds threshold_info : thresholds_per_feature_) {
          if (threshold_info.feature_id == feature) {
            found_for_feature = true;
            reference_size = threshold_info.thresholds.size();
          }
        }
        if (!found_for_feature) {
          size_t next_power_of_two = static_cast<size_t>(std::pow(2, std::ceil(std::log2(reference_size + 1))));
          if (reference_size + 1 > next_power_of_two) {
            // TODO every entry in every tree requires as threshold reference + 1 bit
            // con_mem.bits += ?;
          }
        }
        for (feature_info feature_info : featureinfo_used_global_) {
          if (feature_info.feature_id == feature) {
            bits_needed = feature_info.bits_needed;
          }
        }
        con_mem.bits += bits_needed;
        con_mem.new_threshold = true;
      } else {
        con_mem.tindex = std::distance(thresholds_used_global_.begin(), threshold_it);
      }
      return threshold;
    }

    double RoundDecimals(double number, double decimals) {
      // printf("input number %f ", number);
      double rounded = ((double)((int)(number * pow(10.0, decimals) + .5))) / pow(10.0, decimals);
      // printf("rounded number %f \n", rounded);
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
    }
    void printForest() {
      std::stringstream out;
      out << "Leftover memory: " << est_leftover_memory;
      out << "\nThresholds used: ";
      for (double threshold : thresholds_used_global_) {
        out << threshold << " ";
      }
      out << "\nFeatures used: ";
      for (int feature : features_used_global_) {
        out << feature << " ";
      }
      out << "\nTree Information collected: ";
      for (ref_tree refe_tree : ref_trees_) {
        out << refe_tree << "; ";
      }
      out << "\nFeature Information collected: ";
      for (feature_info feature : featureinfo_used_global_) {
        out << feature << "; ";
      }
      out << "\n";
      std::cout << out.str();
    }
    bool init_;
    int est_leftover_memory{};
    int max_depth;
    // TODO: choose between E5M2 range(+/-57344)/E4M3 range(+/-448) -- https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
    double precision;
    const SerialTreeLearner *tree_learner_;
    /*! \brief count feature use; */
    /*! \brief record thresholds used for split; */
    std::vector<double> thresholds_used_global_;
    std::vector<int> tree_size_;
    /* \brief Arrays to save information for Prefix Sum Tree structure */
    std::vector<feature_thresholds> thresholds_per_feature_;
    std::vector<uint32_t> features_used_global_;
    std::vector<feature_info> featureinfo_used_global_;
    std::vector<ref_tree> ref_trees_;
    int treecounter = 0;
  };
}
#endif //LIGHTGBM_MEMORY_RESTRICTED_FOREST_H
