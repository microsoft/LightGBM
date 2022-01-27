/*!
* Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Licensed under the MIT License. See LICENSE file in the project root for license information.
*/

#include <algorithm>

#include "category_feature_encoder.hpp"

namespace LightGBM {
  void CategoryFeatureTargetInformationCollector::HandleRecord(int fold_id, const std::vector<double>& record, double label) {
    std::unordered_map<int, CategoryFeatureTargetInformation>& category_target_information_records = category_target_information_[fold_id];

    for (auto iterator = categorical_features_.begin(); iterator != categorical_features_.end(); iterator++) {
      int feature_id = *iterator;
      int category = static_cast<int>(record[feature_id]);

      CategoryFeatureTargetInformation& category_target_information_record = category_target_information_records[feature_id];
      category_target_information_record.category_count[category] += 1;
      category_target_information_record.category_label_sum[category] += label;
      category_target_information_record.total_count += 1;
      category_target_information_record.label_sum += label;

      CategoryFeatureTargetInformation& global_category_target_information_record = global_category_target_information_[feature_id];
      global_category_target_information_record.category_count[category] += 1;
      global_category_target_information_record.category_label_sum[category] += label;
      global_category_target_information_record.total_count += 1;
      global_category_target_information_record.label_sum += label;
    }

    count_[fold_id] += 1;
    label_sum_[fold_id] += label;
  }

  void CategoryFeatureTargetInformationCollector::AppendFrom(const CategoryFeatureTargetInformationCollector& collector) {
    const std::vector<data_size_t>& target_count_record = collector.GetCounts();
    count_.reserve(count_.size() + target_count_record.size());
    count_.insert(count_.end(), target_count_record.begin(), target_count_record.end());

	const std::vector<double>& target_sum_record = collector.GetLabelSum();
    label_sum_.reserve(label_sum_.size() + target_sum_record.size());
    label_sum_.insert(label_sum_.end(), target_sum_record.begin(), target_sum_record.end());

	const std::vector<std::unordered_map<int, CategoryFeatureTargetInformation>>& target_category_target_information = collector.GetCategoryTargetInformation();
    for (auto& entry : target_category_target_information) {
      category_target_information_.push_back(entry);
    }

	const std::unordered_map<int, CategoryFeatureTargetInformation>& global_category_target_information_record = collector.GetGlobalCategoryTargetInformation();
    for (auto& feature_information : global_category_target_information_record) {
      for (auto& category_count : feature_information.second.category_count) {
        global_category_target_information_[feature_information.first].category_count[category_count.first] += category_count.second;
      }

      for (auto& label_sum : feature_information.second.category_label_sum) {
        global_category_target_information_[feature_information.first].category_label_sum[label_sum.first] += label_sum.second;
      }

      global_category_target_information_[feature_information.first].total_count += global_category_target_information_record.at(feature_information.first).total_count;
      global_category_target_information_[feature_information.first].label_sum += global_category_target_information_record.at(feature_information.first).label_sum;
    }
  }
}  // namespace LightGBM
