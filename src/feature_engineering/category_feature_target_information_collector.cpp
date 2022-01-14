/*!
* Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Licensed under the MIT License. See LICENSE file in the project root for license information.
*/

#include <algorithm>

#include "category_feature_encoder.hpp"

namespace LightGBM {
  void CategoryFeatureTargetInformationCollector::HandleRecord(int fold_id, const std::vector<double>& record, double label) {
    std::unordered_map<int, CategoryFeatureTargetInformation> category_target_information_records = category_target_information_[fold_id];

    for (auto iterator = categorical_features_.begin(); iterator != categorical_features_.end(); iterator++ ) {
      int feature_id = *iterator;
      int category = static_cast<int>(record[feature_id]);

	  CategoryFeatureTargetInformation category_target_information_record = category_target_information_records[feature_id];
      category_target_information_record.category_count[category] += 1;
      category_target_information_record.category_label_sum[category] += label;
    }

    count_[fold_id] += 1;
    label_sum_[fold_id] += label;
  }

  void CategoryFeatureTargetInformationCollector::AppendFrom(CategoryFeatureTargetInformationCollector& collector) {
    std::vector<data_size_t> target_count_record = collector.GetCounts();
    count_.reserve(count_.size() + target_count_record.size());
    count_.insert(count_.end(), target_count_record.begin(), target_count_record.end());

    std::vector<double> target_sum_record = collector.GetLabelSum();
    label_sum_.reserve(label_sum_.size() + target_sum_record.size());
    label_sum_.insert(label_sum_.end(), target_sum_record.begin(), target_sum_record.end());

    std::vector<std::unordered_map<int, CategoryFeatureTargetInformation>> target_category_target_information = collector.GetCategoryTargetInformation();
    category_target_information_.reserve(category_target_information_.size() + target_category_target_information.size());
    category_target_information_.insert(category_target_information_.end(), target_category_target_information.begin(), target_category_target_information.end());
  }
}
