
/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_PREDICTION_CONTROL_PARAMETER_H_
#define LIGHTGBM_PREDICTION_CONTROL_PARAMETER_H_


#include <vector>
#include <algorithm>

namespace LightGBM {

/*!
* \brief Control paramters for prediction, used to implement variants of prediction algorithm
*/
struct PredictionControlParameter {
  public:
    PredictionControlParameter() {}
    PredictionControlParameter(const std::vector<int>& ra_features) : random_assign_features(ra_features) {
      // std::stable_sort(random_assign_features.begin(), random_assign_features.end(), std::less<int>());
    }

    /*!
    * \brief try to enable random assignment mechanism with the split feature of current node
    * \param split_feat_idx real index of the split feature
    */
    inline bool EnableRandomAssign(int split_feat_idx) const {
      // return std::binary_search(random_assign_features.begin(), random_assign_features.end(), split_feat_idx);
      return (std::find(random_assign_features.begin(), random_assign_features.end(), split_feat_idx) 
              != random_assign_features.end());
    }

    std::vector<int> random_assign_features;
};

}   // namespace LightGBM

#endif  // LIGHTGBM_PREDICTION_CONTROL_PARAMETER_H_
