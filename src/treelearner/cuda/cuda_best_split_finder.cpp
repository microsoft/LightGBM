/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "cuda_histogram_constructor.hpp"

namespace LightGBM {

CUDABestSplitFinder::CUDABestSplitFinder(const hist_t* /*cuda_hist*/, const Dataset* /*train_data*/,
    const std::vector<int>& /*feature_group_ids*/, const int /*max_num_leaves*/) {}

void CUDABestSplitFinder::FindBestSplitsForLeaf(const int* /*leaf_id*/) {}

void CUDABestSplitFinder::FindBestFromAllSplits() {}

}  // namespace LightGBM
