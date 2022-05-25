/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <string>
#include <gtest/gtest.h>

#include <testutils.h>
#include <LightGBM/c_api.h>
#include <LightGBM/utils/random.h>

using LightGBM::Log;
using LightGBM::Random;

namespace LightGBM {

/*!
* Creates a Dataset from the internal repository examples.
*/
int TestUtils::LoadDatasetFromExamples(const char* filename, const char* config, DatasetHandle *out) {
  std::string fullPath("..\\examples\\");
  fullPath += filename;
  // TODO check file exists
  return LGBM_DatasetCreateFromFile(
    fullPath.c_str(),
    config,
    nullptr,
    out);

    /*print(LIB.LGBM_GetLastError())
    num_data = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    print(f'#data: {num_data.value} #feature: {num_feature.value}')
    return handle*/
}

/*!
* Creates fake data in the passed vectors.
*/
void TestUtils::CreateRandomDenseData(int32_t nrows,
                                      int32_t ncols,
                                      std::vector<double> *features,
                                      std::vector<float> *labels,
                                      std::vector<float> *weights,
                                      std::vector<double> *init_scores,
                                      std::vector<int32_t> *groups) {
  Random rand(42);
  features->reserve(nrows * ncols);

  for (int32_t row = 0; row < nrows; row++)
  {
    for (int32_t col = 0; col < ncols; col++)
    {
      features->push_back(rand.NextFloat());
    }
  }

  CreateRandomMetadata(nrows, labels, weights, init_scores, groups);
}


/*!
* Creates fake data in the passed vectors.
*/
void TestUtils::CreateRandomSparseData(int32_t nrows,
                                       int32_t ncols,
                                       float sparse_percent,
                                       std::vector<int32_t>* indptr,
                                       std::vector<int32_t>* indices,
                                       std::vector<double>* values,
                                       std::vector<float>* labels,
                                       std::vector<float>* weights,
                                       std::vector<double>* init_scores,
                                       std::vector<int32_t>* groups) {
  Random rand(42);
  indptr->reserve(static_cast<int32_t>(sparse_percent * nrows * ncols + 1));
  indices->reserve(static_cast<int32_t>(sparse_percent * nrows * ncols));
  values->reserve(static_cast<int32_t>(sparse_percent * nrows * ncols));

  indptr->push_back(0);
  for (int32_t row = 0; row < nrows; row++)
  {
    for (int32_t col = 0; col < ncols; col++)
    {
      float rnd = rand.NextFloat();
      if (rnd < sparse_percent)
      {
        indices->push_back(ncols * row + col);
        values->push_back(rand.NextFloat());
      }
    }
    indptr->push_back(static_cast<int32_t>(indices->size() - 1));
  }

  CreateRandomMetadata(nrows, labels, weights, init_scores, groups);
}

/*!
* Creates fake data in the passed vectors.
*/
void TestUtils::CreateRandomMetadata(int32_t nrows,
                                     std::vector<float>* labels,
                                     std::vector<float>* weights,
                                     std::vector<double>* init_scores,
                                     std::vector<int32_t>* groups) {
  Random rand(42);
  labels->reserve(nrows);
  if (weights) {
    weights->reserve(nrows);
  }
  if (init_scores) {
    init_scores->reserve(nrows);
  }
  if (groups) {
    weights->reserve(nrows);
  }

  int32_t group = 0;

  for (int32_t row = 0; row < nrows; row++)
  {
    labels->push_back(rand.NextFloat());
    if (weights) {
      weights->push_back(rand.NextFloat());
    }
    if (init_scores) {
      init_scores->push_back(rand.NextFloat());
    }
    if (groups) {
      if (rand.NextFloat() > 0.95) {
        group++;
      }
      groups->push_back(group);
    }
  }
}

void TestUtils::StreamDenseDataset(DatasetHandle dataset_handle,
                                   int32_t nrows,
                                   int32_t ncols,
                                   int32_t batch_count,
                                   const std::vector<double> *features,
                                   const std::vector<float> *labels,
                                   const std::vector<float> *weights,
                                   const std::vector<double> *init_scores,
                                   const std::vector<int32_t> *groups) {

  const double* features_ptr = features->data();
  const float* labels_ptr = labels->data();
  const float* weights_ptr = nullptr;
  if (weights) {
    weights_ptr = weights->data();
  }
  const double* init_scores_ptr = nullptr;
  if (init_scores) {
    init_scores_ptr = init_scores->data();
  }
  const int32_t* groups_ptr = nullptr;
  if (groups) {
    groups_ptr = groups->data();
  }

  auto start_time = std::chrono::steady_clock::now();

  for (int32_t i = 0; i < nrows; i += batch_count)
  {
    int result = LGBM_DatasetPushRowsWithMetadata(dataset_handle,
                                                  features_ptr,
                                                  1,
                                                  batch_count,
                                                  ncols,
                                                  i,
                                                  labels_ptr,
                                                  weights_ptr,
                                                  init_scores_ptr,
                                                  groups_ptr);
    EXPECT_EQ(0, result) << "LGBM_DatasetPushRowsWithMetadata result code: " << result;

    features_ptr += batch_count * ncols;
    labels_ptr += batch_count;
    if (weights_ptr) {
      weights_ptr += batch_count;
    }
    if (init_scores_ptr) {
      init_scores_ptr += batch_count;
    }
    if (groups_ptr) {
      groups_ptr += batch_count;
    }
  }

  int result = LGBM_DatasetFinishStreaming(dataset_handle);
  EXPECT_EQ(0, result) << "LGBM_DatasetFinishStreaming result code: " << result;

  auto cur_time = std::chrono::steady_clock::now();
  Log::Info(" Time: %d", cur_time - start_time);

}

void TestUtils::StreamSparseDataset(DatasetHandle dataset_handle,
                                    int32_t nrows,
                                    int32_t batch_count,
                                    const std::vector<int32_t> *indptr,
                                    const std::vector<int32_t> *indices,
                                    const std::vector<double> *values,
                                    const std::vector<float> *labels,
                                    const std::vector<float> *weights,
                                    const std::vector<double> *init_scores,
                                    const std::vector<int32_t> *groups) {

  const int32_t* indptr_ptr = indptr->data();
  const int32_t* indices_ptr = indices->data();
  const double* values_ptr = values->data();
  const float* labels_ptr = labels->data();
  const float* weights_ptr = nullptr;
  if (weights) {
    weights_ptr = weights->data();
  }
  const double* init_scores_ptr = nullptr;
  if (init_scores) {
    init_scores_ptr = init_scores->data();
  }
  const int32_t* groups_ptr = nullptr;
  if (groups) {
    groups_ptr = groups->data();
  }

  for (int32_t i = 0; i < nrows; i += batch_count)
  {
    int32_t nelem = indptr->at(i + batch_count - 1) - indptr->at(i);

    int result = LGBM_DatasetPushRowsByCSRWithMetadata(dataset_handle,
                                                       indptr,
                                                       2,
                                                       indices_ptr,
                                                       values_ptr,
                                                       1,
                                                       batch_count + 1,
                                                       nelem,
                                                       i,
                                                       labels_ptr,
                                                       weights_ptr,
                                                       init_scores_ptr,
                                                       groups_ptr);
    EXPECT_EQ(0, result) << "LGBM_DatasetPushRowsByCSRWithMetadata result code: " << result;

    indptr_ptr += batch_count;
    indices_ptr += nelem;
    values_ptr += nelem;
    labels_ptr += batch_count;
    if (weights_ptr) {
      weights_ptr += batch_count;
    }
    if (init_scores_ptr) {
      init_scores_ptr += batch_count;
    }
    if (groups_ptr) {
      groups_ptr += batch_count;
    }
  }
}

void TestUtils::AssertMetadata(const Metadata* metadata,
                               const std::vector<float>* ref_labels,
                               const std::vector<float>* ref_weights,
                               const std::vector<double>* ref_init_scores,
                               const std::vector<int32_t>* ref_groups) {

  const float* labels = metadata->label();
  auto nTotal = static_cast<int32_t>(ref_labels->size());
  for (auto i = 0; i < nTotal; i++)
  {
    EXPECT_EQ(ref_labels->at(i), labels[i]) << "Coalesced data: " << ref_labels->at(i);
  }

  const float* weights = metadata->weights();
  if (weights)
  {
    if (!ref_weights) {
      FAIL() << "Expected null weights";
    }
    for (auto i = 0; i < nTotal; i++)
    {
      EXPECT_EQ(ref_weights->at(i), weights[i]) << "Coalesced data: " << ref_weights->at(i);
    }
  } else if (ref_weights) {
    FAIL() << "Expected non-null weights";
  }

  const double* init_scores = metadata->init_score();
  if (init_scores)
  {
    if (!ref_init_scores) {
      FAIL() << "Expected null init_scores";
    }
    for (auto i = 0; i < nTotal; i++)
    {
      EXPECT_EQ(ref_init_scores->at(i), init_scores[i]) << "Coalesced data: " << ref_init_scores->at(i);
    }
  } else if (ref_init_scores) {
    FAIL() << "Expected non-null init_scores";
  }

  const int32_t* query_boundaries = metadata->query_boundaries();
  if (query_boundaries) {
    if (!ref_groups) {
      FAIL() << "Expected null query_boundaries";
    }
    // Calculate expected boundaries
    std::vector<int32_t> ref_query_boundaries;
    ref_query_boundaries.push_back(0);
    int group_val = ref_groups->at(0);
    for (auto i = 1; i < nTotal; i++)
    {
      if (ref_groups->at(i) != group_val) {
        ref_query_boundaries.push_back(i);
        group_val = ref_groups->at(i);
      }
    }
    ref_query_boundaries.push_back(nTotal);

    for (auto i = 0; i < ref_query_boundaries.size(); i++)
    {
      EXPECT_EQ(ref_query_boundaries[i], query_boundaries[i]) << "Coalesced data group: " << ref_query_boundaries[i];
    }
  } else if (ref_groups) {
    FAIL() << "Expected non-null query_boundaries";
  }
}
} // LightGBM
