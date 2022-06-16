/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <testutils.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/c_api.h>
#include <LightGBM/dataset.h>

#include <iostream>

using LightGBM::Dataset;
using LightGBM::Log;
using LightGBM::TestUtils;

void test_stream_dense(DatasetHandle ref_datset_handle,
  int32_t nrows,
  int32_t ncols,
  int32_t nclasses,
  int batch_count,
  const std::vector<double>* features,
  const std::vector<float>* labels,
  const std::vector<float>* weights,
  const std::vector<double>* init_scores,
  const std::vector<int32_t>* groups) {
  Log::Info("Streaming %d rows dense data with a batch size of %d", nrows, batch_count);
  DatasetHandle dataset_handle;

  try {
    int result = LGBM_DatasetCreateByReference(ref_datset_handle, nrows, &dataset_handle);
    EXPECT_EQ(0, result) << "LGBM_DatasetCreateByReference result code: " << result;

    Dataset* dataset = static_cast<Dataset*>(dataset_handle);

    TestUtils::StreamDenseDataset(dataset_handle,
      nrows,
      ncols,
      nclasses,
      batch_count,
      features,
      labels,
      weights,
      init_scores,
      groups);

    dataset->FinishLoad();

    // TODO(svotaw) we should assert actual feature data, but we would need to calculate bin values

    TestUtils::AssertMetadata(&dataset->metadata(),
      labels,
      weights,
      init_scores,
      groups);
  }
  catch (...) {
  }

  if (dataset_handle) {
    int result = LGBM_DatasetFree(dataset_handle);
    EXPECT_EQ(0, result) << "LGBM_DatasetFree result code: " << result;
  }
}

void test_stream_sparse(DatasetHandle ref_datset_handle,
  int32_t nrows,
  int32_t nclasses,
  int batch_count,
  const std::vector<int32_t>* indptr,
  const std::vector<int32_t>* indices,
  const std::vector<double>* vals,
  const std::vector<float>* labels,
  const std::vector<float>* weights,
  const std::vector<double>* init_scores,
  const std::vector<int32_t>* groups) {
  Log::Info("Streaming %d rows sparse data with a batch size of %d", nrows, batch_count);
  DatasetHandle dataset_handle;

  try {
    int result = LGBM_DatasetCreateByReference(ref_datset_handle, nrows, &dataset_handle);
    EXPECT_EQ(0, result) << "LGBM_DatasetCreateByReference result code: " << result;

    Dataset* dataset = static_cast<Dataset*>(dataset_handle);

    TestUtils::StreamSparseDataset(dataset_handle,
      nrows,
      nclasses,
      batch_count,
      indptr,
      indices,
      vals,
      labels,
      weights,
      init_scores,
      groups);

    dataset->FinishLoad();

    // TODO(svotaw) we should assert actual feature data

    TestUtils::AssertMetadata(&dataset->metadata(),
      labels,
      weights,
      init_scores,
      groups);
  }
  catch (...) {
  }

  if (dataset_handle) {
    int result = LGBM_DatasetFree(dataset_handle);
    EXPECT_EQ(0, result) << "LGBM_DatasetFree result code: " << result;
  }
}

TEST(Stream, PushDenseRowsWithMetadata) {
  // Load some test data
  DatasetHandle ref_datset_handle;
  const char* params = "max_bin=15";
  // Use the smaller ".test" data because we don't care about the actual data and it's smaller
  int result = TestUtils::LoadDatasetFromExamples("binary_classification/binary.test", params, &ref_datset_handle);
  EXPECT_EQ(0, result) << "LoadDatasetFromExamples result code: " << result;

  Dataset* ref_dataset = static_cast<Dataset*>(ref_datset_handle);
  auto noriginalrows = ref_dataset->num_data();
  Log::Info("Row count: %d", noriginalrows);
  Log::Info("Feature group count: %d", noriginalrows);

  // Add some fake initial_scores and groups so we can test streaming them
  int nclasses = 2;  // choose > 1 just to test multi-class handling
  std::vector<double> unused_init_scores;
  unused_init_scores.resize(noriginalrows * nclasses);
  std::vector<int32_t> unused_groups;
  unused_groups.assign(noriginalrows, 1);
  result = LGBM_DatasetSetField(ref_datset_handle, "init_score", unused_init_scores.data(), noriginalrows * nclasses, 1);
  EXPECT_EQ(0, result) << "LGBM_DatasetSetField init_score result code: " << result;
  result = LGBM_DatasetSetField(ref_datset_handle, "group", unused_groups.data(), noriginalrows, 2);
  EXPECT_EQ(0, result) << "LGBM_DatasetSetField group result code: " << result;

  // Now use the reference dataset schema to make some testable Datasets with N rows each
  int32_t nrows = 100;
  int32_t ncols = ref_dataset->num_features();
  std::vector<double> features;
  std::vector<float> labels;
  std::vector<float> weights;
  std::vector<double> init_scores;
  std::vector<int32_t> groups;

  Log::Info("Creating random data");
  TestUtils::CreateRandomDenseData(nrows, ncols, nclasses, &features, &labels, &weights, &init_scores, &groups);

  int32_t batch_count = 1;
  test_stream_dense(ref_datset_handle, nrows, ncols, nclasses, batch_count, &features, &labels, &weights, &init_scores, &groups);

  batch_count = nrows / 100;
  test_stream_dense(ref_datset_handle, nrows, ncols, nclasses, batch_count, &features, &labels, &weights, &init_scores, &groups);

  batch_count = nrows / 10;
  test_stream_dense(ref_datset_handle, nrows, ncols, nclasses, batch_count, &features, &labels, &weights, &init_scores, &groups);

  batch_count = nrows;
  test_stream_dense(ref_datset_handle, nrows, ncols, nclasses, batch_count, &features, &labels, &weights, &init_scores, &groups);

  result = LGBM_DatasetFree(ref_datset_handle);
  EXPECT_EQ(0, result) << "LGBM_DatasetFree result code: " << result;
}


TEST(Stream, PushSparseRowsWithMetadata) {
  // Load some test data
  DatasetHandle ref_datset_handle;
  const char* params = "max_bin=15";
  // Use the smaller ".test" data because we don't care about the actual data and it's smaller
  int result = TestUtils::LoadDatasetFromExamples("binary_classification/binary.test", params, &ref_datset_handle);
  EXPECT_EQ(0, result) << "LoadDatasetFromExamples result code: " << result;

  Dataset* ref_dataset = static_cast<Dataset*>(ref_datset_handle);
  auto noriginalrows = ref_dataset->num_data();
  Log::Info("Row count: %d", noriginalrows);
  Log::Info("Feature group count: %d", noriginalrows);

  // Add some fake initial_scores and groups so we can test streaming them
  int32_t nclasses = 2;
  std::vector<double> unused_init_scores;
  unused_init_scores.resize(noriginalrows * nclasses);
  std::vector<int32_t> unused_groups;
  unused_groups.assign(noriginalrows, 1);
  result = LGBM_DatasetSetField(ref_datset_handle, "init_score", unused_init_scores.data(), noriginalrows * nclasses, 1);
  EXPECT_EQ(0, result) << "LGBM_DatasetSetField result code: " << result;
  result = LGBM_DatasetSetField(ref_datset_handle, "group", unused_groups.data(), noriginalrows, 2);
  EXPECT_EQ(0, result) << "LGBM_DatasetSetField result code: " << result;

  // Now use the reference dataset schema to make some testable Datasets with N rows each
  int32_t nrows = 100;
  int32_t ncols = ref_dataset->num_features();
  std::vector<int32_t> indptr;
  std::vector<int32_t> indices;
  std::vector<double> vals;
  std::vector<float> labels;
  std::vector<float> weights;
  std::vector<double> init_scores;
  std::vector<int32_t> groups;

  Log::Info("Creating random data");
  float sparse_percent = .1f;
  TestUtils::CreateRandomSparseData(nrows, ncols, nclasses, sparse_percent, &indptr, &indices, &vals, &labels, &weights, &init_scores, &groups);

  int32_t batch_count = 1;
  test_stream_sparse(ref_datset_handle, nrows, nclasses, batch_count, &indptr, &indices, &vals, &labels, &weights, &init_scores, &groups);

  batch_count = nrows / 100;
  test_stream_sparse(ref_datset_handle, nrows, nclasses, batch_count, &indptr, &indices, &vals, &labels, &weights, &init_scores, &groups);

  batch_count = nrows / 10;
  test_stream_sparse(ref_datset_handle, nrows, nclasses, batch_count, &indptr, &indices, &vals, &labels, &weights, &init_scores, &groups);

  batch_count = nrows;
  test_stream_sparse(ref_datset_handle, nrows, nclasses, batch_count, &indptr, &indices, &vals, &labels, &weights, &init_scores, &groups);

  result = LGBM_DatasetFree(ref_datset_handle);
  EXPECT_EQ(0, result) << "LGBM_DatasetFree result code: " << result;
}
