/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <testutils.h>
#include <LightGBM/utils/byte_buffer.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/c_api.h>
#include <LightGBM/bin.h>
#include <LightGBM/dataset.h>

#include <iostream>
#include <vector>

using LightGBM::Bin;
using LightGBM::BinIterator;
using LightGBM::Dataset;
using LightGBM::Log;
using LightGBM::Metadata;
using LightGBM::TestUtils;
using namespace std;

void test_dense_insertion(int num_bin, const std::vector<uint8_t>* dataset1, const std::vector<uint8_t>* dataset2) {
  std::unique_ptr<Bin> denseBin1;
  std::unique_ptr<Bin> denseBin2;

  denseBin1.reset(Bin::CreateDenseBin(10, num_bin));
  denseBin2.reset(Bin::CreateDenseBin(10, num_bin));

  Log::Info("Loading source datasets...");

  int32_t nDataset1 = static_cast<int32_t>(dataset1->size());
  int32_t nDataset2 = static_cast<int32_t>(dataset2->size());
  auto nTotal = nDataset1 + nDataset2;
  for (int i = 0; i < nDataset1; ++i) {
    denseBin1->Push(0, i, dataset1->at(i));
  }
  for (int i = 0; i < nDataset2; ++i) {
    denseBin2->Push(0, i, dataset2->at(i));
  }

  std::unique_ptr<Bin> denseBin_coalesced;
  denseBin_coalesced.reset(Bin::CreateDenseBin(nTotal, num_bin));
  Log::Info("Inserting from bin1...");
  denseBin_coalesced->InsertFrom(denseBin1.get(), 0, nDataset1);
  Log::Info("Inserting from bin2...");
  denseBin_coalesced->InsertFrom(denseBin2.get(), nDataset1, nDataset2);
  Log::Info("Finish loading...");
  denseBin_coalesced->FinishLoad();

  EXPECT_EQ(nTotal, denseBin_coalesced->num_data()) << "Result dataset size: " << nTotal;

  std::unique_ptr<BinIterator> iterator;
  iterator.reset(denseBin_coalesced->GetIterator(1, 16, 0));
  iterator->Reset(0);

  // Now assert the resulting iterator values from the coalesced data match the original arrays
  std::vector<uint8_t> coalesced_data;
  coalesced_data.reserve(nDataset1 + nDataset2);
  coalesced_data.insert(coalesced_data.end(), dataset1->begin(), dataset1->end());
  coalesced_data.insert(coalesced_data.end(), dataset2->begin(), dataset2->end());
  for (auto i = 0; i < nTotal; i++)
  {
    EXPECT_EQ(coalesced_data[i], iterator->Get(i)) << "Coalesced data: " << coalesced_data[i];
    Log::Info("Row value: %d", iterator->Get(i));
  }
}

void test_sparse_insertion(int32_t count1,
                           const std::vector<int32_t>* dataset1_ind,
                           const std::vector<uint8_t>* dataset1_val,
                           int32_t count2,
                           const std::vector<int32_t>* dataset2_ind,
                           const std::vector<uint8_t>* dataset2_val) {
  std::unique_ptr<Bin> sparseBin1;
  std::unique_ptr<Bin> sparseBin2;

  sparseBin1.reset(Bin::CreateSparseBin(count1, 256));
  sparseBin2.reset(Bin::CreateSparseBin(count2, 256));

  Log::Info("  Loading source datasets...");
  int32_t nDataset1 = static_cast<int32_t>(dataset1_ind->size());
  int32_t nDataset2 = static_cast<int32_t>(dataset2_ind->size());
  Log::Info("  Loading source dataset1...");
  for (int i = 0; i < nDataset1; ++i) {
    Log::Info("  Pushing %d...", dataset1_val->at(i));
    sparseBin1->Push(0, dataset1_ind->at(i), dataset1_val->at(i));
  }
  Log::Info("  Loading source dataset2...");
  for (int i = 0; i < nDataset2; ++i) {
    sparseBin2->Push(0, dataset2_ind->at(i), dataset2_val->at(i));
  }

  std::unique_ptr<Bin> sparseBin_coalesced;
  sparseBin_coalesced.reset(Bin::CreateSparseBin(count1 + count2, 256));
  Log::Info("  Inserting from bin1...");
  sparseBin_coalesced->InsertFrom(sparseBin1.get(), 0, count1);
  Log::Info("  Inserting from bin2...");
  sparseBin_coalesced->InsertFrom(sparseBin2.get(), count1, count2);
  Log::Info("  Finish loading...");
  sparseBin_coalesced->FinishLoad();

  EXPECT_EQ(count1 + count2, sparseBin_coalesced->num_data()) << "Result dataset size: " << count1 + count2;

  // Now assert the resulting iterator values from the coalesced data match the original arrays
  std::vector<uint8_t> coalesced_data_val;
  std::vector<int32_t> coalesced_data_ind;
  coalesced_data_val.reserve(nDataset1 + nDataset2);
  coalesced_data_ind.reserve(nDataset1 + nDataset2);
  coalesced_data_val.insert(coalesced_data_val.end(), dataset1_val->begin(), dataset1_val->end());
  coalesced_data_val.insert(coalesced_data_val.end(), dataset2_val->begin(), dataset2_val->end());
  coalesced_data_ind.insert(coalesced_data_ind.end(), dataset1_ind->begin(), dataset1_ind->end());
  for (auto i = 0; i < nDataset2; i++)
  {
    coalesced_data_ind.push_back(dataset2_ind->at(i) + count1);
  }
  std::unique_ptr<BinIterator> iterator;
  iterator.reset(sparseBin_coalesced->GetIterator(1, 16, 0));
  for (auto i = 0; i < nDataset1 + nDataset2; i++)
  {
    auto val = iterator->Get(coalesced_data_ind.at(i));
    EXPECT_EQ(coalesced_data_val[i], val) << "Coalesced data: " << coalesced_data_val[i];
    Log::Info("  Row value: %d", val);
  }
}

void init_metadata(Metadata* metadata,
                   int32_t size,
                   const std::vector<float>* dataset_weight,
                   const std::vector<double>* dataset_init_score,
                   const std::vector<uint8_t>* dataset_group) {
  Log::Info("    Initializing dataset...");
  int weight_idx = -1;
  int init_score_idx = -1;
  int group_idx = -1;
  if (dataset_weight != nullptr) {
    weight_idx = 0;
  }
  if (dataset_init_score != nullptr) {
    init_score_idx = 0;
  }
  if (dataset_group != nullptr) {
    group_idx = 0;
  }
  metadata->Init(size, weight_idx, group_idx);
  if (dataset_init_score) {
    metadata->InitInitScore();
  }
}

void load_metadata(Metadata* metadata,
                   const std::vector<float>* dataset_label,
                   const std::vector<float>* dataset_weight,
                   const std::vector<double>* dataset_init_score,
                   const std::vector<uint8_t>* dataset_group) {
  init_metadata(metadata, static_cast<int32_t>(dataset_label->size()), dataset_weight, dataset_init_score, dataset_group);

  Log::Info("    Loading label...");
  for (int i = 0; i < dataset_label->size(); ++i) {
    metadata->SetLabelAt(i, dataset_label->at(i));
  }
  if (dataset_weight) {
    Log::Info("    Loading weight...");
    for (int i = 0; i < dataset_weight->size(); ++i) {
      metadata->SetWeightAt(i, dataset_weight->at(i));
    }
  }
  if (dataset_init_score) {
    Log::Info("    Loading init_score...");
    for (int i = 0; i < dataset_init_score->size(); ++i) {
      metadata->SetInitScoreAt(i, dataset_init_score->at(i));
    }
  }
  if (dataset_group) {
    Log::Info("    Loading group...");
    for (int i = 0; i < dataset_group->size(); ++i) {
      metadata->AppendQueryToBoundaries(dataset_group->at(i));
    }
  }
  Log::Info("    Finish streaming metadata...");
  metadata->FinishStreaming();
}

void test_metadata_insertion(const std::vector<float>* dataset1_label,
                             const std::vector<float>* dataset1_weight,
                             const std::vector<double>* dataset1_init_score,
                             const std::vector<uint8_t>* dataset1_group,
                             const std::vector<float>* dataset2_label,
                             const std::vector<float>* dataset2_weight,
                             const std::vector<double>* dataset2_init_score,
                             const std::vector<uint8_t>* dataset2_group) {
  std::unique_ptr<Metadata> metadata1;
  std::unique_ptr<Metadata> metadata2;

  metadata1.reset(new Metadata());
  metadata2.reset(new Metadata());

  int32_t nDataset1 = static_cast<int32_t>(dataset1_label->size());
  int32_t nDataset2 = static_cast<int32_t>(dataset2_label->size());
  int32_t nTotal = nDataset1 + nDataset2;
  Log::Info("  Loading source dataset 1...");
  load_metadata(metadata1.get(), dataset1_label, dataset1_weight, dataset1_init_score, dataset1_group);
  Log::Info("  Loading source dataset 2...");
  load_metadata(metadata2.get(), dataset2_label, dataset2_weight, dataset2_init_score, dataset2_group);

  std::unique_ptr<Metadata> metadata_coalesced;
  metadata_coalesced.reset(new Metadata());
  init_metadata(metadata_coalesced.get(), nTotal, dataset1_weight, dataset1_init_score,  dataset1_group);

  Log::Info("  Inserting from dataset1...");
  metadata_coalesced->AppendFrom(metadata1.get(), nDataset1);
  Log::Info("  Inserting from dataset2...");
  metadata_coalesced->AppendFrom(metadata2.get(), nDataset2);
  Log::Info("  Finish coalescing...");
  metadata_coalesced->FinishCoalesce();

  // Now assert the resulting coalesced values match the cumulative original arrays
  Log::Info("  Validating metadata coalesce...");
  std::vector<float>* coalesced_data_label_ptr = nullptr;
  std::vector<float>* coalesced_data_weight_ptr = nullptr;
  std::vector<double>* coalesced_data_init_score_ptr = nullptr;
  std::vector<int32_t>* coalesced_data_group_ptr = nullptr;
  std::vector<float> coalesced_data_label;
  std::vector<float> coalesced_data_weight;
  std::vector<double> coalesced_data_init_score;
  std::vector<int32_t> coalesced_data_group;
  coalesced_data_label.reserve(nDataset1 + nDataset2);
  coalesced_data_label_ptr = &coalesced_data_label;
  coalesced_data_weight.reserve(nDataset1 + nDataset2);
  coalesced_data_init_score.reserve(nDataset1 + nDataset2);
  coalesced_data_group.reserve(nDataset1 + nDataset2);
  coalesced_data_label.insert(coalesced_data_label.end(), dataset1_label->begin(), dataset1_label->end());
  coalesced_data_label.insert(coalesced_data_label.end(), dataset2_label->begin(), dataset2_label->end());
  if (dataset1_weight) {
    coalesced_data_weight_ptr = &coalesced_data_weight;
    coalesced_data_weight.insert(coalesced_data_weight.end(), dataset1_weight->begin(), dataset1_weight->end());
    coalesced_data_weight.insert(coalesced_data_weight.end(), dataset2_weight->begin(), dataset2_weight->end());
  }
  if (dataset1_init_score) {
    coalesced_data_init_score_ptr = &coalesced_data_init_score;
    coalesced_data_init_score.insert(coalesced_data_init_score.end(), dataset1_init_score->begin(), dataset1_init_score->end());
    coalesced_data_init_score.insert(coalesced_data_init_score.end(), dataset2_init_score->begin(), dataset2_init_score->end());
  }
  if (dataset1_group) {
    coalesced_data_group_ptr = &coalesced_data_group;
    coalesced_data_group.insert(coalesced_data_group.end(), dataset1_group->begin(), dataset1_group->end());
    coalesced_data_group.insert(coalesced_data_group.end(), dataset2_group->begin(), dataset2_group->end());
  }

  TestUtils::AssertMetadata(metadata_coalesced.get(),
                            coalesced_data_label_ptr,
                            coalesced_data_weight_ptr,
                            coalesced_data_init_score_ptr,
                            coalesced_data_group_ptr);
}

TEST(Coalesce, DenseBinInsertion) {
  const std::vector<uint8_t> dataset1_even = { 1,3,5,7 };
  const std::vector<uint8_t> dataset2_even = { 9,11,13,15 };
  const std::vector<uint8_t> dataset1_odd = { 1,3,5 };
  const std::vector<uint8_t> dataset2_odd = { 7,9,11 };

  Log::Info("Testing even + even");
  test_dense_insertion(256, &dataset1_even, &dataset2_even);
  Log::Info("Testing even + odd");
  test_dense_insertion(256, &dataset1_even, &dataset2_odd);
  Log::Info("Testing odd + even");
  test_dense_insertion(256, &dataset1_odd, &dataset2_even);
  Log::Info("Testing odd + odd");
  test_dense_insertion(256, &dataset1_odd, &dataset2_odd);

  Log::Info("Testing even + even");
  test_dense_insertion(16, &dataset1_even, &dataset2_even);
  Log::Info("Testing even + odd");
  test_dense_insertion(16, &dataset1_even, &dataset2_odd);
  Log::Info("Testing odd + even");
  test_dense_insertion(16, &dataset1_odd, &dataset2_even);
  Log::Info("Testing odd + odd");
  test_dense_insertion(16, &dataset1_odd, &dataset2_odd);
}

TEST(Coalesce, SparseBinInsertion) {
  const std::vector<int32_t> dataset1_ind = { 100,200,300,400,500,600 };  // less sparse
  const std::vector<int32_t> dataset2_ind = { 100,900 };                  // more sparse (> 255 gap)
  const std::vector<int32_t> dataset3_ind = { 0,500,999 };                // 0 and N-1
  const std::vector<int32_t> dataset4_ind = { 100,101,102,103,104 };      // sequential
  const std::vector<int32_t> dataset5_ind = { };                          // empty
  const std::vector<uint8_t> dataset1_val = { 1,3,5,7,9,11 };
  const std::vector<uint8_t> dataset2_val = { 13,15 };
  const std::vector<uint8_t> dataset3_val = { 2,2,2 };
  const std::vector<uint8_t> dataset4_val = { 1,2,1,2,1 };
  const std::vector<uint8_t> dataset5_val = { };

  Log::Info("Testing 1 & 2");
  test_sparse_insertion(1000, &dataset1_ind, &dataset1_val, 2000, &dataset2_ind, &dataset2_val);
  Log::Info("Testing 3 & 3");
  test_sparse_insertion(1000, &dataset3_ind, &dataset3_val, 2000, &dataset3_ind, &dataset3_val);
  Log::Info("Testing 3 & 4");
  test_sparse_insertion(1000, &dataset3_ind, &dataset3_val, 2000, &dataset4_ind, &dataset4_val);
  Log::Info("Testing 4 & 5");
  test_sparse_insertion(200, &dataset4_ind, &dataset4_val, 100, &dataset5_ind, &dataset5_val);
  Log::Info("Testing 5 & 2");
  test_sparse_insertion(50, &dataset5_ind, &dataset5_val, 2000, &dataset2_ind, &dataset2_val);
}

TEST(Coalesce, MetadataInsertion) {
  const std::vector<float> dataset1_label = { 1.0,3.0,5.0 };
  const std::vector<float> dataset1_weight = { 1.5,3.5,5.5 };
  const std::vector<double> dataset1_init_score = { 1.0,1.5,2.0 };
  const std::vector<uint8_t> dataset1_group = { 1, 1, 1 };

  const std::vector<float> dataset2_label = { 2.0,4.0,6.0,8.0 };
  const std::vector<float> dataset2_weight = { 2.5,4.5,6.5,8.5 };
  const std::vector<double> dataset2_init_score = { 1.5,2.5,3.5,4.5 };
  const std::vector<uint8_t> dataset2_group = { 2, 3, 3, 3 };

  // Test all coalesced properties, plus the null version of optional ones
  Log::Info("Testing all not null");
  test_metadata_insertion(&dataset1_label,
                          &dataset1_weight,
                          &dataset1_init_score,
                          &dataset1_group,
                          &dataset2_label,
                          &dataset2_weight,
                          &dataset2_init_score,
                          &dataset2_group);
  Log::Info("Testing weight null");
  test_metadata_insertion(&dataset1_label,
                          nullptr,
                          &dataset1_init_score,
                          &dataset1_group,
                          &dataset2_label,
                          nullptr,
                          &dataset2_init_score,
                          &dataset2_group);
  Log::Info("Testing initial score null");
  test_metadata_insertion(&dataset1_label,
                          &dataset1_weight,
                          nullptr,
                          &dataset1_group,
                          &dataset2_label,
                          &dataset2_weight,
                          nullptr,
                          &dataset2_group);
  Log::Info("Testing groups null");
  test_metadata_insertion(&dataset1_label,
                          &dataset1_weight,
                          &dataset1_init_score,
                          nullptr,
                          &dataset2_label,
                          &dataset2_weight,
                          &dataset2_init_score,
                          nullptr);
}

TEST(Coalesce, EndToEndDense) {
  // Load some test data
  DatasetHandle ref_datset_handle;
  const char* params = "max_bin=15";
  int result = TestUtils::LoadDatasetFromExamples("binary_classification\\binary.test", params, &ref_datset_handle);
  EXPECT_EQ(0, result) << "LoadDatasetFromExamples result code: " << result;

  Dataset* ref_dataset = static_cast<Dataset*>(ref_datset_handle);
  Log::Info("Row count: %d", ref_dataset->num_data());
  Log::Info("Feature group count: %d", ref_dataset->num_feature_groups());

  // Add some fake initial_scores and groups so we can test streaming them
  auto noriginalrows = ref_dataset->num_data();
  std::vector<double> unused_init_scores;
  unused_init_scores.resize(noriginalrows);
  std::vector<int32_t> unused_groups;
  unused_groups.assign(noriginalrows, 1);
  result = LGBM_DatasetSetField(ref_datset_handle, "init_score", unused_init_scores.data(), noriginalrows, 1);
  EXPECT_EQ(0, result) << "LGBM_DatasetSetField result code: " << result;
  result = LGBM_DatasetSetField(ref_datset_handle, "group", unused_groups.data(), noriginalrows, 2);
  EXPECT_EQ(0, result) << "LGBM_DatasetSetField result code: " << result;

  // Get 1 Datasets with the same schema and push data to them
  int32_t batch_count = 10;
  int32_t ncols = ref_dataset->num_features();

  // Source Dataset 1
  DatasetHandle source_dataset1_handle;
  int32_t nDataset1 = 100;
  result = LGBM_DatasetCreateByReference(ref_datset_handle, nDataset1, &source_dataset1_handle);
  EXPECT_EQ(0, result) << "LGBM_DatasetCreateByReference result code: " << result;

  std::vector<double> features1;
  std::vector<float> labels1;
  std::vector<float> weights1;
  std::vector<double> init_scores1;
  std::vector<int32_t> groups1;

  Log::Info("Creating random data 1");
  TestUtils::CreateRandomDenseData(nDataset1, ncols, &features1, &labels1, &weights1, &init_scores1, &groups1);

  Log::Info("Pushing batches 1");
  TestUtils::StreamDenseDataset(source_dataset1_handle, nDataset1, ncols, batch_count, &features1, &labels1, &weights1, &init_scores1, &groups1);

  // Source Dataset 2
  DatasetHandle source_dataset2_handle;
  int32_t nDataset2 = 150; // TODO test partial loading?
  result = LGBM_DatasetCreateByReference(ref_datset_handle, nDataset2, &source_dataset2_handle);
  EXPECT_EQ(0, result) << "LGBM_DatasetCreateByReference result code: " << result;

  std::vector<double> features2;
  std::vector<float> labels2;
  std::vector<float> weights2;
  std::vector<double> init_scores2;
  std::vector<int32_t> groups2;

  Log::Info("Creating random data 2");
  TestUtils::CreateRandomDenseData(nDataset2, ncols, &features2, &labels2, &weights2, &init_scores2, &groups2);

  Log::Info("Pushing batches 2");
  TestUtils::StreamDenseDataset(source_dataset2_handle, nDataset2, ncols, batch_count, &features2, &labels2, &weights2, &init_scores2, &groups2);

  // Target coalesced Dataset
  DatasetHandle coalesced_dataset_handle;
  result = LGBM_DatasetCreateByReference(ref_datset_handle, nDataset1 + nDataset2, &coalesced_dataset_handle);
  EXPECT_EQ(0, result) << "LGBM_DatasetCreateByReference result code: " << result;

  Dataset* coalesced_dataset = static_cast<Dataset*>(coalesced_dataset_handle);

  // Coalesce
  std::vector<DatasetHandle> sources;
  sources.push_back(source_dataset1_handle);
  sources.push_back(source_dataset2_handle);
  result = LGBM_DatasetCoalesce(coalesced_dataset_handle, sources.data(), 2);
  EXPECT_EQ(0, result) << "LGBM_DatasetCoalesce result code: " << result;

  // Now create the expected coalesced values
  Log::Info("  Validating metadata coalesce...");
  std::vector<float>* coalesced_data_label_ptr = nullptr;
  std::vector<float>* coalesced_data_weight_ptr = nullptr;
  std::vector<double>* coalesced_data_init_score_ptr = nullptr;
  std::vector<int32_t>* coalesced_data_group_ptr = nullptr;
  std::vector<float> coalesced_data_label;
  std::vector<float> coalesced_data_weight;
  std::vector<double> coalesced_data_init_score;
  std::vector<int32_t> coalesced_data_group;
  coalesced_data_label.reserve(nDataset1 + nDataset2);
  coalesced_data_label_ptr = &coalesced_data_label;
  coalesced_data_weight.reserve(nDataset1 + nDataset2);
  coalesced_data_init_score.reserve(nDataset1 + nDataset2);
  coalesced_data_group.reserve(nDataset1 + nDataset2);
  coalesced_data_label.insert(coalesced_data_label.end(), labels1.begin(), labels1.end());
  coalesced_data_label.insert(coalesced_data_label.end(), labels2.begin(), labels2.end());
  if (weights1.size() > 0) {
    coalesced_data_weight_ptr = &coalesced_data_weight;
    coalesced_data_weight.insert(coalesced_data_weight.end(), weights1.begin(), weights1.end());
    coalesced_data_weight.insert(coalesced_data_weight.end(), weights2.begin(), weights2.end());
  }
  if (init_scores1.size() > 0) {
    coalesced_data_init_score_ptr = &coalesced_data_init_score;
    coalesced_data_init_score.insert(coalesced_data_init_score.end(), init_scores1.begin(), init_scores1.end());
    coalesced_data_init_score.insert(coalesced_data_init_score.end(), init_scores2.begin(), init_scores2.end());
  }
  if (groups1.size() > 0) {
    coalesced_data_group_ptr = &coalesced_data_group;
    coalesced_data_group.insert(coalesced_data_group.end(), groups1.begin(), groups1.end());
    coalesced_data_group.insert(coalesced_data_group.end(), groups2.begin(), groups2.end());
  }

  TestUtils::AssertMetadata(&coalesced_dataset->metadata(),
                            coalesced_data_label_ptr,
                            coalesced_data_weight_ptr,
                            coalesced_data_init_score_ptr,
                            coalesced_data_group_ptr);
}

