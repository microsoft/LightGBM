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

  Log::Info("Loading source datasets...");
  int32_t nDataset1 = static_cast<int32_t>(dataset1_ind->size());
  int32_t nDataset2 = static_cast<int32_t>(dataset2_ind->size());
  Log::Info("Loading source dataset1...");
  for (int i = 0; i < nDataset1; ++i) {
    Log::Info("Pushing %d...", dataset1_val->at(i));
    sparseBin1->Push(0, dataset1_ind->at(i), dataset1_val->at(i));
  }
  Log::Info("Loading source dataset2...");
  for (int i = 0; i < nDataset2; ++i) {
    sparseBin2->Push(0, dataset2_ind->at(i), dataset2_val->at(i));
  }

  std::unique_ptr<Bin> sparseBin_coalesced;
  sparseBin_coalesced.reset(Bin::CreateSparseBin(count1 + count2, 256));
  Log::Info("Inserting from bin1...");
  sparseBin_coalesced->InsertFrom(sparseBin1.get(), 0, count1);
  Log::Info("Inserting from bin2...");
  sparseBin_coalesced->InsertFrom(sparseBin2.get(), count1, count2);
  Log::Info("Finish loading...");
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
    Log::Info("Row value: %d", val);
  }
}

void init_metadata(Metadata* metadata,
                   int32_t size,
                   const std::vector<float>* dataset_weight,
                   const std::vector<double>* dataset_init_score,
                   const std::vector<uint8_t>* dataset_group) {
  Log::Info("Initializing dataset...");
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
  metadata->InitInitScore();
}

void load_metadata(Metadata* metadata,
  const std::vector<float>* dataset_label,
  const std::vector<float>* dataset_weight,
  const std::vector<double>* dataset_init_score,
  const std::vector<uint8_t>* dataset_group) {
  init_metadata(metadata, static_cast<int32_t>(dataset_label->size()), dataset_weight, dataset_init_score, dataset_group);

  Log::Info("Loading label...");
  for (int i = 0; i < dataset_label->size(); ++i) {
    metadata->SetLabelAt(i, dataset_label->at(i));
  }
  if (dataset_weight != nullptr) {
    Log::Info("Loading weight...");
    for (int i = 0; i < dataset_weight->size(); ++i) {
      metadata->SetWeightAt(i, dataset_weight->at(i));
    }
  }
  if (dataset_init_score != nullptr) {
    Log::Info("Loading init_score...");
    for (int i = 0; i < dataset_init_score->size(); ++i) {
      metadata->SetInitScoreAt(i, dataset_init_score->at(i));
    }
  }
  if (dataset_group != nullptr) {
    Log::Info("Loading group...");
    for (int i = 0; i < dataset_init_score->size(); ++i) {
      metadata->AppendQueryToBoundaries(dataset_group->at(i));
    }
  }
  metadata->FinishLoad();
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
  Log::Info("Loading source dataset 1...");
  load_metadata(metadata1.get(), dataset1_label, dataset1_weight, dataset1_init_score, dataset1_group);
  Log::Info("Loading source dataset 2...");
  load_metadata(metadata2.get(), dataset2_label, dataset2_weight, dataset2_init_score, dataset2_group);

  // debug
  const double* init_scores_1 = metadata1->init_score();
  for (auto i = 0; i < nDataset1; i++)
  {
    Log::Info("Dataset 1 init_score at %d is %f", i, init_scores_1[i]);
  }

  std::unique_ptr<Metadata> metadata_coalesced;
  metadata_coalesced.reset(new Metadata());
  init_metadata(metadata_coalesced.get(), nTotal, dataset1_weight, dataset1_init_score,  dataset1_group);

  Log::Info("Inserting from dataset1...");
  metadata_coalesced->AppendFrom(metadata1.get(), nDataset1);
  Log::Info("Inserting from dataset2...");
  metadata_coalesced->AppendFrom(metadata2.get(), nDataset2);
  Log::Info("Finish loading...");
  metadata_coalesced->FinishCoalesce();

  // Now assert the resulting coalesced values match the cumulative original arrays
  std::vector<float> coalesced_data_label;
  std::vector<float> coalesced_data_weight;
  std::vector<double> coalesced_data_init_score;
  std::vector<int32_t> coalesced_data_group;
  coalesced_data_label.reserve(nDataset1 + nDataset2);
  coalesced_data_weight.reserve(nDataset1 + nDataset2);
  coalesced_data_init_score.reserve(nDataset1 + nDataset2);
  coalesced_data_group.reserve(nDataset1 + nDataset2);
  coalesced_data_label.insert(coalesced_data_label.end(), dataset1_label->begin(), dataset1_label->end());
  coalesced_data_label.insert(coalesced_data_label.end(), dataset2_label->begin(), dataset2_label->end());
  coalesced_data_weight.insert(coalesced_data_weight.end(), dataset1_weight->begin(), dataset1_weight->end());
  coalesced_data_weight.insert(coalesced_data_weight.end(), dataset2_weight->begin(), dataset2_weight->end());
  coalesced_data_init_score.insert(coalesced_data_init_score.end(), dataset1_init_score->begin(), dataset1_init_score->end());
  coalesced_data_init_score.insert(coalesced_data_init_score.end(), dataset2_init_score->begin(), dataset2_init_score->end());
  coalesced_data_group.insert(coalesced_data_group.end(), dataset1_group->begin(), dataset1_group->end());
  coalesced_data_group.insert(coalesced_data_group.end(), dataset2_group->begin(), dataset2_group->end());

  const float* labels = metadata_coalesced->label();
  for (auto i = 0; i < nTotal; i++)
  {
    EXPECT_EQ(coalesced_data_label[i], labels[i]) << "Coalesced data: " << coalesced_data_label[i];
  }
  const float* weights = metadata_coalesced->weights();
  if (weights)
  {
    for (auto i = 0; i < nTotal; i++)
    {
      EXPECT_EQ(coalesced_data_weight[i], weights[i]) << "Coalesced data: " << coalesced_data_weight[i];
    }
  }
  const double* init_scores = metadata_coalesced->init_score();
  if (init_scores)
  {
    for (auto i = 0; i < nTotal; i++)
    {
      EXPECT_EQ(coalesced_data_init_score[i], init_scores[i]) << "Coalesced data: " << coalesced_data_init_score[i];
    }
  }

  // Calculate expected boundaries
  std::vector<int32_t> coalesced_query_boundaries;
  coalesced_query_boundaries.push_back(0);
  int group_count = 1;
  int group_val = coalesced_data_group[0];
  for (auto i = 1; i < nTotal; i++)
  {
    if (coalesced_data_group[i] == group_val) {
      group_count++;
    } else {
      coalesced_query_boundaries.push_back(group_count);
      group_count = 1;
      group_val = coalesced_data_group[i];
    }
  }

  const int32_t* query_boundaries = metadata_coalesced->query_boundaries();
  if (query_boundaries)
  {
    for (auto i = 0; i < coalesced_query_boundaries.size(); i++)
    {
      Log::Info("Coalesced boundary at %d should be %d, is %d", i, coalesced_query_boundaries[i], query_boundaries[i]);
      EXPECT_EQ(coalesced_query_boundaries[i], query_boundaries[i]) << "Coalesced data group: " << coalesced_query_boundaries[i];
    }
  }
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

  test_metadata_insertion(&dataset1_label,
                          &dataset1_weight,
                          &dataset1_init_score,
                          &dataset1_group,
                          &dataset2_label,
                          &dataset2_weight,
                          &dataset2_init_score,
                          &dataset2_group);
}
