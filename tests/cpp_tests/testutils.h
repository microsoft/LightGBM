/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TESTS_UTILS_H_
#define LIGHTGBM_TESTS_UTILS_H_

#include <LightGBM/c_api.h>
#include <LightGBM/dataset.h>

#include <vector>

using LightGBM::Metadata;
using namespace std;

namespace LightGBM {

class TestUtils {
  public:

    /*!
    * Creates a Dataset from the internal repository examples.
    */
    static int LoadDatasetFromExamples(const char* filename, const char* config, DatasetHandle* out);


    /*!
    * Creates a dense Dataset of random values.
    */
    static void CreateRandomDenseData(int32_t nrows,
                                      int32_t ncols,
                                      int32_t nclasses,
                                      std::vector<double>* features,
                                      std::vector<float>* labels,
                                      std::vector<float>* weights,
                                      std::vector<double>* init_scores,
                                      std::vector<int32_t>* groups);

    /*!
    * Creates a CSR sparse Dataset of random values.
    */
    static void TestUtils::CreateRandomSparseData(int32_t nrows,
                                                  int32_t ncols,
                                                  int32_t nclasses,
                                                 float sparse_percent,
                                                  std::vector<int32_t>* indptr,
                                                  std::vector<int32_t>* indices,
                                                  std::vector<double>* values,
                                                  std::vector<float>* labels,
                                                  std::vector<float>* weights,
                                                  std::vector<double>* init_scores,
                                                  std::vector<int32_t>* groups);

    /*!
    * Creates a CSR sparse Dataset of random values.
    */
    static void TestUtils::CreateRandomMetadata(int32_t nrows,
                                                int32_t nclasses,
                                                std::vector<float>* labels,
                                                std::vector<float>* weights,
                                                std::vector<double>* init_scores,
                                                std::vector<int32_t>* groups);

    /*!
    * Pushes nrows of data to a Dataset in batches of batch_count.
    */
    static void StreamDenseDataset(DatasetHandle dataset_handle,
                                   int32_t nrows,
                                   int32_t ncols,
                                   int32_t nclasses,
                                   int32_t batch_count,
                                   const std::vector<double> *features,
                                   const std::vector<float> *labels,
                                   const std::vector<float> *weights,
                                   const std::vector<double> *init_scores,
                                   const std::vector<int32_t> *groups);

    /*!
    * Pushes nrows of data to a Dataset in batches of batch_count.
    */
    static void StreamSparseDataset(DatasetHandle dataset_handle,
                                    int32_t nrows,
                                    int32_t nclasses,
                                    int32_t batch_count,
                                    const std::vector<int32_t> *indptr,
                                    const std::vector<int32_t> *indices,
                                    const std::vector<double> *values,
                                    const std::vector<float> *labels,
                                    const std::vector<float> *weights,
                                    const std::vector<double> *init_scores,
                                    const std::vector<int32_t> *groups);

    /*!
    * Validates metadata against reference vectors.
    */
    static void AssertMetadata(const Metadata* metadata,
                               const std::vector<float>* labels,
                               const std::vector<float>* weights,
                               const std::vector<double>* init_scores,
                               const std::vector<int32_t>* groups);

    static const double* TestUtils::CreateInitScoreBatch(std::vector<double>& init_score_batch,
                                                int32_t index,
                                                int32_t nrows,
                                                int32_t nclasses,
                                                int32_t batch_count,
                                                const std::vector<double>* original_init_scores);
};
}
#endif  // LIGHTGBM_TESTS_UTILS_H_

