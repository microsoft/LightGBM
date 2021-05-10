/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/array_args.h>


template<class Value>
double ComputeExpectationOfMVS(const std::vector<Value> &grads, double threshold) {
  using namespace LightGBM;
  double expectation = 0.0;
  for (const auto &value : grads) {
    if (value >= threshold) {
      expectation += 1.;
    } else {
      expectation += value / threshold;
    }
  }
  return expectation;
}

void ComputeSamplingRate(std::vector<LightGBM::score_t> gradients,
                       const double sampling_fraction,
                       double *expected_sample_size,
                       double *resulting_sample_size) {
  using namespace LightGBM;
  CHECK(expected_sample_size != nullptr);
  CHECK(resulting_sample_size != nullptr);
  *expected_sample_size = sampling_fraction * static_cast<double>(gradients.size());
  double threshold = ArrayArgs<score_t>::CalculateThresholdMVS(&gradients, 0, gradients.size(), *expected_sample_size);
  *resulting_sample_size = ComputeExpectationOfMVS(gradients, threshold);
}

TEST(SearchThresholdMVS, Basic) {
  using namespace LightGBM;
  std::vector<score_t> gradients({0.5f, 5.0f, 1.0f, 2.0f, 2.0f});
  double expected, resulting;
  ComputeSamplingRate(gradients, 0.5, &expected, &resulting);
  EXPECT_DOUBLE_EQ(expected, resulting);
}

TEST(ArrayArgs, Partition) {
  using namespace LightGBM;
  std::vector<score_t> gradients({0.5f, 5.0f, 1.0f, 2.0f, 2.0f});
  data_size_t middle_begin = -1, middle_end = gradients.size();
  ArrayArgs<score_t>::Partition(&gradients, 0, gradients.size(), &middle_begin, &middle_end);
  EXPECT_EQ(gradients[middle_begin + 1], gradients[middle_end - 1]);
  EXPECT_GT(gradients[0], gradients[middle_begin + 1]);
  EXPECT_GT(gradients[middle_begin + 1], gradients.back());
}

TEST(SearchThresholdMVS, PartitionOneElement) {
  using namespace LightGBM;
  std::vector<score_t> gradients({0.5f});
  data_size_t middle_begin = -1, middle_end = gradients.size();
  ArrayArgs<score_t>::Partition(&gradients, 0, gradients.size(), &middle_begin, &middle_end);
  EXPECT_EQ(gradients[middle_begin + 1], gradients[middle_end - 1]);
}
