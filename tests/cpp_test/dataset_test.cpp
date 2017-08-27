#include <fstream>

// #include <LightGBM/metric.h>
#include <LightGBM/dataset.h>


#include "gtest/gtest.h"

TEST(DataSetLoaderMetadata, Init)
{
  LightGBM::Metadata meta;
  meta.Init(1, NO_SPECIFIC, NO_SPECIFIC);
  EXPECT_EQ(meta.SizesInByte(), 12 + 4 * 1);

  meta.Init(10, NO_SPECIFIC, NO_SPECIFIC);
  EXPECT_EQ(meta.SizesInByte(), 12 + 4 * 10);
}

TEST(DataSetLoaderMetadata, Weights)
{
  LightGBM::Metadata meta;
  meta.Init(10, NO_SPECIFIC, NO_SPECIFIC);

  std::vector<float> weightsVec(3, 10);

  // data is correct and size is wrong
  EXPECT_THROW(meta.SetWeights(weightsVec.data(), 3), std::runtime_error);

  // data is correct and size is correct
  meta.SetWeights(weightsVec.data(), 10);
  for (size_t i = 0; i < weightsVec.size(); ++i) {
    EXPECT_EQ(meta.weights()[i], weightsVec.at(i));
  }

  // change a specific weight
  meta.SetWeightAt(2, 4.0);
  EXPECT_EQ(meta.weights()[2], 4.0);

  // clear up weights by giving nullptr
  meta.SetWeights(nullptr, 10);
  EXPECT_EQ(meta.weights(), nullptr);
}

TEST(DataSetLoaderMetadata, Labels)
{
  LightGBM::Metadata meta;
  meta.Init(10, NO_SPECIFIC, NO_SPECIFIC);

  std::vector<float> labelsVec(1, 10);

  // data is correct and size is wrong
  EXPECT_THROW(meta.SetLabel(labelsVec.data(), 3), std::runtime_error);

  // data is wrong
  EXPECT_THROW(meta.SetLabel(nullptr, 10), std::runtime_error);

  // data is wrong and size is wrong
  EXPECT_THROW(meta.SetLabel(nullptr, 2), std::runtime_error);

  // data is correct and size is correct
  meta.SetLabel(labelsVec.data(), 10);
  for (size_t i = 0; i < labelsVec.size(); ++i) {
    EXPECT_EQ(meta.label()[0], labelsVec.at(i));
  }

  // change a specific label
  meta.SetLabelAt(2, 0.0);
  EXPECT_EQ(meta.label()[2], 0.0);
}
