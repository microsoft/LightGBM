/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <testutils.h>
#include <LightGBM/utils/byte_buffer.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/c_api.h>
#include <LightGBM/dataset.h>

#include <iostream>

using LightGBM::ByteBuffer;
using LightGBM::Dataset;
using LightGBM::Log;
using LightGBM::TestUtils;
using namespace std;

TEST(Serialization, JustWorks) {
  // Load some test data
  DatasetHandle datset_handle;
  const char* params = "max_bin=15";
  int result = TestUtils::LoadDatasetFromExamples("binary_classification\\binary.train", params, &datset_handle);
  EXPECT_EQ(0, result) << "LoadDatasetFromExamples result code: " << result;

  Dataset* dataset = static_cast<Dataset*>(datset_handle);
  Log::Info("Row count: %d", dataset->num_data());
  Log::Info("Feature group count: %d", dataset->num_feature_groups());

  // Serialize the reference
  ByteBufferHandle buffer_handle;
  int32_t buffer_len;
  result = LGBM_DatasetSerializeReferenceToBinary(datset_handle, &buffer_handle, &buffer_len);
  EXPECT_EQ(0, result) << "LGBM_DatasetSerializeReferenceToBinary result code: " << result;

  ByteBuffer* buffer = static_cast<ByteBuffer*>(buffer_handle);
  Log::Info("Buffer size: %d", buffer->GetSize());

  // Deserialize the reference
  DatasetHandle deserialized_datset_handle;
  result = LGBM_DatasetCreateFromSerializedReference(buffer->Data(),
    static_cast<int32_t>(buffer->GetSize()),
    dataset->num_data(),
    0, // num_classes
    params,
    &deserialized_datset_handle);
  EXPECT_EQ(0, result) << "LGBM_DatasetCreateFromSerializedReference result code: " << result;

  Dataset* deserialized_dataset = static_cast<Dataset*>(deserialized_datset_handle);
  Log::Info("Deserialized Row count: %d", deserialized_dataset->num_data());
  Log::Info("Feture group count: %d", deserialized_dataset->num_feature_groups());

  // Free memory
  result = LGBM_ByteBufferFree(buffer);
  EXPECT_EQ(0, result) << "LGBM_ByteBufferFree result code: " << result;

  result = LGBM_DatasetFree(dataset);
  EXPECT_EQ(0, result) << "LGBM_DatasetFree result code: " << result;
}
