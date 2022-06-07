/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "multi_val_dense_bin.hpp"

namespace LightGBM {


#ifdef USE_CUDA_EXP
template <>
const void* MultiValDenseBin<uint8_t>::GetRowWiseData(uint8_t* bit_type,
    size_t* total_size,
    bool* is_sparse,
    const void** out_data_ptr,
    uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = data_.data();
  *bit_type = 8;
  *total_size = static_cast<size_t>(num_data_) * static_cast<size_t>(num_feature_);
  CHECK_EQ(*total_size, data_.size());
  *is_sparse = false;
  *out_data_ptr = nullptr;
  *data_ptr_bit_type = 0;
  return to_return;
}

template <>
const void* MultiValDenseBin<uint16_t>::GetRowWiseData(uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint16_t* data_ptr = data_.data();
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_ptr);
  *bit_type = 16;
  *total_size = static_cast<size_t>(num_data_) * static_cast<size_t>(num_feature_);
  CHECK_EQ(*total_size, data_.size());
  *is_sparse = false;
  *out_data_ptr = nullptr;
  *data_ptr_bit_type = 0;
  return to_return;
}

template <>
const void* MultiValDenseBin<uint32_t>::GetRowWiseData(uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint32_t* data_ptr = data_.data();
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_ptr);
  *bit_type = 32;
  *total_size = static_cast<size_t>(num_data_) * static_cast<size_t>(num_feature_);
  CHECK_EQ(*total_size, data_.size());
  *is_sparse = false;
  *out_data_ptr = nullptr;
  *data_ptr_bit_type = 0;
  return to_return;
}

#endif  // USE_CUDA_EXP

}  // namespace LightGBM
