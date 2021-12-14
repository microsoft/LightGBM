/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "multi_val_sparse_bin.hpp"

namespace LightGBM {

#ifdef USE_CUDA_EXP

template <>
const void* MultiValSparseBin<uint16_t, uint8_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = data_.data();
  *bit_type = 8;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 16;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint16_t, uint16_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 16;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 16;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint16_t, uint32_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 32;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 16;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint32_t, uint8_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = data_.data();
  *bit_type = 8;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 32;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint32_t, uint16_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 16;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 32;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint32_t, uint32_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 32;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 32;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint64_t, uint8_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = data_.data();
  *bit_type = 8;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 64;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint64_t, uint16_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 16;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 64;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint64_t, uint32_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 32;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 64;
  return to_return;
}

#endif  // USE_CUDA_EXP

}  // namespace LightGBM
