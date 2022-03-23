/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "dense_bin.hpp"

namespace LightGBM {

template <>
const void* DenseBin<uint8_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int /*num_threads*/) const {
  *is_sparse = false;
  *bit_type = 8;
  bin_iterator->clear();
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint16_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int /*num_threads*/) const {
  *is_sparse = false;
  *bit_type = 16;
  bin_iterator->clear();
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint32_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int /*num_threads*/) const {
  *is_sparse = false;
  *bit_type = 32;
  bin_iterator->clear();
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint8_t, true>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int /*num_threads*/) const {
  *is_sparse = false;
  *bit_type = 4;
  bin_iterator->clear();
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint8_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = false;
  *bit_type = 8;
  *bin_iterator = nullptr;
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint16_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = false;
  *bit_type = 16;
  *bin_iterator = nullptr;
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint32_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = false;
  *bit_type = 32;
  *bin_iterator = nullptr;
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint8_t, true>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = false;
  *bit_type = 4;
  *bin_iterator = nullptr;
  return reinterpret_cast<const void*>(data_.data());
}

}  // namespace LightGBM
