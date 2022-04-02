/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "sparse_bin.hpp"

namespace LightGBM {

template <>
const void* SparseBin<uint8_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int num_threads) const {
  *is_sparse = true;
  *bit_type = 8;
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    bin_iterator->emplace_back(new SparseBinIterator<uint8_t>(this, 0));
  }
  return nullptr;
}

template <>
const void* SparseBin<uint16_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int num_threads) const {
  *is_sparse = true;
  *bit_type = 16;
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    bin_iterator->emplace_back(new SparseBinIterator<uint16_t>(this, 0));
  }
  return nullptr;
}

template <>
const void* SparseBin<uint32_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int num_threads) const {
  *is_sparse = true;
  *bit_type = 32;
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    bin_iterator->emplace_back(new SparseBinIterator<uint32_t>(this, 0));
  }
  return nullptr;
}

template <>
const void* SparseBin<uint8_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = true;
  *bit_type = 8;
  *bin_iterator = new SparseBinIterator<uint8_t>(this, 0);
  return nullptr;
}

template <>
const void* SparseBin<uint16_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = true;
  *bit_type = 16;
  *bin_iterator = new SparseBinIterator<uint16_t>(this, 0);
  return nullptr;
}

template <>
const void* SparseBin<uint32_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = true;
  *bit_type = 32;
  *bin_iterator = new SparseBinIterator<uint32_t>(this, 0);
  return nullptr;
}

}  // namespace LightGBM
