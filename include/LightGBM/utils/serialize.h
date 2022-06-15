/*!
 * Copyright (c) 2018 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_UTILS_SERIALIZE_H_
#define LIGHTGBM_UTILS_SERIALIZE_H_

#include <cstdlib>
#include <vector>

namespace LightGBM {

/*!
 * \brief An interface for serializing binary data to a buffer
 */
struct BinaryWriter {
  virtual size_t Write(const void* data, size_t bytes) = 0;
  size_t AlignedWrite(const void* data, size_t bytes, size_t alignment = 8) {
    auto ret = Write(data, bytes);
    if (bytes % alignment != 0) {
      size_t padding = AlignedSize(bytes, alignment) - bytes;
      std::vector<char> tmp(padding, 0);
      ret += Write(tmp.data(), padding);
    }
    return ret;
  }

  static size_t AlignedSize(size_t bytes, size_t alignment = 8) {
    if (bytes % alignment == 0) {
      return bytes;
    } else {
      return bytes / alignment * alignment + alignment;
    }
  }
};

}  // namespace LightGBM

#endif   // LightGBM_UTILS_SERIALIZE_H_
