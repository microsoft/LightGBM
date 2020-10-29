/*!
 * Copyright (c) 2018 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_UTILS_FILE_IO_H_
#define LIGHTGBM_UTILS_FILE_IO_H_

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace LightGBM {

/*!
 * \brief An interface for writing files from buffers
 */
struct VirtualFileWriter {
  virtual ~VirtualFileWriter() {}
  /*!
   * \brief Initialize the writer
   * \return True when the file is available for writes
   */
  virtual bool Init() = 0;
  /*!
   * \brief Append buffer to file
   * \param data Buffer to write from
   * \param bytes Number of bytes to write from buffer
   * \return Number of bytes written
   */
  virtual size_t Write(const void* data, size_t bytes) const = 0;

  size_t AlignedWrite(const void* data, size_t bytes, size_t alignment = 8) const {
    auto ret = Write(data, bytes);
    if (bytes % alignment != 0) {
      size_t padding = AlignedSize(bytes, alignment) - bytes;
      std::vector<char> tmp(padding, 0);
      ret += Write(tmp.data(), padding);
    }
    return ret;
  }
  /*!
   * \brief Create appropriate writer for filename
   * \param filename Filename of the data
   * \return File writer instance
   */
  static std::unique_ptr<VirtualFileWriter> Make(const std::string& filename);
  /*!
   * \brief Check filename existence
   * \param filename Filename of the data
   * \return True when the file exists
   */
  static bool Exists(const std::string& filename);

  static size_t AlignedSize(size_t bytes, size_t alignment = 8) {
    if (bytes % alignment == 0) {
      return bytes;
    } else {
      return bytes / alignment * alignment + alignment;
    }
  }
};

/**
 * \brief An interface for reading files into buffers
 */
struct VirtualFileReader {
  /*!
   * \brief Constructor
   * \param filename Filename of the data
   */
  virtual ~VirtualFileReader() {}
  /*!
   * \brief Initialize the reader
   * \return True when the file is available for read
   */
  virtual bool Init() = 0;
  /*!
   * \brief Read data into buffer
   * \param buffer Buffer to read data into
   * \param bytes Number of bytes to read
   * \return Number of bytes read
   */
  virtual size_t Read(void* buffer, size_t bytes) const = 0;
  /*!
   * \brief Create appropriate reader for filename
   * \param filename Filename of the data
   * \return File reader instance
   */
  static std::unique_ptr<VirtualFileReader> Make(const std::string& filename);
};

}  // namespace LightGBM

#endif   // LightGBM_UTILS_FILE_IO_H_
