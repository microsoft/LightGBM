/*!
 * Copyright (c) 2018 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <LightGBM/utils/file_io.h>

#include <LightGBM/utils/log.h>

#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace LightGBM {

struct LocalFile : VirtualFileReader, VirtualFileWriter {
  LocalFile(const std::string& filename, const std::string& mode)
      : filename_(filename), mode_(mode) {}
  virtual ~LocalFile() {
    if (file_ != NULL) {
      fclose(file_);
    }
  }

  bool Init() {
    if (file_ == NULL) {
#if _MSC_VER
      fopen_s(&file_, filename_.c_str(), mode_.c_str());
#else
      file_ = fopen(filename_.c_str(), mode_.c_str());
#endif
    }
    return file_ != NULL;
  }

  bool Exists() const {
    LocalFile file(filename_, "rb");
    return file.Init();
  }

  size_t Read(void* buffer, size_t bytes) const {
    return fread(buffer, 1, bytes, file_);
  }

  size_t Write(const void* buffer, size_t bytes) {
    return fwrite(buffer, bytes, 1, file_) == 1 ? bytes : 0;
  }

 private:
  FILE* file_ = NULL;
  const std::string filename_;
  const std::string mode_;
};

std::unique_ptr<VirtualFileReader> VirtualFileReader::Make(
    const std::string& filename) {
  return std::unique_ptr<VirtualFileReader>(new LocalFile(filename, "rb"));
}

std::unique_ptr<VirtualFileWriter> VirtualFileWriter::Make(
    const std::string& filename) {
  return std::unique_ptr<VirtualFileWriter>(new LocalFile(filename, "wb"));
}

bool VirtualFileWriter::Exists(const std::string& filename) {
  LocalFile file(filename, "rb");
  return file.Exists();
}

}  // namespace LightGBM
