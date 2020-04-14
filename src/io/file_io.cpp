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

#ifdef USE_HDFS
#include <hdfs.h>
#endif

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

  size_t Write(const void* buffer, size_t bytes) const {
    return fwrite(buffer, bytes, 1, file_) == 1 ? bytes : 0;
  }

 private:
  FILE* file_ = NULL;
  const std::string filename_;
  const std::string mode_;
};

const char* kHdfsProto = "hdfs://";

#ifdef USE_HDFS
const size_t kHdfsProtoLength = static_cast<size_t>(strlen(kHdfsProto));

struct HDFSFile : VirtualFileReader, VirtualFileWriter {
  HDFSFile(const std::string& filename, int flags)
      : filename_(filename), flags_(flags) {}
  ~HDFSFile() {
    if (file_ != NULL) {
      hdfsCloseFile(fs_, file_);
    }
  }

  bool Init() {
    if (file_ == NULL) {
      if (fs_ == NULL) {
        fs_ = GetHDFSFileSystem(filename_);
      }
      if (fs_ != NULL &&
          (flags_ == O_WRONLY || 0 == hdfsExists(fs_, filename_.c_str()))) {
        file_ = hdfsOpenFile(fs_, filename_.c_str(), flags_, 0, 0, 0);
      }
    }
    return file_ != NULL;
  }

  bool Exists() const {
    if (fs_ == NULL) {
      fs_ = GetHDFSFileSystem(filename_);
    }
    return fs_ != NULL && 0 == hdfsExists(fs_, filename_.c_str());
  }

  size_t Read(void* data, size_t bytes) const {
    return FileOperation<void*>(data, bytes, &hdfsRead);
  }

  size_t Write(const void* data, size_t bytes) const {
    return FileOperation<const void*>(data, bytes, &hdfsWrite);
  }

 private:
  template <typename BufferType>
  using fileOp = tSize (*)(hdfsFS, hdfsFile, BufferType, tSize);

  template <typename BufferType>
  inline size_t FileOperation(BufferType data, size_t bytes,
                              fileOp<BufferType> op) const {
    char* buffer = const_cast<char*>(static_cast<const char*>(data));
    size_t remain = bytes;
    while (remain != 0) {
      size_t nmax = static_cast<size_t>(std::numeric_limits<tSize>::max());
      tSize ret = op(fs_, file_, buffer, std::min(nmax, remain));
      if (ret > 0) {
        size_t n = static_cast<size_t>(ret);
        remain -= n;
        buffer += n;
      } else if (ret == 0) {
        break;
      } else if (errno != EINTR) {
        Log::Fatal("Failed HDFS file operation [%s]", strerror(errno));
      }
    }
    return bytes - remain;
  }

  static hdfsFS GetHDFSFileSystem(const std::string& uri) {
    size_t end = uri.find("/", kHdfsProtoLength);
    if (uri.find(kHdfsProto) != 0 || end == std::string::npos) {
      Log::Warning("Bad HDFS uri, no namenode found [%s]", uri.c_str());
      return NULL;
    }
    std::string hostport = uri.substr(kHdfsProtoLength, end - kHdfsProtoLength);
    if (fs_cache_.count(hostport) == 0) {
      fs_cache_[hostport] = MakeHDFSFileSystem(hostport);
    }
    return fs_cache_[hostport];
  }

  static hdfsFS MakeHDFSFileSystem(const std::string& hostport) {
    std::istringstream iss(hostport);
    std::string host;
    tPort port = 0;
    std::getline(iss, host, ':');
    iss >> port;
    hdfsFS fs = iss.eof() ? hdfsConnect(host.c_str(), port) : NULL;
    if (fs == NULL) {
      Log::Warning("Could not connect to HDFS namenode [%s]", hostport.c_str());
    }
    return fs;
  }

  mutable hdfsFS fs_ = NULL;
  hdfsFile file_ = NULL;
  const std::string filename_;
  const int flags_;
  static std::unordered_map<std::string, hdfsFS> fs_cache_;
};

std::unordered_map<std::string, hdfsFS> HDFSFile::fs_cache_ =
    std::unordered_map<std::string, hdfsFS>();

#define WITH_HDFS(x) x
#else
#define WITH_HDFS(x) Log::Fatal("HDFS support is not enabled")
#endif  // USE_HDFS

std::unique_ptr<VirtualFileReader> VirtualFileReader::Make(
    const std::string& filename) {
#ifdef USE_HDFS
  if (0 == filename.find(kHdfsProto)) {
    WITH_HDFS(return std::unique_ptr<VirtualFileReader>(
        new HDFSFile(filename, O_RDONLY)));
  }
#endif
  return std::unique_ptr<VirtualFileReader>(new LocalFile(filename, "rb"));
}

std::unique_ptr<VirtualFileWriter> VirtualFileWriter::Make(
    const std::string& filename) {
#ifdef USE_HDFS
  if (0 == filename.find(kHdfsProto)) {
    WITH_HDFS(return std::unique_ptr<VirtualFileWriter>(
        new HDFSFile(filename, O_WRONLY)));
  }
#endif
  return std::unique_ptr<VirtualFileWriter>(new LocalFile(filename, "wb"));
}

bool VirtualFileWriter::Exists(const std::string& filename) {
#ifdef USE_HDFS
  if (0 == filename.find(kHdfsProto)) {
    WITH_HDFS(HDFSFile file(filename, O_RDONLY); return file.Exists());
  }
#endif
  LocalFile file(filename, "rb");
  return file.Exists();
}

}  // namespace LightGBM
