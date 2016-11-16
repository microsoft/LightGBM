#include <LightGBM/utils/common.h>
#include <LightGBM/bin.h>

#include "dense_bin.hpp"
#include "sparse_bin.hpp"

#include <cmath>
#include <cstring>
#include <cstdint>

#include <limits>
#include <vector>
#include <algorithm>

namespace LightGBM {

BinMapper::BinMapper() {
}

// deep copy function for BinMapper
BinMapper::BinMapper(const BinMapper& other) {
  num_bin_ = other.num_bin_;
  is_trival_ = other.is_trival_;
  sparse_rate_ = other.sparse_rate_;
  bin_upper_bound_ = std::vector<double>(num_bin_);
  for (int i = 0; i < num_bin_; ++i) {
    bin_upper_bound_[i] = other.bin_upper_bound_[i];
  }
}

BinMapper::BinMapper(const void* memory) {
  CopyFrom(reinterpret_cast<const char*>(memory));
}

BinMapper::~BinMapper() {

}

void BinMapper::FindBin(std::vector<double>* values, size_t total_sample_cnt, int max_bin) {
  std::vector<double>& ref_values = (*values);
  size_t sample_size = total_sample_cnt;
  int zero_cnt = static_cast<int>(total_sample_cnt - ref_values.size());
  // find distinct_values first
  std::vector<double> distinct_values;
  std::vector<int> counts;

  std::sort(ref_values.begin(), ref_values.end());

  // push zero in the front
  if (ref_values.size() == 0 || (ref_values[0] > 0.0f && zero_cnt > 0)) {
    distinct_values.push_back(0);
    counts.push_back(zero_cnt);
  }

  if (ref_values.size() > 0) {
    distinct_values.push_back(ref_values[0]);
    counts.push_back(1);
  }

  for (size_t i = 1; i < ref_values.size(); ++i) {
    if (ref_values[i] != ref_values[i - 1]) {
      if (ref_values[i - 1] == 0.0f) {
        counts.back() += zero_cnt;
      } else if (ref_values[i - 1] < 0.0f && ref_values[i] > 0.0f) {
        distinct_values.push_back(0);
        counts.push_back(zero_cnt);
      }
      distinct_values.push_back(ref_values[i]);
      counts.push_back(1);
    } else {
      ++counts.back();
    }
  }

  // push zero in the back
  if (ref_values.size() > 0 && ref_values.back() < 0.0f && zero_cnt > 0) {
    distinct_values.push_back(0);
    counts.push_back(zero_cnt);
  }

  int num_values = static_cast<int>(distinct_values.size());
  int cnt_in_bin0 = 0;
  if (num_values <= max_bin) {
    std::sort(distinct_values.begin(), distinct_values.end());
    // use distinct value is enough
    num_bin_ = num_values;
    bin_upper_bound_ = std::vector<double>(num_values);
    for (int i = 0; i < num_values - 1; ++i) {
      bin_upper_bound_[i] = (distinct_values[i] + distinct_values[i + 1]) / 2;
    }
    cnt_in_bin0 = counts[0];
    bin_upper_bound_[num_values - 1] = std::numeric_limits<double>::infinity();
  } else {
    // mean size for one bin
    double mean_bin_size = sample_size / static_cast<double>(max_bin);
    double static_mean_bin_size = mean_bin_size;
    std::vector<double> upper_bounds(max_bin, std::numeric_limits<double>::infinity());
    std::vector<double> lower_bounds(max_bin, std::numeric_limits<double>::infinity());

    int rest_sample_cnt = static_cast<int>(sample_size);
    int bin_cnt = 0;
    lower_bounds[bin_cnt] = distinct_values[0];
    int cur_cnt_inbin = 0;
    for (int i = 0; i < num_values - 1; ++i) {
      rest_sample_cnt -= counts[i];
      cur_cnt_inbin += counts[i];
      // need a new bin
      if (counts[i] >= static_mean_bin_size || cur_cnt_inbin >= mean_bin_size ||
        (counts[i + 1] >= static_mean_bin_size && cur_cnt_inbin >= std::max(1.0, mean_bin_size * 0.5f))) {
        upper_bounds[bin_cnt] = distinct_values[i];
        if (bin_cnt == 0) {
          cnt_in_bin0 = cur_cnt_inbin;
        }
        ++bin_cnt;
        lower_bounds[bin_cnt] = distinct_values[i + 1];
        if (bin_cnt >= max_bin - 1) { break; }
        cur_cnt_inbin = 0;
        mean_bin_size = rest_sample_cnt / static_cast<double>(max_bin - bin_cnt);
      }
    }
    //
    ++bin_cnt;
    // update bin upper bound
    bin_upper_bound_ = std::vector<double>(bin_cnt);
    num_bin_ = bin_cnt;
    for (int i = 0; i < bin_cnt - 1; ++i) {
      bin_upper_bound_[i] = (upper_bounds[i] + lower_bounds[i + 1]) / 2.0f;
    }
    // last bin upper bound
    bin_upper_bound_[bin_cnt - 1] = std::numeric_limits<double>::infinity();
  }
  // check trival(num_bin_ == 1) feature
  if (num_bin_ <= 1) {
    is_trival_ = true;
  } else {
    is_trival_ = false;
  }
  // calculate sparse rate
  sparse_rate_ = static_cast<double>(cnt_in_bin0) / static_cast<double>(sample_size);
}


int BinMapper::SizeForSpecificBin(int bin) {
  int size = 0;
  size += sizeof(int);
  size += sizeof(bool);
  size += sizeof(double);
  size += bin * sizeof(double);
  return size;
}

void BinMapper::CopyTo(char * buffer) {
  std::memcpy(buffer, &num_bin_, sizeof(num_bin_));
  buffer += sizeof(num_bin_);
  std::memcpy(buffer, &is_trival_, sizeof(is_trival_));
  buffer += sizeof(is_trival_);
  std::memcpy(buffer, &sparse_rate_, sizeof(sparse_rate_));
  buffer += sizeof(sparse_rate_);
  std::memcpy(buffer, bin_upper_bound_.data(), num_bin_ * sizeof(double));
}

void BinMapper::CopyFrom(const char * buffer) {
  std::memcpy(&num_bin_, buffer, sizeof(num_bin_));
  buffer += sizeof(num_bin_);
  std::memcpy(&is_trival_, buffer, sizeof(is_trival_));
  buffer += sizeof(is_trival_);
  std::memcpy(&sparse_rate_, buffer, sizeof(sparse_rate_));
  buffer += sizeof(sparse_rate_);
  bin_upper_bound_ = std::vector<double>(num_bin_);
  std::memcpy(bin_upper_bound_.data(), buffer, num_bin_ * sizeof(double));
}

void BinMapper::SaveBinaryToFile(FILE* file) const {
  fwrite(&num_bin_, sizeof(num_bin_), 1, file);
  fwrite(&is_trival_, sizeof(is_trival_), 1, file);
  fwrite(&sparse_rate_, sizeof(sparse_rate_), 1, file);
  fwrite(bin_upper_bound_.data(), sizeof(double), num_bin_, file);
}

size_t BinMapper::SizesInByte() const {
  return sizeof(num_bin_) + sizeof(is_trival_) + sizeof(sparse_rate_) + sizeof(double) * num_bin_;
}

template class DenseBin<uint8_t>;
template class DenseBin<uint16_t>;
template class DenseBin<uint32_t>;

template class SparseBin<uint8_t>;
template class SparseBin<uint16_t>;
template class SparseBin<uint32_t>;

template class OrderedSparseBin<uint8_t>;
template class OrderedSparseBin<uint16_t>;
template class OrderedSparseBin<uint32_t>;


Bin* Bin::CreateBin(data_size_t num_data, int num_bin, double sparse_rate, bool is_enable_sparse, bool* is_sparse, int default_bin) {
  // sparse threshold
  const double kSparseThreshold = 0.8f;
  if (sparse_rate >= kSparseThreshold && is_enable_sparse) {
    *is_sparse = true;
    return CreateSparseBin(num_data, num_bin, default_bin);
  } else {
    *is_sparse = false;
    return CreateDenseBin(num_data, num_bin, default_bin);
  }
}

Bin* Bin::CreateDenseBin(data_size_t num_data, int num_bin, int default_bin) {
  if (num_bin <= 256) {
    return new DenseBin<uint8_t>(num_data, default_bin);
  } else if (num_bin <= 65536) {
    return new DenseBin<uint16_t>(num_data, default_bin);
  } else {
    return new DenseBin<uint32_t>(num_data, default_bin);
  }
}

Bin* Bin::CreateSparseBin(data_size_t num_data, int num_bin, int default_bin) {
  if (num_bin <= 256) {
    return new SparseBin<uint8_t>(num_data, default_bin);
  } else if (num_bin <= 65536) {
    return new SparseBin<uint16_t>(num_data, default_bin);
  } else {
    return new SparseBin<uint32_t>(num_data, default_bin);
  }
}

}  // namespace LightGBM
