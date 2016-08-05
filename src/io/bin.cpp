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

BinMapper::BinMapper()
  :bin_upper_bound_(nullptr) {
}

// deep copy function for BinMapper
BinMapper::BinMapper(const BinMapper& other)
  : bin_upper_bound_(nullptr) {
  num_bin_ = other.num_bin_;
  is_trival_ = other.is_trival_;
  sparse_rate_ = other.sparse_rate_;
  bin_upper_bound_ = new double[num_bin_];
  for (int i = 0; i < num_bin_; ++i) {
    bin_upper_bound_[i] = other.bin_upper_bound_[i];
  }
}

BinMapper::BinMapper(const void* memory)
  :bin_upper_bound_(nullptr) {
  CopyFrom(reinterpret_cast<const char*>(memory));
}

BinMapper::~BinMapper() {
  delete[] bin_upper_bound_;
}

void BinMapper::FindBin(std::vector<double>* values, int max_bin) {
  size_t sample_size = values->size();
  // find distinct_values first
  double* distinct_values = new double[sample_size];
  int *counts = new int[sample_size];
  int num_values = 1;
  std::sort(values->begin(), values->end());
  distinct_values[0] = (*values)[0];
  counts[0] = 1;
  for (size_t i = 1; i < values->size(); ++i) {
    if ((*values)[i] != (*values)[i - 1]) {
      distinct_values[num_values] = (*values)[i];
      counts[num_values] = 1;
      ++num_values;
    } else {
      ++counts[num_values - 1];
    }
  }
  int cnt_in_bin0 = 0;

  if (num_values <= max_bin) {
    // use distinct value is enough
    num_bin_ = num_values;
    bin_upper_bound_ = new double[num_values];
    for (int i = 0; i < num_values - 1; ++i) {
      bin_upper_bound_[i] = (distinct_values[i] + distinct_values[i + 1]) / 2;
    }
    cnt_in_bin0 = counts[0];
    bin_upper_bound_[num_values - 1] = std::numeric_limits<double>::infinity();
  } else {
    // need find bins
    num_bin_ = max_bin;
    bin_upper_bound_ = new double[max_bin];
    double * bin_lower_bound = new double[max_bin];
    // mean size for one bin
    double mean_bin_size = sample_size / static_cast<double>(max_bin);
    int rest_sample_cnt = static_cast<int>(sample_size);
    int cur_cnt_inbin = 0;
    int bin_cnt = 0;
    bin_lower_bound[0] = distinct_values[0];
    for (int i = 0; i < num_values - 1; ++i) {
      rest_sample_cnt -= counts[i];
      cur_cnt_inbin += counts[i];
      // need a new bin
      if (cur_cnt_inbin >= mean_bin_size) {
        bin_upper_bound_[bin_cnt] = distinct_values[i];
        if (bin_cnt == 0) { cnt_in_bin0 = cur_cnt_inbin; }
        ++bin_cnt;
        bin_lower_bound[bin_cnt] = distinct_values[i + 1];
        cur_cnt_inbin = 0;
        mean_bin_size = rest_sample_cnt / static_cast<double>(max_bin - bin_cnt);
      }
    }
    cur_cnt_inbin += counts[num_values - 1];
    // update bin upper bound
    for (int i = 0; i < bin_cnt; ++i) {
      bin_upper_bound_[i] = (bin_upper_bound_[i] + bin_lower_bound[i + 1]) / 2.0;
    }
    // last bin upper bound
    bin_upper_bound_[bin_cnt] = std::numeric_limits<double>::infinity();
    ++bin_cnt;
    delete[] bin_lower_bound;
    // if no so much bin
    if (bin_cnt < max_bin) {
      // old bin data
      double * tmp_bin_upper_bound = bin_upper_bound_;
      num_bin_ = bin_cnt;
      bin_upper_bound_ = new double[num_bin_];
      // copy back
      for (int i = 0; i < num_bin_; ++i) {
        bin_upper_bound_[i] = tmp_bin_upper_bound[i];
      }
      // free old space
      delete[] tmp_bin_upper_bound;
    }
  }
  delete[] distinct_values;
  delete[] counts;
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
  std::memcpy(buffer, bin_upper_bound_, num_bin_ * sizeof(double));
}

void BinMapper::CopyFrom(const char * buffer) {
  std::memcpy(&num_bin_, buffer, sizeof(num_bin_));
  buffer += sizeof(num_bin_);
  std::memcpy(&is_trival_, buffer, sizeof(is_trival_));
  buffer += sizeof(is_trival_);
  std::memcpy(&sparse_rate_, buffer, sizeof(sparse_rate_));
  buffer += sizeof(sparse_rate_);
  if (bin_upper_bound_ != nullptr) { delete[] bin_upper_bound_; }
  bin_upper_bound_ = new double[num_bin_];
  std::memcpy(bin_upper_bound_, buffer, num_bin_ * sizeof(double));
}

void BinMapper::SaveBinaryToFile(FILE* file) const {
  fwrite(&num_bin_, sizeof(num_bin_), 1, file);
  fwrite(&is_trival_, sizeof(is_trival_), 1, file);
  fwrite(&sparse_rate_, sizeof(sparse_rate_), 1, file);
  fwrite(bin_upper_bound_, sizeof(double), num_bin_, file);
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


Bin* Bin::CreateBin(data_size_t num_data, int num_bin, double sparse_rate, bool is_enable_sparse, bool* is_sparse) {
  // sparse threshold
  const double kSparseThreshold = 0.8;
  if (sparse_rate >= kSparseThreshold && is_enable_sparse) {
    *is_sparse = true;
    return CreateSparseBin(num_data, num_bin);
  } else {
    *is_sparse = false;
    return CreateDenseBin(num_data, num_bin);
  }
}

Bin* Bin::CreateDenseBin(data_size_t num_data, int num_bin) {
  if (num_bin <= 256) {
    return new DenseBin<uint8_t>(num_data);
  } else if (num_bin <= 65536) {
    return new DenseBin<uint16_t>(num_data);
  } else {
    return new DenseBin<uint32_t>(num_data);
  }
}

Bin* Bin::CreateSparseBin(data_size_t num_data, int num_bin) {
  if (num_bin <= 256) {
    return new SparseBin<uint8_t>(num_data);
  } else if (num_bin <= 65536) {
    return new SparseBin<uint16_t>(num_data);
  } else {
    return new SparseBin<uint32_t>(num_data);
  }
}

}  // namespace LightGBM
