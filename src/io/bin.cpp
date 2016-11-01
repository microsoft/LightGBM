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

BinMapper::BinMapper()
  :bin_upper_bound_(nullptr) {
}

// deep copy function for BinMapper
BinMapper::BinMapper(const BinMapper& other)
  : bin_upper_bound_(nullptr) {
  num_bin_ = other.num_bin_;
  is_trival_ = other.is_trival_;
  sparse_rate_ = other.sparse_rate_;
  bin_upper_bound_ = new float[num_bin_];
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

void BinMapper::FindBin(std::vector<float>* values, int max_bin) {
  std::vector<float>& ref_values = (*values);
  size_t sample_size = values->size();
  // find distinct_values first
  std::vector<float> distinct_values;
  std::vector<int> counts;
  
  std::sort(ref_values.begin(), ref_values.end());
  distinct_values.push_back(ref_values[0]);
  counts.push_back(1);
  for (size_t i = 1; i < ref_values.size(); ++i) {
    if (ref_values[i] != ref_values[i - 1]) {
      distinct_values.push_back(ref_values[i]);
      counts.push_back(1);
    } else {
      ++counts.back();
    }
  }
  int num_values = static_cast<int>(distinct_values.size());
  int cnt_in_bin0 = 0;

  if (num_values <= max_bin) {
    // use distinct value is enough
    num_bin_ = num_values;
    bin_upper_bound_ = new float[num_values];
    for (int i = 0; i < num_values - 1; ++i) {
      bin_upper_bound_[i] = (distinct_values[i] + distinct_values[i + 1]) / 2;
    }
    cnt_in_bin0 = counts[0];
    bin_upper_bound_[num_values - 1] = std::numeric_limits<float>::infinity();
  } else {
    // mean size for one bin
    float mean_bin_size = sample_size / static_cast<float>(max_bin);
    int rest_sample_cnt = static_cast<int>(sample_size);
    int bin_cnt = 0;

    num_bin_ = max_bin;
    std::vector<float> upper_bounds(max_bin, std::numeric_limits<float>::infinity());
    std::vector<float> lower_bounds(max_bin, std::numeric_limits<float>::infinity());
    // sort by count, descent
    Common::SortForPair(counts, distinct_values, 0, true);
    // fetch big slot as unique bin 
    while (counts[bin_cnt] > mean_bin_size) {
      upper_bounds[bin_cnt] = distinct_values[bin_cnt];
      lower_bounds[bin_cnt] = distinct_values[bin_cnt];
      rest_sample_cnt -= counts[bin_cnt];
      ++bin_cnt;
    }
    // process reminder bins
    if (bin_cnt < max_bin) {
      // sort rest by values
      Common::SortForPair<float, int>(distinct_values, counts, bin_cnt, false);
      mean_bin_size = rest_sample_cnt / static_cast<float>(max_bin - bin_cnt);
      lower_bounds[bin_cnt] = distinct_values[bin_cnt];
      int cur_cnt_inbin = 0;
      for (int i = bin_cnt; i < num_values - 1; ++i) {
        rest_sample_cnt -= counts[i];
        cur_cnt_inbin += counts[i];
        // need a new bin
        if (cur_cnt_inbin >= mean_bin_size) {
          upper_bounds[bin_cnt] = distinct_values[i];
          if (bin_cnt == 0) { cnt_in_bin0 = cur_cnt_inbin; }
          ++bin_cnt;
          lower_bounds[bin_cnt] = distinct_values[i + 1];
          if (bin_cnt >= max_bin - 1) break;
          cur_cnt_inbin = 0;
          mean_bin_size = rest_sample_cnt / static_cast<float>(max_bin - bin_cnt);
        }
      }
      cur_cnt_inbin += counts[num_values - 1];
      
    }
    Common::SortForPair<float, float>(lower_bounds, upper_bounds, 0, false);
    // update bin upper bound
    bin_upper_bound_ = new float[bin_cnt];
    for (int i = 0; i < bin_cnt - 1; ++i) {
      bin_upper_bound_[i] = (upper_bounds[i] + lower_bounds[i + 1]) / 2.0f;
    }

    // last bin upper bound
    bin_upper_bound_[bin_cnt - 1] = std::numeric_limits<float>::infinity();
    
    CHECK(bin_cnt <= max_bin);

  }
  // check trival(num_bin_ == 1) feature
  if (num_bin_ <= 1) {
    is_trival_ = true;
  } else {
    is_trival_ = false;
  }
  // calculate sparse rate
  sparse_rate_ = static_cast<float>(cnt_in_bin0) / static_cast<float>(sample_size);
}


int BinMapper::SizeForSpecificBin(int bin) {
  int size = 0;
  size += sizeof(int);
  size += sizeof(bool);
  size += sizeof(float);
  size += bin * sizeof(float);
  return size;
}

void BinMapper::CopyTo(char * buffer) {
  std::memcpy(buffer, &num_bin_, sizeof(num_bin_));
  buffer += sizeof(num_bin_);
  std::memcpy(buffer, &is_trival_, sizeof(is_trival_));
  buffer += sizeof(is_trival_);
  std::memcpy(buffer, &sparse_rate_, sizeof(sparse_rate_));
  buffer += sizeof(sparse_rate_);
  std::memcpy(buffer, bin_upper_bound_, num_bin_ * sizeof(float));
}

void BinMapper::CopyFrom(const char * buffer) {
  std::memcpy(&num_bin_, buffer, sizeof(num_bin_));
  buffer += sizeof(num_bin_);
  std::memcpy(&is_trival_, buffer, sizeof(is_trival_));
  buffer += sizeof(is_trival_);
  std::memcpy(&sparse_rate_, buffer, sizeof(sparse_rate_));
  buffer += sizeof(sparse_rate_);
  if (bin_upper_bound_ != nullptr) { delete[] bin_upper_bound_; }
  bin_upper_bound_ = new float[num_bin_];
  std::memcpy(bin_upper_bound_, buffer, num_bin_ * sizeof(float));
}

void BinMapper::SaveBinaryToFile(FILE* file) const {
  fwrite(&num_bin_, sizeof(num_bin_), 1, file);
  fwrite(&is_trival_, sizeof(is_trival_), 1, file);
  fwrite(&sparse_rate_, sizeof(sparse_rate_), 1, file);
  fwrite(bin_upper_bound_, sizeof(float), num_bin_, file);
}

size_t BinMapper::SizesInByte() const {
  return sizeof(num_bin_) + sizeof(is_trival_) + sizeof(sparse_rate_) + sizeof(float) * num_bin_;
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


Bin* Bin::CreateBin(data_size_t num_data, int num_bin, float sparse_rate, bool is_enable_sparse, bool* is_sparse, int default_bin) {
  // sparse threshold
  const float kSparseThreshold = 0.8f;
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
