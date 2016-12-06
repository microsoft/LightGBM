#include <LightGBM/utils/common.h>
#include <LightGBM/bin.h>

#include "dense_bin.hpp"
#include "sparse_bin.hpp"
#include "ordered_sparse_bin.hpp"

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
  bin_type_ = other.bin_type_;
  if (bin_type_ == BinType::NumericalBin) {
    bin_upper_bound_ = other.bin_upper_bound_;
  } else {
    bin_2_categorical_ = other.bin_2_categorical_;
    categorical_2_bin_ = other.categorical_2_bin_;
  }

}

BinMapper::BinMapper(const void* memory) {
  CopyFrom(reinterpret_cast<const char*>(memory));
}

BinMapper::~BinMapper() {

}

void BinMapper::FindBin(std::vector<double>* values, size_t total_sample_cnt, int max_bin, BinType bin_type) {
  bin_type_ = bin_type;
  std::vector<double>& ref_values = (*values);
  size_t sample_size = total_sample_cnt;
  int zero_cnt = static_cast<int>(total_sample_cnt - ref_values.size());
  // find distinct_values first
  std::vector<double> distinct_values;
  std::vector<int> counts;

  std::sort(ref_values.begin(), ref_values.end());

  // push zero in the front
  if (ref_values.empty() || (ref_values[0] > 0.0f && zero_cnt > 0)) {
    distinct_values.push_back(0);
    counts.push_back(zero_cnt);
  }

  if (!ref_values.empty()) {
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
  if (!ref_values.empty() && ref_values.back() < 0.0f && zero_cnt > 0) {
    distinct_values.push_back(0);
    counts.push_back(zero_cnt);
  }

  int num_values = static_cast<int>(distinct_values.size());
  int cnt_in_bin0 = 0;
  if (bin_type_ == BinType::NumericalBin) {
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
      int rest_bin_cnt = max_bin;
      int rest_sample_cnt = static_cast<int>(sample_size);
      std::vector<bool> is_big_count_value(num_values, false);
      for (int i = 0; i < num_values; ++i) {
        if (counts[i] >= mean_bin_size) {
          is_big_count_value[i] = true;
          --rest_bin_cnt;
          rest_sample_cnt -= counts[i];
        }
      }
      mean_bin_size = rest_sample_cnt / static_cast<double>(rest_bin_cnt);

      std::vector<double> upper_bounds(max_bin, std::numeric_limits<double>::infinity());
      std::vector<double> lower_bounds(max_bin, std::numeric_limits<double>::infinity());

      int bin_cnt = 0;
      lower_bounds[bin_cnt] = distinct_values[0];
      int cur_cnt_inbin = 0;
      for (int i = 0; i < num_values - 1; ++i) {
        if (!is_big_count_value[i]) {
          rest_sample_cnt -= counts[i];
        }
        cur_cnt_inbin += counts[i];
        // need a new bin
        if (is_big_count_value[i] || cur_cnt_inbin >= mean_bin_size ||
          (is_big_count_value[i + 1] && cur_cnt_inbin >= std::max(1.0, mean_bin_size * 0.5f))) {
          upper_bounds[bin_cnt] = distinct_values[i];
          if (bin_cnt == 0) {
            cnt_in_bin0 = cur_cnt_inbin;
          }
          ++bin_cnt;
          lower_bounds[bin_cnt] = distinct_values[i + 1];
          if (bin_cnt >= max_bin - 1) { break; }
          cur_cnt_inbin = 0;
          if (!is_big_count_value[i]) {
            --rest_bin_cnt;
            mean_bin_size = rest_sample_cnt / static_cast<double>(rest_bin_cnt);
          }
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

  } else {
    // convert to int type first
    std::vector<int> distinct_values_int;
    std::vector<int> counts_int;
    distinct_values_int.push_back(static_cast<int>(distinct_values[0]));
    counts_int.push_back(counts[0]);
    for (size_t i = 1; i < distinct_values.size(); ++i) {
      if (static_cast<int>(distinct_values[i]) != distinct_values_int.back()) {
        distinct_values_int.push_back(static_cast<int>(distinct_values[i]));
        counts_int.push_back(counts[i]);
      } else {
        counts_int.back() += counts[i];
      }
    }
    // sort by counts
    Common::SortForPair<int, int>(counts_int, distinct_values_int, 0, true);
    // will ingore the categorical of small counts
    num_bin_ = std::min(max_bin, static_cast<int>(counts_int.size()));
    categorical_2_bin_.clear();
    bin_2_categorical_ = std::vector<int>(num_bin_);
    int used_cnt = 0;
    for (int i = 0; i < num_bin_; ++i) {
      bin_2_categorical_[i] = distinct_values_int[i];
      categorical_2_bin_[distinct_values_int[i]] = static_cast<unsigned int>(i);
      used_cnt += counts_int[i];
    }
    if (used_cnt / static_cast<double>(sample_size) < 0.95f) {
      Log::Warning("Too many categoricals are ignored, \
                   please use bigger max_bin or partition this column ");
    }
    cnt_in_bin0 = static_cast<int>(sample_size) - used_cnt + counts_int[0];
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
  size += sizeof(BinType);
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
  std::memcpy(buffer, &bin_type_, sizeof(bin_type_));
  buffer += sizeof(bin_type_);
  if (bin_type_ == BinType::NumericalBin) {
    std::memcpy(buffer, bin_upper_bound_.data(), num_bin_ * sizeof(double));
  } else {
    std::memcpy(buffer, bin_2_categorical_.data(), num_bin_ * sizeof(int));
  }
}

void BinMapper::CopyFrom(const char * buffer) {
  std::memcpy(&num_bin_, buffer, sizeof(num_bin_));
  buffer += sizeof(num_bin_);
  std::memcpy(&is_trival_, buffer, sizeof(is_trival_));
  buffer += sizeof(is_trival_);
  std::memcpy(&sparse_rate_, buffer, sizeof(sparse_rate_));
  buffer += sizeof(sparse_rate_);
  std::memcpy(&bin_type_, buffer, sizeof(bin_type_));
  buffer += sizeof(bin_type_);
  if (bin_type_ == BinType::NumericalBin) {
    bin_upper_bound_ = std::vector<double>(num_bin_);
    std::memcpy(bin_upper_bound_.data(), buffer, num_bin_ * sizeof(double));
  } else {
    bin_2_categorical_ = std::vector<int>(num_bin_);
    std::memcpy(bin_2_categorical_.data(), buffer, num_bin_ * sizeof(int));
    categorical_2_bin_.clear();
    for (int i = 0; i < num_bin_; ++i) {
      categorical_2_bin_[bin_2_categorical_[i]] = static_cast<unsigned int>(i);
    }
  }
}

void BinMapper::SaveBinaryToFile(FILE* file) const {
  fwrite(&num_bin_, sizeof(num_bin_), 1, file);
  fwrite(&is_trival_, sizeof(is_trival_), 1, file);
  fwrite(&sparse_rate_, sizeof(sparse_rate_), 1, file);
  fwrite(&bin_type_, sizeof(bin_type_), 1, file);
  if (bin_type_ == BinType::NumericalBin) {
    fwrite(bin_upper_bound_.data(), sizeof(double), num_bin_, file);
  } else {
    fwrite(bin_2_categorical_.data(), sizeof(int), num_bin_, file);
  }
}

size_t BinMapper::SizesInByte() const {
  size_t ret = sizeof(num_bin_) + sizeof(is_trival_) + sizeof(sparse_rate_)
    + sizeof(bin_type_);
  if (bin_type_ == BinType::NumericalBin) {
    ret += sizeof(double) *  num_bin_;
  } else {
    ret += sizeof(int) * num_bin_;
  }
  return ret;
}

template class DenseBin<uint8_t>;
template class DenseBin<uint16_t>;
template class DenseBin<uint32_t>;

template class DenseCategoricalBin<uint8_t>;
template class DenseCategoricalBin<uint16_t>;
template class DenseCategoricalBin<uint32_t>;

template class SparseBin<uint8_t>;
template class SparseBin<uint16_t>;
template class SparseBin<uint32_t>;

template class SparseCategoricalBin<uint8_t>;
template class SparseCategoricalBin<uint16_t>;
template class SparseCategoricalBin<uint32_t>;

template class OrderedSparseBin<uint8_t>;
template class OrderedSparseBin<uint16_t>;
template class OrderedSparseBin<uint32_t>;


Bin* Bin::CreateBin(data_size_t num_data, int num_bin, double sparse_rate, 
  bool is_enable_sparse, bool* is_sparse, int default_bin, BinType bin_type) {
  // sparse threshold
  const double kSparseThreshold = 0.8f;
  if (sparse_rate >= kSparseThreshold && is_enable_sparse) {
    *is_sparse = true;
    return CreateSparseBin(num_data, num_bin, default_bin, bin_type);
  } else {
    *is_sparse = false;
    return CreateDenseBin(num_data, num_bin, default_bin, bin_type);
  }
}

Bin* Bin::CreateDenseBin(data_size_t num_data, int num_bin, int default_bin, BinType bin_type) {
  if (bin_type == BinType::NumericalBin) {
    if (num_bin <= 256) {
      return new DenseBin<uint8_t>(num_data, default_bin);
    } else if (num_bin <= 65536) {
      return new DenseBin<uint16_t>(num_data, default_bin);
    } else {
      return new DenseBin<uint32_t>(num_data, default_bin);
    }
  } else {
    if (num_bin <= 256) {
      return new DenseCategoricalBin<uint8_t>(num_data, default_bin);
    } else if (num_bin <= 65536) {
      return new DenseCategoricalBin<uint16_t>(num_data, default_bin);
    } else {
      return new DenseCategoricalBin<uint32_t>(num_data, default_bin);
    }
  }
}

Bin* Bin::CreateSparseBin(data_size_t num_data, int num_bin, int default_bin, BinType bin_type) {
  if (bin_type == BinType::NumericalBin) {
    if (num_bin <= 256) {
      return new SparseBin<uint8_t>(num_data, default_bin);
    } else if (num_bin <= 65536) {
      return new SparseBin<uint16_t>(num_data, default_bin);
    } else {
      return new SparseBin<uint32_t>(num_data, default_bin);
    }
  } else {
    if (num_bin <= 256) {
      return new SparseCategoricalBin<uint8_t>(num_data, default_bin);
    } else if (num_bin <= 65536) {
      return new SparseCategoricalBin<uint16_t>(num_data, default_bin);
    } else {
      return new SparseCategoricalBin<uint32_t>(num_data, default_bin);
    }
  }
}

}  // namespace LightGBM
