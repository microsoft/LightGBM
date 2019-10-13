/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_BIN_H_
#define LIGHTGBM_BIN_H_

#include <LightGBM/meta.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/file_io.h>

#include <limits>
#include <string>
#include <functional>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace LightGBM {

enum BinType {
  NumericalBin,
  CategoricalBin
};

enum MissingType {
  None,
  Zero,
  NaN
};

/*! \brief Store data for one histogram bin */
struct HistogramBinEntry {
 public:
  /*! \brief Sum of gradients on this bin */
  double sum_gradients = 0.0f;
  /*! \brief Sum of hessians on this bin */
  double sum_hessians = 0.0f;
  /*! \brief Number of data on this bin */
  data_size_t cnt = 0;
  /*!
  * \brief Sum up (reducers) functions for histogram bin
  */
  inline static void SumReducer(const char *src, char *dst, int type_size, comm_size_t len) {
    comm_size_t used_size = 0;
    const HistogramBinEntry* p1;
    HistogramBinEntry* p2;
    while (used_size < len) {
      // convert
      p1 = reinterpret_cast<const HistogramBinEntry*>(src);
      p2 = reinterpret_cast<HistogramBinEntry*>(dst);
      // add
      p2->cnt += p1->cnt;
      p2->sum_gradients += p1->sum_gradients;
      p2->sum_hessians += p1->sum_hessians;
      src += type_size;
      dst += type_size;
      used_size += type_size;
    }
  }
};

/*! \brief This class used to convert feature values into bin,
*          and store some meta information for bin*/
class BinMapper {
 public:
  BinMapper();
  BinMapper(const BinMapper& other);
  explicit BinMapper(const void* memory);
  ~BinMapper();

  bool CheckAlign(const BinMapper& other) const {
    if (num_bin_ != other.num_bin_) {
      return false;
    }
    if (missing_type_ != other.missing_type_) {
      return false;
    }
    if (bin_type_ == BinType::NumericalBin) {
      for (int i = 0; i < num_bin_; ++i) {
        if (bin_upper_bound_[i] != other.bin_upper_bound_[i]) {
          return false;
        }
      }
    } else {
      for (int i = 0; i < num_bin_; i++) {
        if (bin_2_categorical_[i] != other.bin_2_categorical_[i]) {
          return false;
        }
      }
    }
    return true;
  }

  /*! \brief Get number of bins */
  inline int num_bin() const { return num_bin_; }
  /*! \brief Missing Type */
  inline MissingType missing_type() const { return missing_type_; }
  /*! \brief True if bin is trivial (contains only one bin) */
  inline bool is_trivial() const { return is_trivial_; }
  /*! \brief Sparsity of this bin ( num_zero_bins / num_data ) */
  inline double sparse_rate() const { return sparse_rate_; }
  /*!
  * \brief Save binary data to file
  * \param file File want to write
  */
  void SaveBinaryToFile(const VirtualFileWriter* writer) const;
  /*!
  * \brief Mapping bin into feature value
  * \param bin
  * \return Feature value of this bin
  */
  inline double BinToValue(uint32_t bin) const {
    if (bin_type_ == BinType::NumericalBin) {
      return bin_upper_bound_[bin];
    } else {
      return bin_2_categorical_[bin];
    }
  }
  /*!
  * \brief Get sizes in byte of this object
  */
  size_t SizesInByte() const;
  /*!
  * \brief Mapping feature value into bin
  * \param value
  * \return bin for this feature value
  */
  inline uint32_t ValueToBin(double value) const;

  /*!
  * \brief Get the default bin when value is 0
  * \return default bin
  */
  inline uint32_t GetDefaultBin() const {
    return default_bin_;
  }
  /*!
  * \brief Construct feature value to bin mapper according feature values
  * \param values (Sampled) values of this feature, Note: not include zero.
  * \param num_values number of values.
  * \param total_sample_cnt number of total sample count, equal with values.size() + num_zeros
  * \param max_bin The maximal number of bin
  * \param min_data_in_bin min number of data in one bin
  * \param min_split_data
  * \param bin_type Type of this bin
  * \param use_missing True to enable missing value handle
  * \param zero_as_missing True to use zero as missing value
  * \param forced_upper_bounds Vector of split points that must be used (if this has size less than max_bin, remaining splits are found by the algorithm)
  */
  void FindBin(double* values, int num_values, size_t total_sample_cnt, int max_bin, int min_data_in_bin, int min_split_data, BinType bin_type,
               bool use_missing, bool zero_as_missing, const std::vector<double>& forced_upper_bounds);

  /*!
  * \brief Use specific number of bin to calculate the size of this class
  * \param bin The number of bin
  * \return Size
  */
  static int SizeForSpecificBin(int bin);

  /*!
  * \brief Serializing this object to buffer
  * \param buffer The destination
  */
  void CopyTo(char* buffer) const;

  /*!
  * \brief Deserializing this object from buffer
  * \param buffer The source
  */
  void CopyFrom(const char* buffer);

  /*!
  * \brief Get bin types
  */
  inline BinType bin_type() const { return bin_type_; }

  /*!
  * \brief Get bin info
  */
  inline std::string bin_info() const {
    if (bin_type_ == BinType::CategoricalBin) {
      return Common::Join(bin_2_categorical_, ":");
    } else {
      std::stringstream str_buf;
      str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
      str_buf << '[' << min_val_ << ':' << max_val_ << ']';
      return str_buf.str();
    }
  }

 private:
  /*! \brief Number of bins */
  int num_bin_;
  MissingType missing_type_;
  /*! \brief Store upper bound for each bin */
  std::vector<double> bin_upper_bound_;
  /*! \brief True if this feature is trivial */
  bool is_trivial_;
  /*! \brief Sparse rate of this bins( num_bin0/num_data ) */
  double sparse_rate_;
  /*! \brief Type of this bin */
  BinType bin_type_;
  /*! \brief Mapper from categorical to bin */
  std::unordered_map<int, unsigned int> categorical_2_bin_;
  /*! \brief Mapper from bin to categorical */
  std::vector<int> bin_2_categorical_;
  /*! \brief minimal feature value */
  double min_val_;
  /*! \brief maximum feature value */
  double max_val_;
  /*! \brief bin value of feature value 0 */
  uint32_t default_bin_;
};

/*!
* \brief Interface for ordered bin data. efficient for construct histogram, especially for sparse bin
*        There are 2 advantages by using ordered bin.
*        1. group the data by leafs to improve the cache hit.
*        2. only store the non-zero bin, which can speed up the histogram construction for sparse features.
*        However it brings additional cost: it need re-order the bins after every split, which will cost much for dense feature.
*        So we only using ordered bin for sparse situations.
*/
class OrderedBin {
 public:
  /*! \brief virtual destructor */
  virtual ~OrderedBin() {}

  /*!
  * \brief Initialization logic.
  * \param used_indices If used_indices.size() == 0 means using all data, otherwise, used_indices[i] == true means i-th data is used
           (this logic was build for bagging logic)
  * \param num_leaves Number of leaves on this iteration
  */
  virtual void Init(const char* used_indices, data_size_t num_leaves) = 0;

  /*!
  * \brief Construct histogram by using this bin
  *        Note: Unlike Bin, OrderedBin doesn't use ordered gradients and ordered hessians.
  *        Because it is hard to know the relative index in one leaf for sparse bin, since we skipped zero bins.
  * \param leaf Using which leaf's data to construct
  * \param gradients Gradients, Note:non-ordered by leaf
  * \param hessians Hessians, Note:non-ordered by leaf
  * \param out Output Result
  */
  virtual void ConstructHistogram(int leaf, const score_t* gradients,
    const score_t* hessians, HistogramBinEntry* out) const = 0;

  /*!
  * \brief Construct histogram by using this bin
  *        Note: Unlike Bin, OrderedBin doesn't use ordered gradients and ordered hessians.
  *        Because it is hard to know the relative index in one leaf for sparse bin, since we skipped zero bins.
  * \param leaf Using which leaf's data to construct
  * \param gradients Gradients, Note:non-ordered by leaf
  * \param out Output Result
  */
  virtual void ConstructHistogram(int leaf, const score_t* gradients, HistogramBinEntry* out) const = 0;

  /*!
  * \brief Split current bin, and perform re-order by leaf
  * \param leaf Using which leaf's to split
  * \param right_leaf The new leaf index after perform this split
  * \param is_in_leaf is_in_leaf[i] == mark means the i-th data will be on left leaf after split
  * \param mark is_in_leaf[i] == mark means the i-th data will be on left leaf after split
  */
  virtual void Split(int leaf, int right_leaf, const char* is_in_leaf, char mark) = 0;

  virtual data_size_t NonZeroCount(int leaf) const = 0;
};

/*! \brief Iterator for one bin column */
class BinIterator {
 public:
  /*!
  * \brief Get bin data on specific row index
  * \param idx Index of this data
  * \return Bin data
  */
  virtual uint32_t Get(data_size_t idx) = 0;
  virtual uint32_t RawGet(data_size_t idx) = 0;
  virtual void Reset(data_size_t idx) = 0;
  virtual ~BinIterator() = default;
};

/*!
* \brief Interface for bin data. This class will store bin data for one feature.
*        unlike OrderedBin, this class will store data by original order.
*        Note that it may cause cache misses when construct histogram,
*        but it doesn't need to re-order operation, So it will be faster than OrderedBin for dense feature
*/
class Bin {
 public:
  /*! \brief virtual destructor */
  virtual ~Bin() {}
  /*!
  * \brief Push one record
  * \pram tid Thread id
  * \param idx Index of record
  * \param value bin value of record
  */
  virtual void Push(int tid, data_size_t idx, uint32_t value) = 0;


  virtual void CopySubset(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) = 0;
  /*!
  * \brief Get bin iterator of this bin for specific feature
  * \param min_bin min_bin of current used feature
  * \param max_bin max_bin of current used feature
  * \param default_bin default bin if bin not in [min_bin, max_bin]
  * \return Iterator of this bin
  */
  virtual BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t default_bin) const = 0;

  /*!
  * \brief Save binary data to file
  * \param file File want to write
  */
  virtual void SaveBinaryToFile(const VirtualFileWriter* writer) const = 0;

  /*!
  * \brief Load from memory
  * \param memory
  * \param local_used_indices
  */
  virtual void LoadFromMemory(const void* memory,
    const std::vector<data_size_t>& local_used_indices) = 0;

  /*!
  * \brief Get sizes in byte of this object
  */
  virtual size_t SizesInByte() const = 0;

  /*! \brief Number of all data */
  virtual data_size_t num_data() const = 0;

  virtual void ReSize(data_size_t num_data) = 0;

  /*!
  * \brief Construct histogram of this feature,
  *        Note: We use ordered_gradients and ordered_hessians to improve cache hit chance
  *        The naive solution is using gradients[data_indices[i]] for data_indices[i] to get gradients,
           which is not cache friendly, since the access of memory is not continuous.
  *        ordered_gradients and ordered_hessians are preprocessed, and they are re-ordered by data_indices.
  *        Ordered_gradients[i] is aligned with data_indices[i]'s gradients (same for ordered_hessians).
  * \param data_indices Used data indices in current leaf
  * \param num_data Number of used data
  * \param ordered_gradients Pointer to gradients, the data_indices[i]-th data's gradient is ordered_gradients[i]
  * \param ordered_hessians Pointer to hessians, the data_indices[i]-th data's hessian is ordered_hessians[i]
  * \param out Output Result
  */
  virtual void ConstructHistogram(
    const data_size_t* data_indices, data_size_t num_data,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    HistogramBinEntry* out) const = 0;

  virtual void ConstructHistogram(data_size_t num_data,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    HistogramBinEntry* out) const = 0;

  /*!
  * \brief Construct histogram of this feature,
  *        Note: We use ordered_gradients and ordered_hessians to improve cache hit chance
  *        The naive solution is using gradients[data_indices[i]] for data_indices[i] to get gradients,
  which is not cache friendly, since the access of memory is not continuous.
  *        ordered_gradients and ordered_hessians are preprocessed, and they are re-ordered by data_indices.
  *        Ordered_gradients[i] is aligned with data_indices[i]'s gradients (same for ordered_hessians).
  * \param data_indices Used data indices in current leaf
  * \param num_data Number of used data
  * \param ordered_gradients Pointer to gradients, the data_indices[i]-th data's gradient is ordered_gradients[i]
  * \param out Output Result
  */
  virtual void ConstructHistogram(const data_size_t* data_indices, data_size_t num_data,
                                  const score_t* ordered_gradients, HistogramBinEntry* out) const = 0;

  virtual void ConstructHistogram(data_size_t num_data,
                                  const score_t* ordered_gradients, HistogramBinEntry* out) const = 0;

  /*!
  * \brief Split data according to threshold, if bin <= threshold, will put into left(lte_indices), else put into right(gt_indices)
  * \param min_bin min_bin of current used feature
  * \param max_bin max_bin of current used feature
  * \param default_bin default bin if bin not in [min_bin, max_bin]
  * \param missing_type missing type
  * \param default_left missing bin will go to left child
  * \param threshold The split threshold.
  * \param data_indices Used data indices. After called this function. The less than or equal data indices will store on this object.
  * \param num_data Number of used data
  * \param lte_indices After called this function. The less or equal data indices will store on this object.
  * \param gt_indices After called this function. The greater data indices will store on this object.
  * \return The number of less than or equal data.
  */
  virtual data_size_t Split(uint32_t min_bin, uint32_t max_bin,
    uint32_t default_bin, MissingType missing_type, bool default_left, uint32_t threshold,
    data_size_t* data_indices, data_size_t num_data,
    data_size_t* lte_indices, data_size_t* gt_indices) const = 0;

  /*!
  * \brief Split data according to threshold, if bin <= threshold, will put into left(lte_indices), else put into right(gt_indices)
  * \param min_bin min_bin of current used feature
  * \param max_bin max_bin of current used feature
  * \param default_bin default bin if bin not in [min_bin, max_bin]
  * \param threshold The split threshold.
  * \param num_threshold Number of threshold
  * \param data_indices Used data indices. After called this function. The less than or equal data indices will store on this object.
  * \param num_data Number of used data
  * \param lte_indices After called this function. The less or equal data indices will store on this object.
  * \param gt_indices After called this function. The greater data indices will store on this object.
  * \return The number of less than or equal data.
  */
  virtual data_size_t SplitCategorical(uint32_t min_bin, uint32_t max_bin,
                            uint32_t default_bin, const uint32_t* threshold, int num_threshold,
                            data_size_t* data_indices, data_size_t num_data,
                            data_size_t* lte_indices, data_size_t* gt_indices) const = 0;

  /*!
  * \brief Create the ordered bin for this bin
  * \return Pointer to ordered bin
  */
  virtual OrderedBin* CreateOrderedBin() const = 0;

  /*!
  * \brief After pushed all feature data, call this could have better refactor for bin data
  */
  virtual void FinishLoad() = 0;

  /*!
  * \brief Create object for bin data of one feature, will call CreateDenseBin or CreateSparseBin according to "is_sparse"
  * \param num_data Total number of data
  * \param num_bin Number of bin
  * \param sparse_rate Sparse rate of this bins( num_bin0/num_data )
  * \param is_enable_sparse True if enable sparse feature
  * \param sparse_threshold Threshold for treating a feature as a sparse feature
  * \param is_sparse Will set to true if this bin is sparse
  * \param default_bin Default bin for zeros value
  * \return The bin data object
  */
  static Bin* CreateBin(data_size_t num_data, int num_bin,
    double sparse_rate, bool is_enable_sparse, double sparse_threshold, bool* is_sparse);

  /*!
  * \brief Create object for bin data of one feature, used for dense feature
  * \param num_data Total number of data
  * \param num_bin Number of bin
  * \return The bin data object
  */
  static Bin* CreateDenseBin(data_size_t num_data, int num_bin);

  /*!
  * \brief Create object for bin data of one feature, used for sparse feature
  * \param num_data Total number of data
  * \param num_bin Number of bin
  * \return The bin data object
  */
  static Bin* CreateSparseBin(data_size_t num_data, int num_bin);

  /*!
  * \brief Deep copy the bin
  */
  virtual Bin* Clone() = 0;
};

inline uint32_t BinMapper::ValueToBin(double value) const {
  if (std::isnan(value)) {
    if (missing_type_ == MissingType::NaN) {
      return num_bin_ - 1;
    } else {
      value = 0.0f;
    }
  }
  if (bin_type_ == BinType::NumericalBin) {
    // binary search to find bin
    int l = 0;
    int r = num_bin_ - 1;
    if (missing_type_ == MissingType::NaN) {
      r -= 1;
    }
    while (l < r) {
      int m = (r + l - 1) / 2;
      if (value <= bin_upper_bound_[m]) {
        r = m;
      } else {
        l = m + 1;
      }
    }
    return l;
  } else {
    int int_value = static_cast<int>(value);
    // convert negative value to NaN bin
    if (int_value < 0) {
      return num_bin_ - 1;
    }
    if (categorical_2_bin_.count(int_value)) {
      return categorical_2_bin_.at(int_value);
    } else {
      return num_bin_ - 1;
    }
  }
}

}  // namespace LightGBM

#endif   // LightGBM_BIN_H_
