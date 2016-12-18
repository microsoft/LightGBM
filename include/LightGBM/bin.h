#ifndef LIGHTGBM_BIN_H_
#define LIGHTGBM_BIN_H_

#include <LightGBM/meta.h>

#include <vector>
#include <functional>
#include <unordered_map>

namespace LightGBM {

enum BinType {
  NumericalBin,
  CategoricalBin
};

/*! \brief Store data for one histogram bin */
struct HistogramBinEntry {
public:
  /*! \brief Sum of gradients on this bin */
  double sum_gradients = 0.0;
  /*! \brief Sum of hessians on this bin */
  double sum_hessians = 0.0;
  /*! \brief Number of data on this bin */
  data_size_t cnt = 0;

  /*!
  * \brief Sum up (reducers) functions for histogram bin
  */
  inline static void SumReducer(const char *src, char *dst, int len) {
    const int type_size = sizeof(HistogramBinEntry);
    int used_size = 0;
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
    if (bin_type_ != other.bin_type_) {
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
  /*! \brief True if bin is trival (contains only one bin) */
  inline bool is_trival() const { return is_trival_; }
  /*! \brief Sparsity of this bin ( num_zero_bins / num_data ) */
  inline double sparse_rate() const { return sparse_rate_; }
  /*!
  * \brief Save binary data to file
  * \param file File want to write
  */
  void SaveBinaryToFile(FILE* file) const;
  /*!
  * \brief Mapping bin into feature value
  * \param bin
  * \return Feature value of this bin
  */
  inline double BinToValue(unsigned int bin) const {
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
  inline unsigned int ValueToBin(double value) const;

  /*!
  * \brief Get the default bin when value is 0 or is firt categorical
  * \return default bin
  */
  inline uint32_t GetDefaultBin() const {
    if (bin_type_ == BinType::NumericalBin) {
      return ValueToBin(0);
    } else {
      return 0;
    }
  }
  /*!
  * \brief Construct feature value to bin mapper according feature values
  * \param values (Sampled) values of this feature
  * \param max_bin The maximal number of bin
  * \param bin_type Type of this bin
  */
  void FindBin(std::vector<double>* values, size_t total_sample_cnt, int max_bin, BinType bin_type);

  /*!
  * \brief Use specific number of bin to calculate the size of this class
  * \param bin The number of bin
  * \return Size
  */
  static int SizeForSpecificBin(int bin);

  /*!
  * \brief Seirilizing this object to buffer
  * \param buffer The destination
  */
  void CopyTo(char* buffer);

  /*!
  * \brief Deserilizing this object from buffer
  * \param buffer The source
  */
  void CopyFrom(const char* buffer);

  inline BinType bin_type() const { return bin_type_; }
private:
  /*! \brief Number of bins */
  int num_bin_;
  /*! \brief Store upper bound for each bin */
  std::vector<double> bin_upper_bound_;
  /*! \brief True if this feature is trival */
  bool is_trival_;
  /*! \brief Sparse rate of this bins( num_bin0/num_data ) */
  double sparse_rate_;
  /*! \brief Type of this bin */
  BinType bin_type_;
  /*! \brief Mapper from categorical to bin */
  std::unordered_map<int, unsigned int> categorical_2_bin_;
  /*! \brief Mapper from bin to categorical */
  std::vector<int> bin_2_categorical_;
};

/*!
* \brief Interface for ordered bin data. efficient for construct histogram, especially for sparse bin
*        There are 2 advantages by using ordered bin.
*        1. group the data by leafs to improve the cache hit.
*        2. only store the non-zero bin, which can speed up the histogram consturction for sparse features.
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
  virtual void Init(const char* used_idices, data_size_t num_leaves) = 0;

  /*!
  * \brief Construct histogram by using this bin
  *        Note: Unlike Bin, OrderedBin doesn't use ordered gradients and ordered hessians.
  *        Because it is hard to know the relative index in one leaf for sparse bin, since we skipped zero bins.
  * \param leaf Using which leaf's data to construct
  * \param gradients Gradients, Note:non-oredered by leaf
  * \param hessians Hessians, Note:non-oredered by leaf
  * \param out Output Result
  */
  virtual void ConstructHistogram(int leaf, const score_t* gradients,
    const score_t* hessians, HistogramBinEntry* out) const = 0;

  /*!
  * \brief Split current bin, and perform re-order by leaf
  * \param leaf Using which leaf's to split
  * \param right_leaf The new leaf index after perform this split
  * \param left_indices left_indices[i] == true means the i-th data will be on left leaf after split
  */
  virtual void Split(int leaf, int right_leaf, const char* left_indices) = 0;
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

  /*!
  * \brief Get bin interator of this bin
  * \param start_idx start index of this 
  * \return Iterator of this bin
  */
  virtual BinIterator* GetIterator(data_size_t start_idx) const = 0;

  /*!
  * \brief Save binary data to file
  * \param file File want to write
  */
  virtual void SaveBinaryToFile(FILE* file) const = 0;

  /*!
  * \brief Load from memory
  * \param file File want to write
  */
  virtual void LoadFromMemory(const void* memory,
    const std::vector<data_size_t>& local_used_indices) = 0;

  /*!
  * \brief Get sizes in byte of this object
  */
  virtual size_t SizesInByte() const = 0;

  /*! \brief Number of all data */
  virtual data_size_t num_data() const = 0;

  /*!
  * \brief Construct histogram of this feature,
  *        Note: We use ordered_gradients and ordered_hessians to improve cache hit chance
  *        The navie solution is use gradients[data_indices[i]] for data_indices[i] to get gradients, 
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

  /*!
  * \brief Split data according to threshold, if bin <= threshold, will put into left(lte_indices), else put into right(gt_indices)
  * \param threshold The split threshold.
  * \param data_indices Used data indices. After called this function. The less than or equal data indices will store on this object.
  * \param num_data Number of used data
  * \param lte_indices After called this function. The less or equal data indices will store on this object.
  * \param gt_indices After called this function. The greater data indices will store on this object.
  * \return The number of less than or equal data.
  */
  virtual data_size_t Split(
    unsigned int threshold,
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
  * \param is_sparse True if this feature is sparse
  * \param sparse_rate Sparse rate of this bins( num_bin0/num_data )
  * \param is_enable_sparse True if enable sparse feature
  * \param is_sparse Will set to true if this bin is sparse
  * \param default_bin Default bin for zeros value
  * \param bin_type type of bin
  * \return The bin data object
  */
  static Bin* CreateBin(data_size_t num_data, int num_bin,
    double sparse_rate, bool is_enable_sparse, 
    bool* is_sparse, int default_bin, BinType bin_type);

  /*!
  * \brief Create object for bin data of one feature, used for dense feature
  * \param num_data Total number of data
  * \param num_bin Number of bin
  * \param default_bin Default bin for zeros value
  * \param bin_type type of bin
  * \return The bin data object
  */
  static Bin* CreateDenseBin(data_size_t num_data, int num_bin, 
    int default_bin, BinType bin_type);

  /*!
  * \brief Create object for bin data of one feature, used for sparse feature
  * \param num_data Total number of data
  * \param num_bin Number of bin
  * \param default_bin Default bin for zeros value
  * \param bin_type type of bin
  * \return The bin data object
  */
  static Bin* CreateSparseBin(data_size_t num_data,
    int num_bin, int default_bin, BinType bin_type);
};

inline unsigned int BinMapper::ValueToBin(double value) const {
  // binary search to find bin
  if (bin_type_ == BinType::NumericalBin) {
    int l = 0;
    int r = num_bin_ - 1;
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
    if (categorical_2_bin_.count(int_value)) {
      return categorical_2_bin_.at(int_value);
    } else {
      return num_bin_ - 1;
    }
  }
}

}  // namespace LightGBM

#endif   // LightGBM_BIN_H_
