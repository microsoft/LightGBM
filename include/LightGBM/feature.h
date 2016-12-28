#ifndef LIGHTGBM_FEATURE_H_
#define LIGHTGBM_FEATURE_H_

#include <LightGBM/utils/random.h>

#include <LightGBM/meta.h>
#include <LightGBM/bin.h>

#include <cstdio>
#include <memory>
#include <vector>

namespace LightGBM {

/*! \brief Using to store data and providing some operations on one feature*/
class Feature {
public:
  /*!
  * \brief Constructor
  * \param feature_idx Index of this feature
  * \param bin_mapper Bin mapper for this feature
  * \param num_data Total number of data
  * \param is_enable_sparse True if enable sparse feature
  */
  Feature(int feature_idx, BinMapper* bin_mapper,
    data_size_t num_data, bool is_enable_sparse)
    :bin_mapper_(bin_mapper) {
    feature_index_ = feature_idx;
    bin_data_.reset(Bin::CreateBin(num_data, bin_mapper_->num_bin(),
      bin_mapper_->sparse_rate(), is_enable_sparse, &is_sparse_, bin_mapper_->GetDefaultBin(), bin_mapper_->bin_type()));
  }
  /*!
  * \brief Constructor from memory
  * \param memory Pointer of memory
  * \param num_all_data Number of global data
  * \param local_used_indices Local used indices, empty means using all data
  */
  Feature(const void* memory, data_size_t num_all_data,
    const std::vector<data_size_t>& local_used_indices) {
    const char* memory_ptr = reinterpret_cast<const char*>(memory);
    // get featuer index
    feature_index_ = *(reinterpret_cast<const int*>(memory_ptr));
    memory_ptr += sizeof(feature_index_);
    // get is_sparse
    is_sparse_ = *(reinterpret_cast<const bool*>(memory_ptr));
    memory_ptr += sizeof(is_sparse_);
    // get bin mapper
    bin_mapper_.reset(new BinMapper(memory_ptr));
    memory_ptr += bin_mapper_->SizesInByte();
    data_size_t num_data = num_all_data;
    if (!local_used_indices.empty()) {
      num_data = static_cast<data_size_t>(local_used_indices.size());
    }
    if (is_sparse_) {
      bin_data_.reset(Bin::CreateSparseBin(num_data, bin_mapper_->num_bin(), bin_mapper_->GetDefaultBin(), bin_mapper_->bin_type()));
    } else {
      bin_data_.reset(Bin::CreateDenseBin(num_data, bin_mapper_->num_bin(), bin_mapper_->GetDefaultBin(), bin_mapper_->bin_type()));
    }
    // get bin data
    bin_data_->LoadFromMemory(memory_ptr, local_used_indices);
  }
  /*! \brief Destructor */
  ~Feature() {
  }

  bool CheckAlign(const Feature& other) const {
    if (feature_index_ != other.feature_index_) {
      return false;
    }
    return bin_mapper_->CheckAlign(*(other.bin_mapper_.get()));
  }

  /*!
  * \brief Push one record, will auto convert to bin and push to bin data
  * \param tid Thread id
  * \param idx Index of record
  * \param value feature value of record
  */
  inline void PushData(int tid, data_size_t line_idx, double value) {
    unsigned int bin = bin_mapper_->ValueToBin(value);
    bin_data_->Push(tid, line_idx, bin);
  }
  inline void PushBin(int tid, data_size_t line_idx, unsigned int bin) {
    bin_data_->Push(tid, line_idx, bin);
  }
  inline void FinishLoad() { bin_data_->FinishLoad(); }
  /*! \brief Index of this feature */
  inline int feature_index() const { return feature_index_; }
  /*! \brief Bin mapper that this feature used */
  inline const BinMapper* bin_mapper() const { return bin_mapper_.get(); }
  /*! \brief Number of bin of this feature */
  inline int num_bin() const { return bin_mapper_->num_bin(); }

  inline BinType bin_type() const { return bin_mapper_->bin_type(); }
  /*! \brief Get bin data of this feature */
  inline const Bin* bin_data() const { return bin_data_.get(); }
  /*!
  * \brief From bin to feature value
  * \param bin
  * \return Feature value of this bin
  */
  inline double BinToValue(unsigned int bin)
    const { return bin_mapper_->BinToValue(bin); }

  /*!
  * \brief Save binary data to file
  * \param file File want to write
  */
  void SaveBinaryToFile(FILE* file) const {
    fwrite(&feature_index_, sizeof(feature_index_), 1, file);
    fwrite(&is_sparse_, sizeof(is_sparse_), 1, file);
    bin_mapper_->SaveBinaryToFile(file);
    bin_data_->SaveBinaryToFile(file);
  }
  /*!
  * \brief Get sizes in byte of this object
  */
  size_t SizesInByte() const {
    return sizeof(feature_index_) + sizeof(is_sparse_) +
      bin_mapper_->SizesInByte() + bin_data_->SizesInByte();
  }
  /*! \brief Disable copy */
  Feature& operator=(const Feature&) = delete;
  /*! \brief Disable copy */
  Feature(const Feature&) = delete;

private:
  /*! \brief Index of this feature */
  int feature_index_;
  /*! \brief Bin mapper that this feature used */
  std::unique_ptr<BinMapper> bin_mapper_;
  /*! \brief Bin data of this feature */
  std::unique_ptr<Bin> bin_data_;
  /*! \brief True if this feature is sparse */
  bool is_sparse_;
};


}  // namespace LightGBM

#endif   // LightGBM_FEATURE_H_
