#ifndef LIGHTGBM_UTILS_LRU_POOL_H_
#define LIGHTGBM_UTILS_LRU_POOL_H_

#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/log.h>

#include <cstring>
#include <functional>
#include <vector>
#include <memory>

namespace LightGBM {

/*!
* \brief A LRU cached object pool, used for store historical histograms
*/
template<typename T>
class LRUPool {
public:

  /*!
  * \brief Constructor
  */
  LRUPool() {
  }

  /*!
  * \brief Destructor
  */
  ~LRUPool() {
  }
  /*!
  * \brief Reset pool size
  * \param cache_size Max cache size
  * \param total_size Total size will be used
  */
  void ResetSize(int cache_size, int total_size) {
    cache_size_ = cache_size;
    // at least need 2 bucket to store smaller leaf and larger leaf
    CHECK(cache_size_ >= 2);
    total_size_ = total_size;
    if (cache_size_ > total_size_) {
      cache_size_ = total_size_;
    }
    is_enough_ = (cache_size_ == total_size_);
    pool_ = std::vector<T>(cache_size_);
    if (!is_enough_) {
      mapper_ = std::vector<int>(total_size_);
      inverse_mapper_ = std::vector<int>(cache_size_);
      last_used_time_ = std::vector<int>(cache_size_);
      ResetMap();
    }
  }

  /*!
  * \brief Reset mapper
  */
  void ResetMap() {
    if (!is_enough_) {
      cur_time_ = 0;
      std::fill(mapper_.begin(), mapper_.end(), -1);
      std::fill(inverse_mapper_.begin(), inverse_mapper_.end(), -1);
      std::fill(last_used_time_.begin(), last_used_time_.end(), 0);
    }
  }

  /*!
  * \brief Fill the pool
  * \param obj_create_fun that used to generate object
  */
  void Fill(std::function<T()> obj_create_fun) {
    for (int i = 0; i < cache_size_; ++i) {
      pool_[i] = std::move(obj_create_fun());
    }
  }

  /*!
  * \brief Get data for the specific index
  * \param idx which index want to get 
  * \param out output data will store into this
  * \return True if this index is in the pool, False if this index is not in the pool
  */
  bool Get(int idx, T* out) {
    if (is_enough_) {
      *out = pool_[idx];
      return true;
    }
    else if (mapper_[idx] >= 0) {
      int slot = mapper_[idx];
      *out = pool_[slot];
      last_used_time_[slot] = ++cur_time_;
      return true;
    } else {
      // choose the least used slot 
      int slot = static_cast<int>(ArrayArgs<int>::ArgMin(last_used_time_));
      *out = pool_[slot];
      last_used_time_[slot] = ++cur_time_;

      // reset previous mapper
      if (inverse_mapper_[slot] >= 0) mapper_[inverse_mapper_[slot]] = -1;

      // update current mapper
      mapper_[idx] = slot;
      inverse_mapper_[slot] = idx;
      return false;
    }
  }

  /*!
  * \brief Move data from one index to another index
  * \param src_idx 
  * \param dst_idx 
  */
  void Move(int src_idx, int dst_idx) {
    if (is_enough_) {
      std::swap(pool_[src_idx], pool_[dst_idx]);
      return;
    }
    if (mapper_[src_idx] < 0) {
      return;
    }
    // get slot of src idx
    int slot = mapper_[src_idx];
    // reset src_idx
    mapper_[src_idx] = -1;

    // move to dst idx
    mapper_[dst_idx] = slot;
    last_used_time_[slot] = ++cur_time_;
    inverse_mapper_[slot] = dst_idx;
  }
private:

  std::vector<T> pool_;
  int cache_size_;
  int total_size_;
  bool is_enough_ = false;
  std::vector<int> mapper_;
  std::vector<int> inverse_mapper_;
  std::vector<int> last_used_time_;
  int cur_time_ = 0;
};

}

#endif  // LIGHTGBM_UTILS_LRU_POOL_H_
