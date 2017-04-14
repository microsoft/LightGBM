#ifndef LIGHTGBM_META_H_
#define LIGHTGBM_META_H_

#include <cstdint>

#include <limits>
#include <vector>
#include <functional>
#include <memory>
#include <cstdlib>

#if defined(_WIN32)

#include <malloc.h>

#else

#include <mm_malloc.h>

#endif // (_WIN32)



namespace LightGBM {

/*! \brief Type of data size, it is better to use signed type*/
typedef int32_t data_size_t;

const float kMinScore = -std::numeric_limits<float>::infinity();

const float kEpsilon = 1e-15f;

using ReduceFunction = std::function<void(const char*, char*, int)>;

using PredictFunction =
std::function<void(const std::vector<std::pair<int, double>>&, double* output)>;

#define NO_SPECIFIC (-1)

template <typename T, std::size_t Alignment>
class aligned_allocator
{
public:
  typedef T * pointer;
  typedef const T * const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;
  typedef std::size_t size_type;
  typedef int64_t difference_type;

  T * address(T& r) const {
    return &r;
  }

  const T * address(const T& s) const {
    return &s;
  }

  std::size_t max_size() const {
    return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
  }

  template <typename U>
  struct rebind {
    typedef aligned_allocator<U, Alignment> other;
  };

  bool operator!=(const aligned_allocator& other) const {
    return !(*this == other);
  }

  void construct(T * const p, const T& t) const {
    void * const pv = static_cast<void *>(p);

    new (pv) T(t);
  }

  void destroy(T * const p) const {
    p->~T();
  }

  bool operator==(const aligned_allocator& other) const {
    return true;
  }

  aligned_allocator() { }

  aligned_allocator(const aligned_allocator&) { }

  template <typename U> aligned_allocator(const aligned_allocator<U, Alignment>&) { }

  ~aligned_allocator() { }

  T * allocate(const std::size_t n) const {
    if (n == 0) {
      return NULL;
    }
    if (n > max_size()) {
      throw std::length_error("aligned_allocator<T>::allocate() - Integer overflow.");
    }

    void * const pv = _mm_malloc(n * sizeof(T), Alignment);
    if (pv == NULL) {
      throw std::bad_alloc();
    }

    return static_cast<T *>(pv);
  }

  void deallocate(T * const p, const std::size_t) const {
    _mm_free(p);
  }

  template <typename U>
  T * allocate(const std::size_t n, const U * /* const hint */) const {
    return allocate(n);
  }

private:
  aligned_allocator& operator=(const aligned_allocator&) = delete;
};

}  // namespace LightGBM

#endif   // LightGBM_META_H_
