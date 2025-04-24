#include <LightGBM/arrow.h>

#ifndef ARROW_TPP_
#define ARROW_TPP_

namespace LightGBM {

/**
 * @brief Obtain a function to access an index from an Arrow array.
 *
 * @tparam T The return type of the function, must be a primitive type.
 * @param dtype The Arrow format string describing the datatype of the Arrow array.
 * @return std::function<T(const ArrowArray*, size_t)> The index accessor function.
 */
template <typename T>
std::function<T(const ArrowArray*, size_t)> get_index_accessor(const char* dtype);

/* ---------------------------------- ITERATOR INITIALIZATION ---------------------------------- */

template <typename T>
inline ArrowChunkedArray::Iterator<T> ArrowChunkedArray::begin() const {
  return ArrowChunkedArray::Iterator<T>(*this, get_index_accessor<T>(schema_->format), 0);
}

template <typename T>
inline ArrowChunkedArray::Iterator<T> ArrowChunkedArray::end() const {
  return ArrowChunkedArray::Iterator<T>(*this, get_index_accessor<T>(schema_->format),
                                        chunk_offsets_.size() - 1);
}

/* ---------------------------------- ITERATOR IMPLEMENTATION ---------------------------------- */

template <typename T>
ArrowChunkedArray::Iterator<T>::Iterator(const ArrowChunkedArray& array, getter_fn get,
                                         int64_t ptr_chunk)
    : array_(array), get_(get), ptr_chunk_(ptr_chunk) {
  this->ptr_offset_ = 0;
}

template <typename T>
T ArrowChunkedArray::Iterator<T>::operator*() const {
  auto chunk = array_.chunks_[ptr_chunk_];
  return get_(chunk, ptr_offset_);
}

template <typename T>
template <typename I>
T ArrowChunkedArray::Iterator<T>::operator[](I idx) const {
  auto it = std::lower_bound(array_.chunk_offsets_.begin(), array_.chunk_offsets_.end(), idx,
                             [](int64_t a, int64_t b) { return a <= b; });

  auto chunk_idx = std::distance(array_.chunk_offsets_.begin() + 1, it);
  auto chunk = array_.chunks_[chunk_idx];

  auto ptr_offset = static_cast<int64_t>(idx) - array_.chunk_offsets_[chunk_idx];
  return get_(chunk, ptr_offset);
}

template <typename T>
ArrowChunkedArray::Iterator<T>& ArrowChunkedArray::Iterator<T>::operator++() {
  if (ptr_offset_ + 1 >= array_.chunks_[ptr_chunk_]->length) {
    ptr_offset_ = 0;
    ptr_chunk_++;
  } else {
    ptr_offset_++;
  }
  return *this;
}

template <typename T>
ArrowChunkedArray::Iterator<T>& ArrowChunkedArray::Iterator<T>::operator--() {
  if (ptr_offset_ == 0) {
    ptr_chunk_--;
    ptr_offset_ = array_.chunks_[ptr_chunk_]->length - 1;
  } else {
    ptr_chunk_--;
  }
  return *this;
}

template <typename T>
ArrowChunkedArray::Iterator<T>& ArrowChunkedArray::Iterator<T>::operator+=(int64_t c) {
  while (ptr_offset_ + c >= array_.chunks_[ptr_chunk_]->length) {
    c -= array_.chunks_[ptr_chunk_]->length - ptr_offset_;
    ptr_offset_ = 0;
    ptr_chunk_++;
  }
  ptr_offset_ += c;
  return *this;
}

template <typename T>
bool operator==(const ArrowChunkedArray::Iterator<T>& a, const ArrowChunkedArray::Iterator<T>& b) {
  return a.ptr_chunk_ == b.ptr_chunk_ && a.ptr_offset_ == b.ptr_offset_;
}

template <typename T>
bool operator!=(const ArrowChunkedArray::Iterator<T>& a, const ArrowChunkedArray::Iterator<T>& b) {
  return a.ptr_chunk_ != b.ptr_chunk_ || a.ptr_offset_ != b.ptr_offset_;
}

template <typename T>
int64_t operator-(const ArrowChunkedArray::Iterator<T>& a,
                  const ArrowChunkedArray::Iterator<T>& b) {
  auto full_offset_a = a.array_.chunk_offsets_[a.ptr_chunk_] + a.ptr_offset_;
  auto full_offset_b = b.array_.chunk_offsets_[b.ptr_chunk_] + b.ptr_offset_;
  return full_offset_a - full_offset_b;
}

/* --------------------------------------- INDEX ACCESSOR -------------------------------------- */

/**
 * @brief The value of "no value" for a primitive type.
 *
 * @tparam T The type for which the missing value is defined.
 * @return T The missing value.
 */
template <typename T>
inline T arrow_primitive_missing_value() {
  return 0;
}

template <>
inline double arrow_primitive_missing_value() {
  return std::numeric_limits<double>::quiet_NaN();
}

template <>
inline float arrow_primitive_missing_value() {
  return std::numeric_limits<float>::quiet_NaN();
}

template <typename T, typename V>
struct ArrayIndexAccessor {
  V operator()(const ArrowArray* array, size_t idx) {
    auto buffer_idx = idx + array->offset;

    // For primitive types, buffer at idx 0 provides validity, buffer at idx 1 data, see:
    // https://arrow.apache.org/docs/format/Columnar.html#buffer-listing-for-each-layout
    auto validity = static_cast<const char*>(array->buffers[0]);

    // Take return value from data buffer conditional on the validity of the index:
    //  - The structure of validity bitmasks is taken from here:
    //    https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps
    //  - If the bitmask is NULL, all indices are valid
    if (validity == nullptr || (validity[buffer_idx / 8] & (1 << (buffer_idx % 8)))) {
      // In case the index is valid, we take it from the data buffer
      auto data = static_cast<const T*>(array->buffers[1]);
      return static_cast<V>(data[buffer_idx]);
    }

    // In case the index is not valid, we return a default value
    return arrow_primitive_missing_value<V>();
  }
};

template <typename V>
struct ArrayIndexAccessor<bool, V> {
  V operator()(const ArrowArray* array, size_t idx) {
    // Custom implementation for booleans as values are bit-packed:
    // https://arrow.apache.org/docs/cpp/api/datatype.html#_CPPv4N5arrow4Type4type4BOOLE
    auto buffer_idx = idx + array->offset;
    auto validity = static_cast<const char*>(array->buffers[0]);
    if (validity == nullptr || (validity[buffer_idx / 8] & (1 << (buffer_idx % 8)))) {
      // In case the index is valid, we have to take the appropriate bit from the buffer
      auto data = static_cast<const char*>(array->buffers[1]);
      auto value = (data[buffer_idx / 8] & (1 << (buffer_idx % 8))) >> (buffer_idx % 8);
      return static_cast<V>(value);
    }
    return arrow_primitive_missing_value<V>();
  }
};

template <typename T>
std::function<T(const ArrowArray*, size_t)> get_index_accessor(const char* dtype) {
  // Mapping obtained from:
  // https://arrow.apache.org/docs/format/CDataInterface.html#data-type-description-format-strings
  switch (dtype[0]) {
    case 'c':
      return ArrayIndexAccessor<int8_t, T>();
    case 'C':
      return ArrayIndexAccessor<uint8_t, T>();
    case 's':
      return ArrayIndexAccessor<int16_t, T>();
    case 'S':
      return ArrayIndexAccessor<uint16_t, T>();
    case 'i':
      return ArrayIndexAccessor<int32_t, T>();
    case 'I':
      return ArrayIndexAccessor<uint32_t, T>();
    case 'l':
      return ArrayIndexAccessor<int64_t, T>();
    case 'L':
      return ArrayIndexAccessor<uint64_t, T>();
    case 'f':
      return ArrayIndexAccessor<float, T>();
    case 'g':
      return ArrayIndexAccessor<double, T>();
    case 'b':
      return ArrayIndexAccessor<bool, T>();
    default:
      throw std::invalid_argument("unsupported Arrow datatype");
  }
}

}  // namespace LightGBM

#endif
