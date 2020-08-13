// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 C. Antonio Sanchez <cantonios@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Utilities for generating random numbers without overflows, which might
// otherwise result in undefined behavior.

namespace Eigen {
namespace internal {

// Default implementation assuming SrcScalar fits into TgtScalar.
template <typename SrcScalar, typename TgtScalar, typename EnableIf = void>
struct random_without_cast_overflow {
  static SrcScalar value() { return internal::random<SrcScalar>(); }
};

// Signed to unsigned integer widening cast.
template <typename SrcScalar, typename TgtScalar>
struct random_without_cast_overflow<
    SrcScalar, TgtScalar,
    typename internal::enable_if<NumTraits<SrcScalar>::IsInteger && NumTraits<TgtScalar>::IsInteger &&
                                 !NumTraits<TgtScalar>::IsSigned &&
                                 (std::numeric_limits<SrcScalar>::digits < std::numeric_limits<TgtScalar>::digits ||
                                  (std::numeric_limits<SrcScalar>::digits == std::numeric_limits<TgtScalar>::digits &&
                                   NumTraits<SrcScalar>::IsSigned))>::type> {
  static SrcScalar value() {
    SrcScalar a = internal::random<SrcScalar>();
    return a < SrcScalar(0) ? -(a + 1) : a;
  }
};

// Integer to unsigned narrowing cast.
template <typename SrcScalar, typename TgtScalar>
struct random_without_cast_overflow<
    SrcScalar, TgtScalar,
    typename internal::enable_if<
        NumTraits<SrcScalar>::IsInteger && NumTraits<TgtScalar>::IsInteger && !NumTraits<SrcScalar>::IsSigned &&
        (std::numeric_limits<SrcScalar>::digits > std::numeric_limits<TgtScalar>::digits)>::type> {
  static SrcScalar value() {
    TgtScalar b = internal::random<TgtScalar>();
    return static_cast<SrcScalar>(b < TgtScalar(0) ? -(b + 1) : b);
  }
};

// Integer to signed narrowing cast.
template <typename SrcScalar, typename TgtScalar>
struct random_without_cast_overflow<
    SrcScalar, TgtScalar,
    typename internal::enable_if<
        NumTraits<SrcScalar>::IsInteger && NumTraits<TgtScalar>::IsInteger && NumTraits<SrcScalar>::IsSigned &&
        (std::numeric_limits<SrcScalar>::digits > std::numeric_limits<TgtScalar>::digits)>::type> {
  static SrcScalar value() { return static_cast<SrcScalar>(internal::random<TgtScalar>()); }
};

// Unsigned to signed integer narrowing cast.
template <typename SrcScalar, typename TgtScalar>
struct random_without_cast_overflow<
    SrcScalar, TgtScalar,
    typename internal::enable_if<NumTraits<SrcScalar>::IsInteger && NumTraits<TgtScalar>::IsInteger &&
                                 !NumTraits<SrcScalar>::IsSigned && NumTraits<TgtScalar>::IsSigned &&
                                 (std::numeric_limits<SrcScalar>::digits ==
                                  std::numeric_limits<TgtScalar>::digits)>::type> {
  static SrcScalar value() { return internal::random<SrcScalar>() / 2; }
};

// Floating-point to integer, full precision.
template <typename SrcScalar, typename TgtScalar>
struct random_without_cast_overflow<
    SrcScalar, TgtScalar,
    typename internal::enable_if<
        !NumTraits<SrcScalar>::IsInteger && !NumTraits<SrcScalar>::IsComplex && NumTraits<TgtScalar>::IsInteger &&
        (std::numeric_limits<TgtScalar>::digits <= std::numeric_limits<SrcScalar>::digits)>::type> {
  static SrcScalar value() { return static_cast<SrcScalar>(internal::random<TgtScalar>()); }
};

// Floating-point to integer, narrowing precision.
template <typename SrcScalar, typename TgtScalar>
struct random_without_cast_overflow<
    SrcScalar, TgtScalar,
    typename internal::enable_if<
        !NumTraits<SrcScalar>::IsInteger && !NumTraits<SrcScalar>::IsComplex && NumTraits<TgtScalar>::IsInteger &&
        (std::numeric_limits<TgtScalar>::digits > std::numeric_limits<SrcScalar>::digits)>::type> {
  static SrcScalar value() {
    // NOTE: internal::random<T>() is limited by RAND_MAX, so random<int64_t> is always within that range.
    // This prevents us from simply shifting bits, which would result in only 0 or -1.
    // Instead, keep least-significant K bits and sign.
    static const TgtScalar KeepMask = (static_cast<TgtScalar>(1) << std::numeric_limits<SrcScalar>::digits) - 1;
    const TgtScalar a = internal::random<TgtScalar>();
    return static_cast<SrcScalar>(a > TgtScalar(0) ? (a & KeepMask) : -(a & KeepMask));
  }
};

// Integer to floating-point, re-use above logic.
template <typename SrcScalar, typename TgtScalar>
struct random_without_cast_overflow<
    SrcScalar, TgtScalar,
    typename internal::enable_if<NumTraits<SrcScalar>::IsInteger && !NumTraits<TgtScalar>::IsInteger &&
                                 !NumTraits<TgtScalar>::IsComplex>::type> {
  static SrcScalar value() {
    return static_cast<SrcScalar>(random_without_cast_overflow<TgtScalar, SrcScalar>::value());
  }
};

// Floating-point narrowing conversion.
template <typename SrcScalar, typename TgtScalar>
struct random_without_cast_overflow<
    SrcScalar, TgtScalar,
    typename internal::enable_if<!NumTraits<SrcScalar>::IsInteger && !NumTraits<SrcScalar>::IsComplex &&
                                 !NumTraits<TgtScalar>::IsInteger && !NumTraits<TgtScalar>::IsComplex &&
                                 (std::numeric_limits<SrcScalar>::digits >
                                  std::numeric_limits<TgtScalar>::digits)>::type> {
  static SrcScalar value() { return static_cast<SrcScalar>(internal::random<TgtScalar>()); }
};

// Complex to non-complex.
template <typename SrcScalar, typename TgtScalar>
struct random_without_cast_overflow<
    SrcScalar, TgtScalar,
    typename internal::enable_if<NumTraits<SrcScalar>::IsComplex && !NumTraits<TgtScalar>::IsComplex>::type> {
  typedef typename NumTraits<SrcScalar>::Real SrcReal;
  static SrcScalar value() { return SrcScalar(random_without_cast_overflow<SrcReal, TgtScalar>::value(), 0); }
};

// Non-complex to complex.
template <typename SrcScalar, typename TgtScalar>
struct random_without_cast_overflow<
    SrcScalar, TgtScalar,
    typename internal::enable_if<!NumTraits<SrcScalar>::IsComplex && NumTraits<TgtScalar>::IsComplex>::type> {
  typedef typename NumTraits<TgtScalar>::Real TgtReal;
  static SrcScalar value() { return random_without_cast_overflow<SrcScalar, TgtReal>::value(); }
};

// Complex to complex.
template <typename SrcScalar, typename TgtScalar>
struct random_without_cast_overflow<
    SrcScalar, TgtScalar,
    typename internal::enable_if<NumTraits<SrcScalar>::IsComplex && NumTraits<TgtScalar>::IsComplex>::type> {
  typedef typename NumTraits<SrcScalar>::Real SrcReal;
  typedef typename NumTraits<TgtScalar>::Real TgtReal;
  static SrcScalar value() {
    return SrcScalar(random_without_cast_overflow<SrcReal, TgtReal>::value(),
                     random_without_cast_overflow<SrcReal, TgtReal>::value());
  }
};

}  // namespace internal
}  // namespace Eigen
