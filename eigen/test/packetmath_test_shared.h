// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <typeinfo>

#if defined __GNUC__ && __GNUC__>=6
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
// using namespace Eigen;

bool g_first_pass = true;

namespace Eigen {
namespace internal {

template<typename T> T negate(const T& x) { return -x; }

template<typename T>
Map<const Array<unsigned char,sizeof(T),1> >
bits(const T& x) {
  return Map<const Array<unsigned char,sizeof(T),1> >(reinterpret_cast<const unsigned char *>(&x));
}

// The following implement bitwise operations on floating point types
template<typename T,typename Bits,typename Func>
T apply_bit_op(Bits a, Bits b, Func f) {
  Array<unsigned char,sizeof(T),1> data;
  T res;
  for(Index i = 0; i < data.size(); ++i)
    data[i] = f(a[i], b[i]);
  // Note: The reinterpret_cast works around GCC's class-memaccess warnings:
  std::memcpy(reinterpret_cast<unsigned char*>(&res), data.data(), sizeof(T));
  return res;
}

#define EIGEN_TEST_MAKE_BITWISE2(OP,FUNC,T)             \
  template<> T EIGEN_CAT(p,OP)(const T& a,const T& b) { \
    return apply_bit_op<T>(bits(a),bits(b),FUNC);     \
  }

#define EIGEN_TEST_MAKE_BITWISE(OP,FUNC)                  \
  EIGEN_TEST_MAKE_BITWISE2(OP,FUNC,float)                 \
  EIGEN_TEST_MAKE_BITWISE2(OP,FUNC,double)                \
  EIGEN_TEST_MAKE_BITWISE2(OP,FUNC,half)                  \
  EIGEN_TEST_MAKE_BITWISE2(OP,FUNC,bfloat16)              \
  EIGEN_TEST_MAKE_BITWISE2(OP,FUNC,std::complex<float>)   \
  EIGEN_TEST_MAKE_BITWISE2(OP,FUNC,std::complex<double>)

EIGEN_TEST_MAKE_BITWISE(xor,std::bit_xor<unsigned char>())
EIGEN_TEST_MAKE_BITWISE(and,std::bit_and<unsigned char>())
EIGEN_TEST_MAKE_BITWISE(or, std::bit_or<unsigned char>())
struct bit_andnot{
  template<typename T> T
  operator()(T a, T b) const { return a & (~b); }
};
EIGEN_TEST_MAKE_BITWISE(andnot, bit_andnot())
template<typename T>
bool biteq(T a, T b) {
  return (bits(a) == bits(b)).all();
}

}

namespace test {

// NOTE: we disable inlining for this function to workaround a GCC issue when using -O3 and the i387 FPU.
template<typename Scalar> EIGEN_DONT_INLINE
bool isApproxAbs(const Scalar& a, const Scalar& b, const typename NumTraits<Scalar>::Real& refvalue)
{
  return internal::isMuchSmallerThan(a-b, refvalue);
}

template<typename Scalar> bool areApproxAbs(const Scalar* a, const Scalar* b, int size, const typename NumTraits<Scalar>::Real& refvalue)
{
  for (int i=0; i<size; ++i)
  {
    if (!isApproxAbs(a[i],b[i],refvalue))
    {
      std::cout << "ref: [" << Map<const Matrix<Scalar,1,Dynamic> >(a,size) << "]" << " != vec: [" << Map<const Matrix<Scalar,1,Dynamic> >(b,size) << "]\n";
      return false;
    }
  }
  return true;
}

template<typename Scalar> bool areApprox(const Scalar* a, const Scalar* b, int size)
{
  for (int i=0; i<size; ++i)
  {
    if (a[i]!=b[i] && !internal::isApprox(a[i],b[i]))
    {
      if((numext::isnan)(a[i]) && (numext::isnan)(b[i]))
      {
        continue;
      }
      std::cout << "ref: [" << Map<const Matrix<Scalar,1,Dynamic> >(a,size) << "]" << " != vec: [" << Map<const Matrix<Scalar,1,Dynamic> >(b,size) << "]\n";
      return false;
    }
  }
  return true;
}

#define CHECK_CWISE1(REFOP, POP) { \
  for (int i=0; i<PacketSize; ++i) \
    ref[i] = REFOP(data1[i]); \
  internal::pstore(data2, POP(internal::pload<Packet>(data1))); \
  VERIFY(test::areApprox(ref, data2, PacketSize) && #POP); \
}

template<bool Cond,typename Packet>
struct packet_helper
{
  template<typename T>
  inline Packet load(const T* from) const { return internal::pload<Packet>(from); }

  template<typename T>
  inline Packet loadu(const T* from) const { return internal::ploadu<Packet>(from); }

  template<typename T>
  inline Packet load(const T* from, unsigned long long umask) const { return internal::ploadu<Packet>(from, umask); }

  template<typename T>
  inline void store(T* to, const Packet& x) const { internal::pstore(to,x); }

  template<typename T>
  inline void store(T* to, const Packet& x, unsigned long long umask) const { internal::pstoreu(to, x, umask); }
};

template<typename Packet>
struct packet_helper<false,Packet>
{
  template<typename T>
  inline T load(const T* from) const { return *from; }

  template<typename T>
  inline T loadu(const T* from) const { return *from; }

  template<typename T>
  inline T load(const T* from, unsigned long long) const { return *from; }

  template<typename T>
  inline void store(T* to, const T& x) const { *to = x; }

  template<typename T>
  inline void store(T* to, const T& x, unsigned long long) const { *to = x; }
};

#define CHECK_CWISE1_IF(COND, REFOP, POP) if(COND) { \
  test::packet_helper<COND,Packet> h; \
  for (int i=0; i<PacketSize; ++i) \
    ref[i] = REFOP(data1[i]); \
  h.store(data2, POP(h.load(data1))); \
  VERIFY(test::areApprox(ref, data2, PacketSize) && #POP); \
}

#define CHECK_CWISE2_IF(COND, REFOP, POP) if(COND) { \
  test::packet_helper<COND,Packet> h; \
  for (int i=0; i<PacketSize; ++i) \
    ref[i] = REFOP(data1[i], data1[i+PacketSize]); \
  h.store(data2, POP(h.load(data1),h.load(data1+PacketSize))); \
  VERIFY(test::areApprox(ref, data2, PacketSize) && #POP); \
}

#define CHECK_CWISE3_IF(COND, REFOP, POP) if (COND) {                      \
  test::packet_helper<COND, Packet> h;                                     \
  for (int i = 0; i < PacketSize; ++i)                                     \
    ref[i] =                                                               \
        REFOP(data1[i], data1[i + PacketSize], data1[i + 2 * PacketSize]); \
  h.store(data2, POP(h.load(data1), h.load(data1 + PacketSize),            \
                     h.load(data1 + 2 * PacketSize)));                     \
  VERIFY(test::areApprox(ref, data2, PacketSize) && #POP);                 \
}

// Specialize the runall struct in your test file by defining run().
template<
  typename Scalar,
  typename PacketType,
  bool IsComplex = NumTraits<Scalar>::IsComplex,
  bool IsInteger = NumTraits<Scalar>::IsInteger>
struct runall;

template<
  typename Scalar,
  typename PacketType = typename internal::packet_traits<Scalar>::type,
  bool Vectorized = internal::packet_traits<Scalar>::Vectorizable,
  bool HasHalf = !internal::is_same<typename internal::unpacket_traits<PacketType>::half,PacketType>::value >
struct runner;

template<typename Scalar,typename PacketType>
struct runner<Scalar,PacketType,true,true>
{
  static void run() {
    runall<Scalar,PacketType>::run();
    runner<Scalar,typename internal::unpacket_traits<PacketType>::half>::run();
  }
};

template<typename Scalar,typename PacketType>
struct runner<Scalar,PacketType,true,false>
{
  static void run() {
    runall<Scalar,PacketType>::run();
    runall<Scalar,Scalar>::run();
  }
};

template<typename Scalar,typename PacketType>
struct runner<Scalar,PacketType,false,false>
{
  static void run() {
    runall<Scalar,PacketType>::run();
  }
};

}
}
