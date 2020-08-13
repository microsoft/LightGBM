// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "packetmath_test_shared.h"
#include "../Eigen/SpecialFunctions"

template<typename Scalar,typename Packet> void packetmath_real()
{
  using std::abs;
  typedef internal::packet_traits<Scalar> PacketTraits;
  const int PacketSize = internal::unpacket_traits<Packet>::size;

  const int size = PacketSize*4;
  EIGEN_ALIGN_MAX Scalar data1[PacketSize*4];
  EIGEN_ALIGN_MAX Scalar data2[PacketSize*4];
  EIGEN_ALIGN_MAX Scalar ref[PacketSize*4];

#if EIGEN_HAS_C99_MATH
  {
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    test::packet_helper<internal::packet_traits<Scalar>::HasLGamma,Packet> h;
    h.store(data2, internal::plgamma(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
  }
  if (internal::packet_traits<Scalar>::HasErf) {
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    test::packet_helper<internal::packet_traits<Scalar>::HasErf,Packet> h;
    h.store(data2, internal::perf(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
  }
  {
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    test::packet_helper<internal::packet_traits<Scalar>::HasErfc,Packet> h;
    h.store(data2, internal::perfc(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
  }
  {
    for (int i=0; i<size; ++i) {
      data1[i] = internal::random<Scalar>(0,1);
    }
    CHECK_CWISE1_IF(internal::packet_traits<Scalar>::HasNdtri, numext::ndtri, internal::pndtri);
  }
#endif  // EIGEN_HAS_C99_MATH

  // For bessel_i*e and bessel_j*, the valid range is negative reals.
  for (int i=0; i<size; ++i)
  {
    data1[i] = internal::random<Scalar>(-1,1) * std::pow(Scalar(10), internal::random<Scalar>(-6,6));
    data2[i] = internal::random<Scalar>(-1,1) * std::pow(Scalar(10), internal::random<Scalar>(-6,6));
  }

  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_i0e, internal::pbessel_i0e);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_i1e, internal::pbessel_i1e);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_j0, internal::pbessel_j0);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_j1, internal::pbessel_j1);

  // Use a smaller data range for the bessel_i* as these can become very large.
  // Following #1693, we also restrict this range further to avoid inf's due to
  // differences in pexp and exp.
  for (int i=0; i<size; ++i) {
      data1[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(9), internal::random<Scalar>(-1,2));
      data2[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(9), internal::random<Scalar>(-1,2));
  }
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_i0, internal::pbessel_i0);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_i1, internal::pbessel_i1);


  // y_i, and k_i are valid for x > 0.
  for (int i=0; i<size; ++i)
  {
    data1[i] = internal::random<Scalar>(0.01,1) * std::pow(Scalar(10), internal::random<Scalar>(-2,5));
    data2[i] = internal::random<Scalar>(0.01,1) * std::pow(Scalar(10), internal::random<Scalar>(-2,5));
  }

  // TODO(srvasude): Re-enable this test once properly investigated why the
  // scalar and vector paths differ.
  // CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_y0, internal::pbessel_y0);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_y1, internal::pbessel_y1);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_k0e, internal::pbessel_k0e);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_k1e, internal::pbessel_k1e);

  // Following #1693, we restrict the range for exp to avoid zeroing out too
  // fast.
  for (int i=0; i<size; ++i) {
      data1[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(9), internal::random<Scalar>(-1,2));
      data2[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(9), internal::random<Scalar>(-1,2));
  }
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_k0, internal::pbessel_k0);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_k1, internal::pbessel_k1);


  for (int i=0; i<size; ++i) {
      data1[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(10), internal::random<Scalar>(-1,2));
      data2[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(10), internal::random<Scalar>(-1,2));
  }

#if EIGEN_HAS_C99_MATH && (__cplusplus > 199711L)
  CHECK_CWISE1_IF(internal::packet_traits<Scalar>::HasLGamma, std::lgamma, internal::plgamma);
  CHECK_CWISE1_IF(internal::packet_traits<Scalar>::HasErf, std::erf, internal::perf);
  CHECK_CWISE1_IF(internal::packet_traits<Scalar>::HasErfc, std::erfc, internal::perfc);
#endif

}

namespace Eigen {
namespace test {

template<typename Scalar,typename PacketType, bool IsComplex, bool IsInteger>
struct runall {
  static void run() {
    packetmath_real<Scalar,PacketType>();
  }
};

}
}

EIGEN_DECLARE_TEST(special_packetmath)
{
  g_first_pass = true;
  for(int i = 0; i < g_repeat; i++) {

    CALL_SUBTEST_1( test::runner<float>::run() );
    CALL_SUBTEST_2( test::runner<double>::run() );
    g_first_pass = false;
  }
}
