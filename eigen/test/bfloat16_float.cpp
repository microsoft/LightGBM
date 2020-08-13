// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>
#include <memory>
#include <math.h>

#include "main.h"

#include <Eigen/src/Core/arch/Default/BFloat16.h>

// Make sure it's possible to forward declare Eigen::bfloat16
namespace Eigen {
struct bfloat16;
}

using Eigen::bfloat16;

float BinaryToFloat(uint32_t sign, uint32_t exponent, uint32_t high_mantissa,
                    uint32_t low_mantissa) {
  float dest;
  uint32_t src = (sign << 31) + (exponent << 23) + (high_mantissa << 16) + low_mantissa;
  memcpy(static_cast<void*>(&dest),
         static_cast<const void*>(&src), sizeof(dest));
  return dest;
}

void test_truncate(float input, float expected_truncation, float expected_rounding){
  bfloat16 truncated = Eigen::bfloat16_impl::truncate_to_bfloat16(input);
  bfloat16 rounded = Eigen::bfloat16_impl::float_to_bfloat16_rtne<false>(input);
  if ((numext::isnan)(input)){
    VERIFY((numext::isnan)(static_cast<float>(truncated)) || (numext::isinf)(static_cast<float>(truncated)));
    VERIFY((numext::isnan)(static_cast<float>(rounded)) || (numext::isinf)(static_cast<float>(rounded)));
    return;
  }
  VERIFY_IS_EQUAL(expected_truncation, static_cast<float>(truncated));
  VERIFY_IS_EQUAL(expected_rounding, static_cast<float>(rounded));
}

template<typename T>
 void test_roundtrip() {
  // Representable T round trip via bfloat16
  VERIFY_IS_EQUAL(static_cast<T>(static_cast<bfloat16>(-std::numeric_limits<T>::infinity())), -std::numeric_limits<T>::infinity());
  VERIFY_IS_EQUAL(static_cast<T>(static_cast<bfloat16>(std::numeric_limits<T>::infinity())), std::numeric_limits<T>::infinity());
  VERIFY_IS_EQUAL(static_cast<T>(static_cast<bfloat16>(T(-1.0))), T(-1.0));
  VERIFY_IS_EQUAL(static_cast<T>(static_cast<bfloat16>(T(-0.5))), T(-0.5));
  VERIFY_IS_EQUAL(static_cast<T>(static_cast<bfloat16>(T(-0.0))), T(-0.0));
  VERIFY_IS_EQUAL(static_cast<T>(static_cast<bfloat16>(T(1.0))), T(1.0));
  VERIFY_IS_EQUAL(static_cast<T>(static_cast<bfloat16>(T(0.5))), T(0.5));
  VERIFY_IS_EQUAL(static_cast<T>(static_cast<bfloat16>(T(0.0))), T(0.0));
}

void test_conversion()
{
  using Eigen::bfloat16_impl::__bfloat16_raw;

  // Conversion from float.
  VERIFY_IS_EQUAL(bfloat16(1.0f).value, 0x3f80);
  VERIFY_IS_EQUAL(bfloat16(0.5f).value, 0x3f00);
  VERIFY_IS_EQUAL(bfloat16(0.33333f).value, 0x3eab);
  VERIFY_IS_EQUAL(bfloat16(3.38e38f).value, 0x7f7e);
  VERIFY_IS_EQUAL(bfloat16(3.40e38f).value, 0x7f80);  // Becomes infinity.

  // Verify round-to-nearest-even behavior.
  float val1 = static_cast<float>(bfloat16(__bfloat16_raw(0x3c00)));
  float val2 = static_cast<float>(bfloat16(__bfloat16_raw(0x3c01)));
  float val3 = static_cast<float>(bfloat16(__bfloat16_raw(0x3c02)));
  VERIFY_IS_EQUAL(bfloat16(0.5f * (val1 + val2)).value, 0x3c00);
  VERIFY_IS_EQUAL(bfloat16(0.5f * (val2 + val3)).value, 0x3c02);

  // Conversion from int.
  VERIFY_IS_EQUAL(bfloat16(-1).value, 0xbf80);
  VERIFY_IS_EQUAL(bfloat16(0).value, 0x0000);
  VERIFY_IS_EQUAL(bfloat16(1).value, 0x3f80);
  VERIFY_IS_EQUAL(bfloat16(2).value, 0x4000);
  VERIFY_IS_EQUAL(bfloat16(3).value, 0x4040);
  VERIFY_IS_EQUAL(bfloat16(12).value, 0x4140);

  // Conversion from bool.
  VERIFY_IS_EQUAL(bfloat16(false).value, 0x0000);
  VERIFY_IS_EQUAL(bfloat16(true).value, 0x3f80);

  // Conversion to bool
  VERIFY_IS_EQUAL(static_cast<bool>(bfloat16(3)), true);
  VERIFY_IS_EQUAL(static_cast<bool>(bfloat16(0.33333f)), true);
  VERIFY_IS_EQUAL(bfloat16(-0.0), false);
  VERIFY_IS_EQUAL(static_cast<bool>(bfloat16(0.0)), false);

  // Explicit conversion to float.
  VERIFY_IS_EQUAL(static_cast<float>(bfloat16(__bfloat16_raw(0x0000))), 0.0f);
  VERIFY_IS_EQUAL(static_cast<float>(bfloat16(__bfloat16_raw(0x3f80))), 1.0f);

  // Implicit conversion to float
  VERIFY_IS_EQUAL(bfloat16(__bfloat16_raw(0x0000)), 0.0f);
  VERIFY_IS_EQUAL(bfloat16(__bfloat16_raw(0x3f80)), 1.0f);

  // Zero representations
  VERIFY_IS_EQUAL(bfloat16(0.0f), bfloat16(0.0f));
  VERIFY_IS_EQUAL(bfloat16(-0.0f), bfloat16(0.0f));
  VERIFY_IS_EQUAL(bfloat16(-0.0f), bfloat16(-0.0f));
  VERIFY_IS_EQUAL(bfloat16(0.0f).value, 0x0000);
  VERIFY_IS_EQUAL(bfloat16(-0.0f).value, 0x8000);

  // Flush denormals to zero
  for (float denorm = -std::numeric_limits<float>::denorm_min();
       denorm < std::numeric_limits<float>::denorm_min();
       denorm = nextafterf(denorm, 1.0f)) {
    bfloat16 bf_trunc = Eigen::bfloat16_impl::truncate_to_bfloat16(denorm);
    VERIFY_IS_EQUAL(static_cast<float>(bf_trunc), 0.0f);

    // Implicit conversion of denormls to bool is correct
    VERIFY_IS_EQUAL(static_cast<bool>(bfloat16(denorm)), false);
    VERIFY_IS_EQUAL(bfloat16(denorm), false);

    if (std::signbit(denorm)) {
      VERIFY_IS_EQUAL(bf_trunc.value, 0x8000);
    } else {
      VERIFY_IS_EQUAL(bf_trunc.value, 0x0000);
    }
    bfloat16 bf_round = Eigen::bfloat16_impl::float_to_bfloat16_rtne<false>(denorm);
    VERIFY_IS_EQUAL(static_cast<float>(bf_round), 0.0f);
    if (std::signbit(denorm)) {
      VERIFY_IS_EQUAL(bf_round.value, 0x8000);
    } else {
      VERIFY_IS_EQUAL(bf_round.value, 0x0000);
    }
  }

  // Default is zero
  VERIFY_IS_EQUAL(static_cast<float>(bfloat16()), 0.0f);

  // Representable floats round trip via bfloat16
  test_roundtrip<float>();
  test_roundtrip<double>();
  test_roundtrip<std::complex<float> >();
  test_roundtrip<std::complex<double> >();

  // Truncate test
  test_truncate(
      BinaryToFloat(0, 0x80, 0x48, 0xf5c3),
      BinaryToFloat(0, 0x80, 0x48, 0x0000),
      BinaryToFloat(0, 0x80, 0x49, 0x0000));
  test_truncate(
      BinaryToFloat(1, 0x80, 0x48, 0xf5c3),
      BinaryToFloat(1, 0x80, 0x48, 0x0000),
      BinaryToFloat(1, 0x80, 0x49, 0x0000));
  test_truncate(
      BinaryToFloat(0, 0x80, 0x48, 0x8000),
      BinaryToFloat(0, 0x80, 0x48, 0x0000),
      BinaryToFloat(0, 0x80, 0x48, 0x0000));
  test_truncate(
      BinaryToFloat(0, 0xff, 0x00, 0x0001),
      BinaryToFloat(0, 0xff, 0x40, 0x0000),
      BinaryToFloat(0, 0xff, 0x40, 0x0000));
  test_truncate(
      BinaryToFloat(0, 0xff, 0x7f, 0xffff),
      BinaryToFloat(0, 0xff, 0x40, 0x0000),
      BinaryToFloat(0, 0xff, 0x40, 0x0000));
  test_truncate(
      BinaryToFloat(1, 0x80, 0x48, 0xc000),
      BinaryToFloat(1, 0x80, 0x48, 0x0000),
      BinaryToFloat(1, 0x80, 0x49, 0x0000));
  test_truncate(
      BinaryToFloat(0, 0x80, 0x48, 0x0000),
      BinaryToFloat(0, 0x80, 0x48, 0x0000),
      BinaryToFloat(0, 0x80, 0x48, 0x0000));
  test_truncate(
      BinaryToFloat(0, 0x80, 0x48, 0x4000),
      BinaryToFloat(0, 0x80, 0x48, 0x0000),
      BinaryToFloat(0, 0x80, 0x48, 0x0000));
  test_truncate(
      BinaryToFloat(0, 0x80, 0x48, 0x8000),
      BinaryToFloat(0, 0x80, 0x48, 0x0000),
      BinaryToFloat(0, 0x80, 0x48, 0x0000));
  test_truncate(
      BinaryToFloat(0, 0x00, 0x48, 0x8000),
      BinaryToFloat(0, 0x00, 0x00, 0x0000),
      BinaryToFloat(0, 0x00, 0x00, 0x0000));
  test_truncate(
      BinaryToFloat(0, 0x00, 0x7f, 0xc000),
      BinaryToFloat(0, 0x00, 0x00, 0x0000),
      BinaryToFloat(0, 0x00, 0x00, 0x0000));

  // Conversion
  Array<float,1,100> a;
  for (int i = 0; i < 100; i++) a(i) = i + 1.25;
  Array<bfloat16,1,100> b = a.cast<bfloat16>();
  Array<float,1,100> c = b.cast<float>();
  for (int i = 0; i < 100; ++i) {
    VERIFY_LE(numext::abs(c(i) - a(i)), a(i) / 128);
  }

  // Epsilon
  VERIFY_LE(1.0f, static_cast<float>((std::numeric_limits<bfloat16>::epsilon)() + bfloat16(1.0f)));
  VERIFY_IS_EQUAL(1.0f, static_cast<float>((std::numeric_limits<bfloat16>::epsilon)() / bfloat16(2.0f) + bfloat16(1.0f)));

  // Negate
  VERIFY_IS_EQUAL(static_cast<float>(-bfloat16(3.0f)), -3.0f);
  VERIFY_IS_EQUAL(static_cast<float>(-bfloat16(-4.5f)), 4.5f);


#if !EIGEN_COMP_MSVC
  // Visual Studio errors out on divisions by 0
  VERIFY((numext::isnan)(static_cast<float>(bfloat16(0.0 / 0.0))));
  VERIFY((numext::isinf)(static_cast<float>(bfloat16(1.0 / 0.0))));
  VERIFY((numext::isinf)(static_cast<float>(bfloat16(-1.0 / 0.0))));

  // Visual Studio errors out on divisions by 0
  VERIFY((numext::isnan)(bfloat16(0.0 / 0.0)));
  VERIFY((numext::isinf)(bfloat16(1.0 / 0.0)));
  VERIFY((numext::isinf)(bfloat16(-1.0 / 0.0)));
#endif

  // NaNs and infinities.
  VERIFY(!(numext::isinf)(static_cast<float>(bfloat16(3.38e38f))));  // Largest finite number.
  VERIFY(!(numext::isnan)(static_cast<float>(bfloat16(0.0f))));
  VERIFY((numext::isinf)(static_cast<float>(bfloat16(__bfloat16_raw(0xff80)))));
  VERIFY((numext::isnan)(static_cast<float>(bfloat16(__bfloat16_raw(0xffc0)))));
  VERIFY((numext::isinf)(static_cast<float>(bfloat16(__bfloat16_raw(0x7f80)))));
  VERIFY((numext::isnan)(static_cast<float>(bfloat16(__bfloat16_raw(0x7fc0)))));

  // Exactly same checks as above, just directly on the bfloat16 representation.
  VERIFY(!(numext::isinf)(bfloat16(__bfloat16_raw(0x7bff))));
  VERIFY(!(numext::isnan)(bfloat16(__bfloat16_raw(0x0000))));
  VERIFY((numext::isinf)(bfloat16(__bfloat16_raw(0xff80))));
  VERIFY((numext::isnan)(bfloat16(__bfloat16_raw(0xffc0))));
  VERIFY((numext::isinf)(bfloat16(__bfloat16_raw(0x7f80))));
  VERIFY((numext::isnan)(bfloat16(__bfloat16_raw(0x7fc0))));
}

void test_numtraits()
{
  std::cout << "epsilon       = " << NumTraits<bfloat16>::epsilon() << "  (0x" << std::hex << NumTraits<bfloat16>::epsilon().value << ")" << std::endl;
  std::cout << "highest       = " << NumTraits<bfloat16>::highest() << "  (0x" << std::hex << NumTraits<bfloat16>::highest().value << ")" << std::endl;
  std::cout << "lowest        = " << NumTraits<bfloat16>::lowest() << "  (0x" << std::hex << NumTraits<bfloat16>::lowest().value << ")" << std::endl;
  std::cout << "min           = " << (std::numeric_limits<bfloat16>::min)() << "  (0x" << std::hex << (std::numeric_limits<bfloat16>::min)().value << ")" << std::endl;
  std::cout << "denorm min    = " << (std::numeric_limits<bfloat16>::denorm_min)() << "  (0x" << std::hex << (std::numeric_limits<bfloat16>::denorm_min)().value << ")" << std::endl;
  std::cout << "infinity      = " << NumTraits<bfloat16>::infinity() << "  (0x" << std::hex << NumTraits<bfloat16>::infinity().value << ")" << std::endl;
  std::cout << "quiet nan     = " << NumTraits<bfloat16>::quiet_NaN() << "  (0x" << std::hex << NumTraits<bfloat16>::quiet_NaN().value << ")" << std::endl;
  std::cout << "signaling nan = " << std::numeric_limits<bfloat16>::signaling_NaN() << "  (0x" << std::hex << std::numeric_limits<bfloat16>::signaling_NaN().value << ")" << std::endl;

  VERIFY(NumTraits<bfloat16>::IsSigned);

  VERIFY_IS_EQUAL( std::numeric_limits<bfloat16>::infinity().value, bfloat16(std::numeric_limits<float>::infinity()).value );
  VERIFY_IS_EQUAL( std::numeric_limits<bfloat16>::quiet_NaN().value, bfloat16(std::numeric_limits<float>::quiet_NaN()).value );
  VERIFY( (std::numeric_limits<bfloat16>::min)() > bfloat16(0.f) );
  VERIFY( (std::numeric_limits<bfloat16>::denorm_min)() > bfloat16(0.f) );
  VERIFY_IS_EQUAL( (std::numeric_limits<bfloat16>::denorm_min)()/bfloat16(2), bfloat16(0.f) );
}

void test_arithmetic()
{
  VERIFY_IS_EQUAL(static_cast<float>(bfloat16(2) + bfloat16(2)), 4);
  VERIFY_IS_EQUAL(static_cast<float>(bfloat16(2) + bfloat16(-2)), 0);
  VERIFY_IS_APPROX(static_cast<float>(bfloat16(0.33333f) + bfloat16(0.66667f)), 1.0f);
  VERIFY_IS_EQUAL(static_cast<float>(bfloat16(2.0f) * bfloat16(-5.5f)), -11.0f);
  VERIFY_IS_APPROX(static_cast<float>(bfloat16(1.0f) / bfloat16(3.0f)), 0.3339f);
  VERIFY_IS_EQUAL(static_cast<float>(-bfloat16(4096.0f)), -4096.0f);
  VERIFY_IS_EQUAL(static_cast<float>(-bfloat16(-4096.0f)), 4096.0f);
}

void test_comparison()
{
  VERIFY(bfloat16(1.0f) > bfloat16(0.5f));
  VERIFY(bfloat16(0.5f) < bfloat16(1.0f));
  VERIFY(!(bfloat16(1.0f) < bfloat16(0.5f)));
  VERIFY(!(bfloat16(0.5f) > bfloat16(1.0f)));

  VERIFY(!(bfloat16(4.0f) > bfloat16(4.0f)));
  VERIFY(!(bfloat16(4.0f) < bfloat16(4.0f)));

  VERIFY(!(bfloat16(0.0f) < bfloat16(-0.0f)));
  VERIFY(!(bfloat16(-0.0f) < bfloat16(0.0f)));
  VERIFY(!(bfloat16(0.0f) > bfloat16(-0.0f)));
  VERIFY(!(bfloat16(-0.0f) > bfloat16(0.0f)));

  VERIFY(bfloat16(0.2f) > bfloat16(-1.0f));
  VERIFY(bfloat16(-1.0f) < bfloat16(0.2f));
  VERIFY(bfloat16(-16.0f) < bfloat16(-15.0f));

  VERIFY(bfloat16(1.0f) == bfloat16(1.0f));
  VERIFY(bfloat16(1.0f) != bfloat16(2.0f));

  // Comparisons with NaNs and infinities.
#if !EIGEN_COMP_MSVC
  // Visual Studio errors out on divisions by 0
  VERIFY(!(bfloat16(0.0 / 0.0) == bfloat16(0.0 / 0.0)));
  VERIFY(bfloat16(0.0 / 0.0) != bfloat16(0.0 / 0.0));

  VERIFY(!(bfloat16(1.0) == bfloat16(0.0 / 0.0)));
  VERIFY(!(bfloat16(1.0) < bfloat16(0.0 / 0.0)));
  VERIFY(!(bfloat16(1.0) > bfloat16(0.0 / 0.0)));
  VERIFY(bfloat16(1.0) != bfloat16(0.0 / 0.0));

  VERIFY(bfloat16(1.0) < bfloat16(1.0 / 0.0));
  VERIFY(bfloat16(1.0) > bfloat16(-1.0 / 0.0));
#endif
}

void test_basic_functions()
{
  VERIFY_IS_EQUAL(static_cast<float>(numext::abs(bfloat16(3.5f))), 3.5f);
  VERIFY_IS_EQUAL(static_cast<float>(abs(bfloat16(3.5f))), 3.5f);
  VERIFY_IS_EQUAL(static_cast<float>(numext::abs(bfloat16(-3.5f))), 3.5f);
  VERIFY_IS_EQUAL(static_cast<float>(abs(bfloat16(-3.5f))), 3.5f);

  VERIFY_IS_EQUAL(static_cast<float>(numext::floor(bfloat16(3.5f))), 3.0f);
  VERIFY_IS_EQUAL(static_cast<float>(floor(bfloat16(3.5f))), 3.0f);
  VERIFY_IS_EQUAL(static_cast<float>(numext::floor(bfloat16(-3.5f))), -4.0f);
  VERIFY_IS_EQUAL(static_cast<float>(floor(bfloat16(-3.5f))), -4.0f);

  VERIFY_IS_EQUAL(static_cast<float>(numext::ceil(bfloat16(3.5f))), 4.0f);
  VERIFY_IS_EQUAL(static_cast<float>(ceil(bfloat16(3.5f))), 4.0f);
  VERIFY_IS_EQUAL(static_cast<float>(numext::ceil(bfloat16(-3.5f))), -3.0f);
  VERIFY_IS_EQUAL(static_cast<float>(ceil(bfloat16(-3.5f))), -3.0f);

  VERIFY_IS_APPROX(static_cast<float>(numext::sqrt(bfloat16(0.0f))), 0.0f);
  VERIFY_IS_APPROX(static_cast<float>(sqrt(bfloat16(0.0f))), 0.0f);
  VERIFY_IS_APPROX(static_cast<float>(numext::sqrt(bfloat16(4.0f))), 2.0f);
  VERIFY_IS_APPROX(static_cast<float>(sqrt(bfloat16(4.0f))), 2.0f);

  VERIFY_IS_APPROX(static_cast<float>(numext::pow(bfloat16(0.0f), bfloat16(1.0f))), 0.0f);
  VERIFY_IS_APPROX(static_cast<float>(pow(bfloat16(0.0f), bfloat16(1.0f))), 0.0f);
  VERIFY_IS_APPROX(static_cast<float>(numext::pow(bfloat16(2.0f), bfloat16(2.0f))), 4.0f);
  VERIFY_IS_APPROX(static_cast<float>(pow(bfloat16(2.0f), bfloat16(2.0f))), 4.0f);

  VERIFY_IS_EQUAL(static_cast<float>(numext::exp(bfloat16(0.0f))), 1.0f);
  VERIFY_IS_EQUAL(static_cast<float>(exp(bfloat16(0.0f))), 1.0f);
  VERIFY_IS_APPROX(static_cast<float>(numext::exp(bfloat16(EIGEN_PI))), 20.f + static_cast<float>(EIGEN_PI));
  VERIFY_IS_APPROX(static_cast<float>(exp(bfloat16(EIGEN_PI))), 20.f + static_cast<float>(EIGEN_PI));

  VERIFY_IS_EQUAL(static_cast<float>(numext::expm1(bfloat16(0.0f))), 0.0f);
  VERIFY_IS_EQUAL(static_cast<float>(expm1(bfloat16(0.0f))), 0.0f);
  VERIFY_IS_APPROX(static_cast<float>(numext::expm1(bfloat16(2.0f))), 6.375f);
  VERIFY_IS_APPROX(static_cast<float>(expm1(bfloat16(2.0f))), 6.375f);

  VERIFY_IS_EQUAL(static_cast<float>(numext::log(bfloat16(1.0f))), 0.0f);
  VERIFY_IS_EQUAL(static_cast<float>(log(bfloat16(1.0f))), 0.0f);
  VERIFY_IS_APPROX(static_cast<float>(numext::log(bfloat16(10.0f))), 2.296875f);
  VERIFY_IS_APPROX(static_cast<float>(log(bfloat16(10.0f))), 2.296875f);

  VERIFY_IS_EQUAL(static_cast<float>(numext::log1p(bfloat16(0.0f))), 0.0f);
  VERIFY_IS_EQUAL(static_cast<float>(log1p(bfloat16(0.0f))), 0.0f);
  VERIFY_IS_APPROX(static_cast<float>(numext::log1p(bfloat16(10.0f))), 2.390625f);
  VERIFY_IS_APPROX(static_cast<float>(log1p(bfloat16(10.0f))), 2.390625f);
}

void test_trigonometric_functions()
{
  VERIFY_IS_APPROX(numext::cos(bfloat16(0.0f)), bfloat16(cosf(0.0f)));
  VERIFY_IS_APPROX(cos(bfloat16(0.0f)), bfloat16(cosf(0.0f)));
  VERIFY_IS_APPROX(numext::cos(bfloat16(EIGEN_PI)), bfloat16(cosf(EIGEN_PI)));
  // VERIFY_IS_APPROX(numext::cos(bfloat16(EIGEN_PI/2)), bfloat16(cosf(EIGEN_PI/2)));
  // VERIFY_IS_APPROX(numext::cos(bfloat16(3*EIGEN_PI/2)), bfloat16(cosf(3*EIGEN_PI/2)));
  VERIFY_IS_APPROX(numext::cos(bfloat16(3.5f)), bfloat16(cosf(3.5f)));

  VERIFY_IS_APPROX(numext::sin(bfloat16(0.0f)), bfloat16(sinf(0.0f)));
  VERIFY_IS_APPROX(sin(bfloat16(0.0f)), bfloat16(sinf(0.0f)));
  // VERIFY_IS_APPROX(numext::sin(bfloat16(EIGEN_PI)), bfloat16(sinf(EIGEN_PI)));
  VERIFY_IS_APPROX(numext::sin(bfloat16(EIGEN_PI/2)), bfloat16(sinf(EIGEN_PI/2)));
  VERIFY_IS_APPROX(numext::sin(bfloat16(3*EIGEN_PI/2)), bfloat16(sinf(3*EIGEN_PI/2)));
  VERIFY_IS_APPROX(numext::sin(bfloat16(3.5f)), bfloat16(sinf(3.5f)));

  VERIFY_IS_APPROX(numext::tan(bfloat16(0.0f)), bfloat16(tanf(0.0f)));
  VERIFY_IS_APPROX(tan(bfloat16(0.0f)), bfloat16(tanf(0.0f)));
  // VERIFY_IS_APPROX(numext::tan(bfloat16(EIGEN_PI)), bfloat16(tanf(EIGEN_PI)));
  // VERIFY_IS_APPROX(numext::tan(bfloat16(EIGEN_PI/2)), bfloat16(tanf(EIGEN_PI/2)));
  // VERIFY_IS_APPROX(numext::tan(bfloat16(3*EIGEN_PI/2)), bfloat16(tanf(3*EIGEN_PI/2)));
  VERIFY_IS_APPROX(numext::tan(bfloat16(3.5f)), bfloat16(tanf(3.5f)));
}

void test_array()
{
  typedef Array<bfloat16,1,Dynamic> ArrayXh;
  Index size = internal::random<Index>(1,10);
  Index i = internal::random<Index>(0,size-1);
  ArrayXh a1 = ArrayXh::Random(size), a2 = ArrayXh::Random(size);
  VERIFY_IS_APPROX( a1+a1, bfloat16(2)*a1 );
  VERIFY( (a1.abs() >= bfloat16(0)).all() );
  VERIFY_IS_APPROX( (a1*a1).sqrt(), a1.abs() );

  VERIFY( ((a1.min)(a2) <= (a1.max)(a2)).all() );
  a1(i) = bfloat16(-10.);
  VERIFY_IS_EQUAL( a1.minCoeff(), bfloat16(-10.) );
  a1(i) = bfloat16(10.);
  VERIFY_IS_EQUAL( a1.maxCoeff(), bfloat16(10.) );

  std::stringstream ss;
  ss << a1;
}

void test_product()
{
  typedef Matrix<bfloat16,Dynamic,Dynamic> MatrixXh;
  Index rows  = internal::random<Index>(1,EIGEN_TEST_MAX_SIZE);
  Index cols  = internal::random<Index>(1,EIGEN_TEST_MAX_SIZE);
  Index depth = internal::random<Index>(1,EIGEN_TEST_MAX_SIZE);
  MatrixXh Ah = MatrixXh::Random(rows,depth);
  MatrixXh Bh = MatrixXh::Random(depth,cols);
  MatrixXh Ch = MatrixXh::Random(rows,cols);
  MatrixXf Af = Ah.cast<float>();
  MatrixXf Bf = Bh.cast<float>();
  MatrixXf Cf = Ch.cast<float>();
  VERIFY_IS_APPROX(Ch.noalias()+=Ah*Bh, (Cf.noalias()+=Af*Bf).cast<bfloat16>());
}

EIGEN_DECLARE_TEST(bfloat16_float)
{
  CALL_SUBTEST(test_numtraits());
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST(test_conversion());
    CALL_SUBTEST(test_arithmetic());
    CALL_SUBTEST(test_comparison());
    CALL_SUBTEST(test_basic_functions());
    CALL_SUBTEST(test_trigonometric_functions());
    CALL_SUBTEST(test_array());
    CALL_SUBTEST(test_product());
  }
}
