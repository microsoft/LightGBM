// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Tal Hadad <tal_hd@hotmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <unsupported/Eigen/EulerAngles>

using namespace Eigen;

// Unfortunately, we need to specialize it in order to work. (We could add it in main.h test framework)
template <typename Scalar, class System>
bool verifyIsApprox(const Eigen::EulerAngles<Scalar, System>& a, const Eigen::EulerAngles<Scalar, System>& b)
{
  return verifyIsApprox(a.angles(), b.angles());
}

// Verify that x is in the approxed range [a, b]
#define VERIFY_APPROXED_RANGE(a, x, b) \
  do { \
  VERIFY_IS_APPROX_OR_LESS_THAN(a, x); \
  VERIFY_IS_APPROX_OR_LESS_THAN(x, b); \
  } while(0)

const char X = EULER_X;
const char Y = EULER_Y;
const char Z = EULER_Z;

template<typename Scalar, class EulerSystem>
void verify_euler(const EulerAngles<Scalar, EulerSystem>& e)
{
  typedef EulerAngles<Scalar, EulerSystem> EulerAnglesType;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Quaternion<Scalar> QuaternionType;
  typedef AngleAxis<Scalar> AngleAxisType;
  
  const Scalar ONE = Scalar(1);
  const Scalar HALF_PI = Scalar(EIGEN_PI / 2);
  const Scalar PI = Scalar(EIGEN_PI);
  
  // It's very important calc the acceptable precision depending on the distance from the pole.
  const Scalar longitudeRadius = std::abs(
    EulerSystem::IsTaitBryan ?
    std::cos(e.beta()) :
    std::sin(e.beta())
    );
  Scalar precision = test_precision<Scalar>() / longitudeRadius;
  
  Scalar betaRangeStart, betaRangeEnd;
  if (EulerSystem::IsTaitBryan)
  {
    betaRangeStart = -HALF_PI;
    betaRangeEnd = HALF_PI;
  }
  else
  {
    if (!EulerSystem::IsBetaOpposite)
    {
      betaRangeStart = 0;
      betaRangeEnd = PI;
    }
    else
    {
      betaRangeStart = -PI;
      betaRangeEnd = 0;
    }
  }
  
  const Vector3 I_ = EulerAnglesType::AlphaAxisVector();
  const Vector3 J_ = EulerAnglesType::BetaAxisVector();
  const Vector3 K_ = EulerAnglesType::GammaAxisVector();
  
  // Is approx checks
  VERIFY(e.isApprox(e));
  VERIFY_IS_APPROX(e, e);
  VERIFY_IS_NOT_APPROX(e, EulerAnglesType(e.alpha() + ONE, e.beta() + ONE, e.gamma() + ONE));

  const Matrix3 m(e);
  VERIFY_IS_APPROX(Scalar(m.determinant()), ONE);

  EulerAnglesType ebis(m);
  
  // When no roll(acting like polar representation), we have the best precision.
  // One of those cases is when the Euler angles are on the pole, and because it's singular case,
  //  the computation returns no roll.
  if (ebis.beta() == 0)
    precision = test_precision<Scalar>();
  
  // Check that eabis in range
  VERIFY_APPROXED_RANGE(-PI, ebis.alpha(), PI);
  VERIFY_APPROXED_RANGE(betaRangeStart, ebis.beta(), betaRangeEnd);
  VERIFY_APPROXED_RANGE(-PI, ebis.gamma(), PI);

  const Matrix3 mbis(AngleAxisType(ebis.alpha(), I_) * AngleAxisType(ebis.beta(), J_) * AngleAxisType(ebis.gamma(), K_));
  VERIFY_IS_APPROX(Scalar(mbis.determinant()), ONE);
  VERIFY_IS_APPROX(mbis, ebis.toRotationMatrix());
  /*std::cout << "===================\n" <<
    "e: " << e << std::endl <<
    "eabis: " << eabis.transpose() << std::endl <<
    "m: " << m << std::endl <<
    "mbis: " << mbis << std::endl <<
    "X: " << (m * Vector3::UnitX()).transpose() << std::endl <<
    "X: " << (mbis * Vector3::UnitX()).transpose() << std::endl;*/
  VERIFY(m.isApprox(mbis, precision));

  // Test if ea and eabis are the same
  // Need to check both singular and non-singular cases
  // There are two singular cases.
  // 1. When I==K and sin(ea(1)) == 0
  // 2. When I!=K and cos(ea(1)) == 0

  // TODO: Make this test work well, and use range saturation function.
  /*// If I==K, and ea[1]==0, then there no unique solution.
  // The remark apply in the case where I!=K, and |ea[1]| is close to +-pi/2.
  if( (i!=k || ea[1]!=0) && (i==k || !internal::isApprox(abs(ea[1]),Scalar(EIGEN_PI/2),test_precision<Scalar>())) ) 
      VERIFY_IS_APPROX(ea, eabis);*/
  
  // Quaternions
  const QuaternionType q(e);
  ebis = q;
  const QuaternionType qbis(ebis);
  VERIFY(internal::isApprox<Scalar>(std::abs(q.dot(qbis)), ONE, precision));
  //VERIFY_IS_APPROX(eabis, eabis2);// Verify that the euler angles are still the same
  
  // A suggestion for simple product test when will be supported.
  /*EulerAnglesType e2(PI/2, PI/2, PI/2);
  Matrix3 m2(e2);
  VERIFY_IS_APPROX(e*e2, m*m2);*/
}

template<signed char A, signed char B, signed char C, typename Scalar>
void verify_euler_vec(const Matrix<Scalar,3,1>& ea)
{
  verify_euler(EulerAngles<Scalar, EulerSystem<A, B, C> >(ea[0], ea[1], ea[2]));
}

template<signed char A, signed char B, signed char C, typename Scalar>
void verify_euler_all_neg(const Matrix<Scalar,3,1>& ea)
{
  verify_euler_vec<+A,+B,+C>(ea);
  verify_euler_vec<+A,+B,-C>(ea);
  verify_euler_vec<+A,-B,+C>(ea);
  verify_euler_vec<+A,-B,-C>(ea);
  
  verify_euler_vec<-A,+B,+C>(ea);
  verify_euler_vec<-A,+B,-C>(ea);
  verify_euler_vec<-A,-B,+C>(ea);
  verify_euler_vec<-A,-B,-C>(ea);
}

template<typename Scalar> void check_all_var(const Matrix<Scalar,3,1>& ea)
{
  verify_euler_all_neg<X,Y,Z>(ea);
  verify_euler_all_neg<X,Y,X>(ea);
  verify_euler_all_neg<X,Z,Y>(ea);
  verify_euler_all_neg<X,Z,X>(ea);
  
  verify_euler_all_neg<Y,Z,X>(ea);
  verify_euler_all_neg<Y,Z,Y>(ea);
  verify_euler_all_neg<Y,X,Z>(ea);
  verify_euler_all_neg<Y,X,Y>(ea);
  
  verify_euler_all_neg<Z,X,Y>(ea);
  verify_euler_all_neg<Z,X,Z>(ea);
  verify_euler_all_neg<Z,Y,X>(ea);
  verify_euler_all_neg<Z,Y,Z>(ea);
}

template<typename Scalar> void check_singular_cases(const Scalar& singularBeta)
{
  typedef Matrix<Scalar,3,1> Vector3;
  const Scalar PI = Scalar(EIGEN_PI);
  
  for (Scalar epsilon = NumTraits<Scalar>::epsilon(); epsilon < 1; epsilon *= Scalar(1.2))
  {
    check_all_var(Vector3(PI/4, singularBeta, PI/3));
    check_all_var(Vector3(PI/4, singularBeta - epsilon, PI/3));
    check_all_var(Vector3(PI/4, singularBeta - Scalar(1.5)*epsilon, PI/3));
    check_all_var(Vector3(PI/4, singularBeta - 2*epsilon, PI/3));
    check_all_var(Vector3(PI*Scalar(0.8), singularBeta - epsilon, Scalar(0.9)*PI));
    check_all_var(Vector3(PI*Scalar(-0.9), singularBeta + epsilon, PI*Scalar(0.3)));
    check_all_var(Vector3(PI*Scalar(-0.6), singularBeta + Scalar(1.5)*epsilon, PI*Scalar(0.3)));
    check_all_var(Vector3(PI*Scalar(-0.5), singularBeta + 2*epsilon, PI*Scalar(0.4)));
    check_all_var(Vector3(PI*Scalar(0.9), singularBeta + epsilon, Scalar(0.8)*PI));
  }
  
  // This one for sanity, it had a problem with near pole cases in float scalar.
  check_all_var(Vector3(PI*Scalar(0.8), singularBeta - Scalar(1E-6), Scalar(0.9)*PI));
}

template<typename Scalar> void eulerangles_manual()
{
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Matrix<Scalar,Dynamic,1> VectorX;
  const Vector3 Zero = Vector3::Zero();
  const Scalar PI = Scalar(EIGEN_PI);
  
  check_all_var(Zero);
  
  // singular cases
  check_singular_cases(PI/2);
  check_singular_cases(-PI/2);
  
  check_singular_cases(Scalar(0));
  check_singular_cases(Scalar(-0));
  
  check_singular_cases(PI);
  check_singular_cases(-PI);
  
  // non-singular cases
  VectorX alpha = VectorX::LinSpaced(20, Scalar(-0.99) * PI, PI);
  VectorX beta =  VectorX::LinSpaced(20, Scalar(-0.49) * PI, Scalar(0.49) * PI);
  VectorX gamma = VectorX::LinSpaced(20, Scalar(-0.99) * PI, PI);
  for (int i = 0; i < alpha.size(); ++i) {
    for (int j = 0; j < beta.size(); ++j) {
      for (int k = 0; k < gamma.size(); ++k) {
        check_all_var(Vector3(alpha(i), beta(j), gamma(k)));
      }
    }
  }
}

template<typename Scalar> void eulerangles_rand()
{
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Array<Scalar,3,1> Array3;
  typedef Quaternion<Scalar> Quaternionx;
  typedef AngleAxis<Scalar> AngleAxisType;

  Scalar a = internal::random<Scalar>(-Scalar(EIGEN_PI), Scalar(EIGEN_PI));
  Quaternionx q1;
  q1 = AngleAxisType(a, Vector3::Random().normalized());
  Matrix3 m;
  m = q1;
  
  Vector3 ea = m.eulerAngles(0,1,2);
  check_all_var(ea);
  ea = m.eulerAngles(0,1,0);
  check_all_var(ea);
  
  // Check with purely random Quaternion:
  q1.coeffs() = Quaternionx::Coefficients::Random().normalized();
  m = q1;
  ea = m.eulerAngles(0,1,2);
  check_all_var(ea);
  ea = m.eulerAngles(0,1,0);
  check_all_var(ea);
  
  // Check with random angles in range [0:pi]x[-pi:pi]x[-pi:pi].
  ea = (Array3::Random() + Array3(1,0,0))*Scalar(EIGEN_PI)*Array3(0.5,1,1);
  check_all_var(ea);
  
  ea[2] = ea[0] = internal::random<Scalar>(0,Scalar(EIGEN_PI));
  check_all_var(ea);
  
  ea[0] = ea[1] = internal::random<Scalar>(0,Scalar(EIGEN_PI));
  check_all_var(ea);
  
  ea[1] = 0;
  check_all_var(ea);
  
  ea.head(2).setZero();
  check_all_var(ea);
  
  ea.setZero();
  check_all_var(ea);
}

EIGEN_DECLARE_TEST(EulerAngles)
{
  // Simple cast test
  EulerAnglesXYZd onesEd(1, 1, 1);
  EulerAnglesXYZf onesEf = onesEd.cast<float>();
  VERIFY_IS_APPROX(onesEd, onesEf.cast<double>());

  // Simple Construction from Vector3 test
  VERIFY_IS_APPROX(onesEd, EulerAnglesXYZd(Vector3d::Ones()));
  
  CALL_SUBTEST_1( eulerangles_manual<float>() );
  CALL_SUBTEST_2( eulerangles_manual<double>() );
  
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_3( eulerangles_rand<float>() );
    CALL_SUBTEST_4( eulerangles_rand<double>() );
  }
  
  // TODO: Add tests for auto diff
  // TODO: Add tests for complex numbers
}
