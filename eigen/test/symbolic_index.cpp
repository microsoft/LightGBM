// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifdef EIGEN_TEST_PART_2
#define EIGEN_MAX_CPP_VER 03

// see indexed_view.cpp
#if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8))
  #pragma GCC diagnostic ignored "-Wdeprecated"
#endif

#endif

#include "main.h"

template<typename T1,typename T2>
bool is_same_symb(const T1& a, const T2& b, Index size)
{
  return a.eval(last=size-1) == b.eval(last=size-1);
}

template<typename T>
void check_is_symbolic(const T&) {
  STATIC_CHECK(( symbolic::is_symbolic<T>::value ))
}

template<typename T>
void check_isnot_symbolic(const T&) {
  STATIC_CHECK(( !symbolic::is_symbolic<T>::value ))
}

#define VERIFY_EQ_INT(A,B) VERIFY_IS_APPROX(int(A),int(B))

void check_symbolic_index()
{
  check_is_symbolic(last);
  check_is_symbolic(lastp1);
  check_is_symbolic(last+1);
  check_is_symbolic(last-lastp1);
  check_is_symbolic(2*last-lastp1/2);
  check_isnot_symbolic(fix<3>());

  Index size=100;

  // First, let's check FixedInt arithmetic:
  VERIFY( is_same_type( (fix<5>()-fix<3>())*fix<9>()/(-fix<3>()), fix<-(5-3)*9/3>() ) );
  VERIFY( is_same_type( (fix<5>()-fix<3>())*fix<9>()/fix<2>(), fix<(5-3)*9/2>() ) );
  VERIFY( is_same_type( fix<9>()/fix<2>(), fix<9/2>() ) );
  VERIFY( is_same_type( fix<9>()%fix<2>(), fix<9%2>() ) );
  VERIFY( is_same_type( fix<9>()&fix<2>(), fix<9&2>() ) );
  VERIFY( is_same_type( fix<9>()|fix<2>(), fix<9|2>() ) );
  VERIFY( is_same_type( fix<9>()/2, int(9/2) ) );

  VERIFY( is_same_symb( lastp1-1, last, size) );
  VERIFY( is_same_symb( lastp1-fix<1>, last, size) );

  VERIFY_IS_EQUAL( ( (last*5-2)/3 ).eval(last=size-1), ((size-1)*5-2)/3 );
  VERIFY_IS_EQUAL( ( (last*fix<5>-fix<2>)/fix<3> ).eval(last=size-1), ((size-1)*5-2)/3 );
  VERIFY_IS_EQUAL( ( -last*lastp1  ).eval(last=size-1), -(size-1)*size );
  VERIFY_IS_EQUAL( ( lastp1-3*last  ).eval(last=size-1), size- 3*(size-1) );
  VERIFY_IS_EQUAL( ( (lastp1-3*last)/lastp1  ).eval(last=size-1), (size- 3*(size-1))/size );

#if EIGEN_HAS_CXX14
  {
    struct x_tag {};  static const symbolic::SymbolExpr<x_tag> x;
    struct y_tag {};  static const symbolic::SymbolExpr<y_tag> y;
    struct z_tag {};  static const symbolic::SymbolExpr<z_tag> z;

    VERIFY_IS_APPROX( int(((x+3)/y+z).eval(x=6,y=3,z=-13)), (6+3)/3+(-13) );
  }
#endif
}

EIGEN_DECLARE_TEST(symbolic_index)
{
  CALL_SUBTEST_1( check_symbolic_index() );
  CALL_SUBTEST_2( check_symbolic_index() );
}
