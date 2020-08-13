// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifdef EIGEN_TEST_PART_2
// Make sure we also check c++11 max implementation
#define EIGEN_MAX_CPP_VER 11
#endif

#ifdef EIGEN_TEST_PART_3
// Make sure we also check c++98 max implementation
#define EIGEN_MAX_CPP_VER 03

// We need to disable this warning when compiling with c++11 while limiting Eigen to c++98
// Ideally we would rather configure the compiler to build in c++98 mode but this needs
// to be done at the CMakeLists.txt level.
#if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8))
  #pragma GCC diagnostic ignored "-Wdeprecated"
#endif

#if defined(__GNUC__) && (__GNUC__ >=9)
  #pragma GCC diagnostic ignored "-Wdeprecated-copy"
#endif

#endif

#include <valarray>
#include <vector>
#include "main.h"

#if EIGEN_HAS_CXX11
#include <array>
#endif

typedef std::pair<Index,Index> IndexPair;

int encode(Index i, Index j) {
  return int(i*100 + j);
}

IndexPair decode(Index ij) {
  return IndexPair(ij / 100, ij % 100);
}

template<typename T>
bool match(const T& xpr, std::string ref, std::string str_xpr = "") {
  EIGEN_UNUSED_VARIABLE(str_xpr);
  std::stringstream str;
  str << xpr;
  if(!(str.str() == ref))
    std::cout << str_xpr << "\n" << xpr << "\n\n";
  return str.str() == ref;
}

#define MATCH(X,R) match(X, R, #X)

template<typename T1,typename T2>
typename internal::enable_if<internal::is_same<T1,T2>::value,bool>::type
is_same_eq(const T1& a, const T2& b)
{
  return (a == b).all();
}

template<typename T1,typename T2>
bool is_same_seq(const T1& a, const T2& b)
{
  bool ok = a.first()==b.first() && a.size() == b.size() && Index(a.incrObject())==Index(b.incrObject());;
  if(!ok)
  {
    std::cerr << "seqN(" << a.first() << ", " << a.size() << ", " << Index(a.incrObject()) << ") != ";
    std::cerr << "seqN(" << b.first() << ", " << b.size() << ", " << Index(b.incrObject()) << ")\n";
  }
  return ok;
}

template<typename T1,typename T2>
typename internal::enable_if<internal::is_same<T1,T2>::value,bool>::type
is_same_seq_type(const T1& a, const T2& b)
{
  return is_same_seq(a,b);
}



#define VERIFY_EQ_INT(A,B) VERIFY_IS_APPROX(int(A),int(B))

// C++03 does not allow local or unnamed enums as index
enum DummyEnum { XX=0, YY=1 };

void check_indexed_view()
{
  Index n = 10;

  ArrayXd a = ArrayXd::LinSpaced(n,0,n-1);
  Array<double,1,Dynamic> b = a.transpose();

  #if EIGEN_COMP_CXXVER>=14
  ArrayXXi A = ArrayXXi::NullaryExpr(n,n, std::ref(encode));
  #else
  ArrayXXi A = ArrayXXi::NullaryExpr(n,n, std::ptr_fun(&encode));
  #endif

  for(Index i=0; i<n; ++i)
    for(Index j=0; j<n; ++j)
      VERIFY( decode(A(i,j)) == IndexPair(i,j) );

  Array4i eii(4); eii << 3, 1, 6, 5;
  std::valarray<int> vali(4); Map<ArrayXi>(&vali[0],4) = eii;
  std::vector<int> veci(4); Map<ArrayXi>(veci.data(),4) = eii;

  VERIFY( MATCH( A(3, seq(9,3,-1)),
    "309  308  307  306  305  304  303")
  );

  VERIFY( MATCH( A(seqN(2,5), seq(9,3,-1)),
    "209  208  207  206  205  204  203\n"
    "309  308  307  306  305  304  303\n"
    "409  408  407  406  405  404  403\n"
    "509  508  507  506  505  504  503\n"
    "609  608  607  606  605  604  603")
  );

  VERIFY( MATCH( A(seqN(2,5), 5),
    "205\n"
    "305\n"
    "405\n"
    "505\n"
    "605")
  );

  VERIFY( MATCH( A(seqN(last,5,-1), seq(2,last)),
    "902  903  904  905  906  907  908  909\n"
    "802  803  804  805  806  807  808  809\n"
    "702  703  704  705  706  707  708  709\n"
    "602  603  604  605  606  607  608  609\n"
    "502  503  504  505  506  507  508  509")
  );

  VERIFY( MATCH( A(eii, veci),
    "303  301  306  305\n"
    "103  101  106  105\n"
    "603  601  606  605\n"
    "503  501  506  505")
  );

  VERIFY( MATCH( A(eii, all),
    "300  301  302  303  304  305  306  307  308  309\n"
    "100  101  102  103  104  105  106  107  108  109\n"
    "600  601  602  603  604  605  606  607  608  609\n"
    "500  501  502  503  504  505  506  507  508  509")
  );

  // take row number 3, and repeat it 5 times
  VERIFY( MATCH( A(seqN(3,5,0), all),
    "300  301  302  303  304  305  306  307  308  309\n"
    "300  301  302  303  304  305  306  307  308  309\n"
    "300  301  302  303  304  305  306  307  308  309\n"
    "300  301  302  303  304  305  306  307  308  309\n"
    "300  301  302  303  304  305  306  307  308  309")
  );

  VERIFY( MATCH( a(seqN(3,3),0), "3\n4\n5" ) );
  VERIFY( MATCH( a(seq(3,5)), "3\n4\n5" ) );
  VERIFY( MATCH( a(seqN(3,3,1)), "3\n4\n5" ) );
  VERIFY( MATCH( a(seqN(5,3,-1)), "5\n4\n3" ) );

  VERIFY( MATCH( b(0,seqN(3,3)), "3  4  5" ) );
  VERIFY( MATCH( b(seq(3,5)), "3  4  5" ) );
  VERIFY( MATCH( b(seqN(3,3,1)), "3  4  5" ) );
  VERIFY( MATCH( b(seqN(5,3,-1)), "5  4  3" ) );

  VERIFY( MATCH( b(all), "0  1  2  3  4  5  6  7  8  9" ) );
  VERIFY( MATCH( b(eii), "3  1  6  5" ) );

  Array44i B;
  B.setRandom();
  VERIFY( (A(seqN(2,5), 5)).ColsAtCompileTime == 1);
  VERIFY( (A(seqN(2,5), 5)).RowsAtCompileTime == Dynamic);
  VERIFY_EQ_INT( (A(seqN(2,5), 5)).InnerStrideAtCompileTime , A.InnerStrideAtCompileTime);
  VERIFY_EQ_INT( (A(seqN(2,5), 5)).OuterStrideAtCompileTime , A.col(5).OuterStrideAtCompileTime);

  VERIFY_EQ_INT( (A(5,seqN(2,5))).InnerStrideAtCompileTime , A.row(5).InnerStrideAtCompileTime);
  VERIFY_EQ_INT( (A(5,seqN(2,5))).OuterStrideAtCompileTime , A.row(5).OuterStrideAtCompileTime);
  VERIFY_EQ_INT( (B(1,seqN(1,2))).InnerStrideAtCompileTime , B.row(1).InnerStrideAtCompileTime);
  VERIFY_EQ_INT( (B(1,seqN(1,2))).OuterStrideAtCompileTime , B.row(1).OuterStrideAtCompileTime);

  VERIFY_EQ_INT( (A(seqN(2,5), seq(1,3))).InnerStrideAtCompileTime , A.InnerStrideAtCompileTime);
  VERIFY_EQ_INT( (A(seqN(2,5), seq(1,3))).OuterStrideAtCompileTime , A.OuterStrideAtCompileTime);
  VERIFY_EQ_INT( (B(seqN(1,2), seq(1,3))).InnerStrideAtCompileTime , B.InnerStrideAtCompileTime);
  VERIFY_EQ_INT( (B(seqN(1,2), seq(1,3))).OuterStrideAtCompileTime , B.OuterStrideAtCompileTime);
  VERIFY_EQ_INT( (A(seqN(2,5,2), seq(1,3,2))).InnerStrideAtCompileTime , Dynamic);
  VERIFY_EQ_INT( (A(seqN(2,5,2), seq(1,3,2))).OuterStrideAtCompileTime , Dynamic);
  VERIFY_EQ_INT( (A(seqN(2,5,fix<2>), seq(1,3,fix<3>))).InnerStrideAtCompileTime , 2);
  VERIFY_EQ_INT( (A(seqN(2,5,fix<2>), seq(1,3,fix<3>))).OuterStrideAtCompileTime , Dynamic);
  VERIFY_EQ_INT( (B(seqN(1,2,fix<2>), seq(1,3,fix<3>))).InnerStrideAtCompileTime , 2);
  VERIFY_EQ_INT( (B(seqN(1,2,fix<2>), seq(1,3,fix<3>))).OuterStrideAtCompileTime , 3*4);

  VERIFY_EQ_INT( (A(seqN(2,fix<5>), seqN(1,fix<3>))).RowsAtCompileTime, 5);
  VERIFY_EQ_INT( (A(seqN(2,fix<5>), seqN(1,fix<3>))).ColsAtCompileTime, 3);
  VERIFY_EQ_INT( (A(seqN(2,fix<5>(5)), seqN(1,fix<3>(3)))).RowsAtCompileTime, 5);
  VERIFY_EQ_INT( (A(seqN(2,fix<5>(5)), seqN(1,fix<3>(3)))).ColsAtCompileTime, 3);
  VERIFY_EQ_INT( (A(seqN(2,fix<Dynamic>(5)), seqN(1,fix<Dynamic>(3)))).RowsAtCompileTime, Dynamic);
  VERIFY_EQ_INT( (A(seqN(2,fix<Dynamic>(5)), seqN(1,fix<Dynamic>(3)))).ColsAtCompileTime, Dynamic);
  VERIFY_EQ_INT( (A(seqN(2,fix<Dynamic>(5)), seqN(1,fix<Dynamic>(3)))).rows(), 5);
  VERIFY_EQ_INT( (A(seqN(2,fix<Dynamic>(5)), seqN(1,fix<Dynamic>(3)))).cols(), 3);

  VERIFY( is_same_seq_type( seqN(2,5,fix<-1>), seqN(2,5,fix<-1>(-1)) ) );
  VERIFY( is_same_seq_type( seqN(2,5), seqN(2,5,fix<1>(1)) ) );
  VERIFY( is_same_seq_type( seqN(2,5,3), seqN(2,5,fix<DynamicIndex>(3)) ) );
  VERIFY( is_same_seq_type( seq(2,7,fix<3>), seqN(2,2,fix<3>) ) );
  VERIFY( is_same_seq_type( seqN(2,fix<Dynamic>(5),3), seqN(2,5,fix<DynamicIndex>(3)) ) );
  VERIFY( is_same_seq_type( seqN(2,fix<5>(5),fix<-2>), seqN(2,fix<5>,fix<-2>()) ) );

  VERIFY( is_same_seq_type( seq(2,fix<5>), seqN(2,4) ) );
#if EIGEN_HAS_CXX11
  VERIFY( is_same_seq_type( seq(fix<2>,fix<5>), seqN(fix<2>,fix<4>) ) );
  VERIFY( is_same_seq( seqN(2,std::integral_constant<int,5>(),std::integral_constant<int,-2>()), seqN(2,fix<5>,fix<-2>()) ) );
  VERIFY( is_same_seq( seq(std::integral_constant<int,1>(),std::integral_constant<int,5>(),std::integral_constant<int,2>()),
                       seq(fix<1>,fix<5>,fix<2>()) ) );
  VERIFY( is_same_seq_type( seqN(2,std::integral_constant<int,5>(),std::integral_constant<int,-2>()), seqN(2,fix<5>,fix<-2>()) ) );
  VERIFY( is_same_seq_type( seq(std::integral_constant<int,1>(),std::integral_constant<int,5>(),std::integral_constant<int,2>()),
                            seq(fix<1>,fix<5>,fix<2>()) ) );

  VERIFY( is_same_seq_type( seqN(2,std::integral_constant<int,5>()), seqN(2,fix<5>) ) );
  VERIFY( is_same_seq_type( seq(std::integral_constant<int,1>(),std::integral_constant<int,5>()), seq(fix<1>,fix<5>) ) );
#else
  // sorry, no compile-time size recovery in c++98/03
  VERIFY( is_same_seq( seq(fix<2>,fix<5>), seqN(fix<2>,fix<4>) ) );
#endif

  VERIFY( (A(seqN(2,fix<5>), 5)).RowsAtCompileTime == 5);
  VERIFY( (A(4, all)).ColsAtCompileTime == Dynamic);
  VERIFY( (A(4, all)).RowsAtCompileTime == 1);
  VERIFY( (B(1, all)).ColsAtCompileTime == 4);
  VERIFY( (B(1, all)).RowsAtCompileTime == 1);
  VERIFY( (B(all,1)).ColsAtCompileTime == 1);
  VERIFY( (B(all,1)).RowsAtCompileTime == 4);

  VERIFY(int( (A(all, eii)).ColsAtCompileTime) == int(eii.SizeAtCompileTime));
  VERIFY_EQ_INT( (A(eii, eii)).Flags&DirectAccessBit, (unsigned int)(0));
  VERIFY_EQ_INT( (A(eii, eii)).InnerStrideAtCompileTime, 0);
  VERIFY_EQ_INT( (A(eii, eii)).OuterStrideAtCompileTime, 0);

  VERIFY_IS_APPROX( A(seq(n-1,2,-2), seqN(n-1-6,3,-1)), A(seq(last,2,fix<-2>), seqN(last-6,3,fix<-1>)) );

  VERIFY_IS_APPROX( A(seq(n-1,2,-2), seqN(n-1-6,4)), A(seq(last,2,-2), seqN(last-6,4)) );
  VERIFY_IS_APPROX( A(seq(n-1-6,n-1-2), seqN(n-1-6,4)), A(seq(last-6,last-2), seqN(6+last-6-6,4)) );
  VERIFY_IS_APPROX( A(seq((n-1)/2,(n)/2+3), seqN(2,4)), A(seq(last/2,(last+1)/2+3), seqN(last+2-last,4)) );
  VERIFY_IS_APPROX( A(seq(n-2,2,-2), seqN(n-8,4)), A(seq(lastp1-2,2,-2), seqN(lastp1-8,4)) );

  // Check all combinations of seq:
  VERIFY_IS_APPROX( A(seq(1,n-1-2,2), seq(1,n-1-2,2)), A(seq(1,last-2,2), seq(1,last-2,fix<2>)) );
  VERIFY_IS_APPROX( A(seq(n-1-5,n-1-2,2), seq(n-1-5,n-1-2,2)), A(seq(last-5,last-2,2), seq(last-5,last-2,fix<2>)) );
  VERIFY_IS_APPROX( A(seq(n-1-5,7,2), seq(n-1-5,7,2)), A(seq(last-5,7,2), seq(last-5,7,fix<2>)) );
  VERIFY_IS_APPROX( A(seq(1,n-1-2), seq(n-1-5,7)), A(seq(1,last-2), seq(last-5,7)) );
  VERIFY_IS_APPROX( A(seq(n-1-5,n-1-2), seq(n-1-5,n-1-2)), A(seq(last-5,last-2), seq(last-5,last-2)) );

  VERIFY_IS_APPROX( A.col(A.cols()-1), A(all,last) );
  VERIFY_IS_APPROX( A(A.rows()-2, A.cols()/2), A(last-1, lastp1/2) );
  VERIFY_IS_APPROX( a(a.size()-2), a(last-1) );
  VERIFY_IS_APPROX( a(a.size()/2), a((last+1)/2) );

  // Check fall-back to Block
  {
    VERIFY( is_same_eq(A.col(0), A(all,0)) );
    VERIFY( is_same_eq(A.row(0), A(0,all)) );
    VERIFY( is_same_eq(A.block(0,0,2,2), A(seqN(0,2),seq(0,1))) );
    VERIFY( is_same_eq(A.middleRows(2,4), A(seqN(2,4),all)) );
    VERIFY( is_same_eq(A.middleCols(2,4), A(all,seqN(2,4))) );

    VERIFY( is_same_eq(A.col(A.cols()-1), A(all,last)) );

    const ArrayXXi& cA(A);
    VERIFY( is_same_eq(cA.col(0), cA(all,0)) );
    VERIFY( is_same_eq(cA.row(0), cA(0,all)) );
    VERIFY( is_same_eq(cA.block(0,0,2,2), cA(seqN(0,2),seq(0,1))) );
    VERIFY( is_same_eq(cA.middleRows(2,4), cA(seqN(2,4),all)) );
    VERIFY( is_same_eq(cA.middleCols(2,4), cA(all,seqN(2,4))) );

    VERIFY( is_same_eq(a.head(4), a(seq(0,3))) );
    VERIFY( is_same_eq(a.tail(4), a(seqN(last-3,4))) );
    VERIFY( is_same_eq(a.tail(4), a(seq(lastp1-4,last))) );
    VERIFY( is_same_eq(a.segment<4>(3), a(seqN(3,fix<4>))) );
  }

  ArrayXXi A1=A, A2 = ArrayXXi::Random(4,4);
  ArrayXi range25(4); range25 << 3,2,4,5;
  A1(seqN(3,4),seq(2,5)) = A2;
  VERIFY_IS_APPROX( A1.block(3,2,4,4), A2 );
  A1 = A;
  A2.setOnes();
  A1(seq(6,3,-1),range25) = A2;
  VERIFY_IS_APPROX( A1.block(3,2,4,4), A2 );

  // check reverse
  {
    VERIFY( is_same_seq_type( seq(3,7).reverse(), seqN(7,5,fix<-1>)  ) );
    VERIFY( is_same_seq_type( seq(7,3,fix<-2>).reverse(), seqN(3,3,fix<2>)  ) );
    VERIFY_IS_APPROX( a(seqN(2,last/2).reverse()), a(seqN(2+(last/2-1)*1,last/2,fix<-1>)) );
    VERIFY_IS_APPROX( a(seqN(last/2,fix<4>).reverse()),a(seqN(last/2,fix<4>)).reverse() );
    VERIFY_IS_APPROX( A(seq(last-5,last-1,2).reverse(), seqN(last-3,3,fix<-2>).reverse()),
                      A(seq(last-5,last-1,2), seqN(last-3,3,fix<-2>)).reverse() );
  }

#if EIGEN_HAS_CXX11
  // check lastN
  VERIFY_IS_APPROX( a(lastN(3)), a.tail(3) );
  VERIFY( MATCH( a(lastN(3)), "7\n8\n9" ) );
  VERIFY_IS_APPROX( a(lastN(fix<3>())), a.tail<3>() );
  VERIFY( MATCH( a(lastN(3,2)), "5\n7\n9" ) );
  VERIFY( MATCH( a(lastN(3,fix<2>())), "5\n7\n9" ) );
  VERIFY( a(lastN(fix<3>())).SizeAtCompileTime == 3 );

  VERIFY( (A(all, std::array<int,4>{{1,3,2,4}})).ColsAtCompileTime == 4);

  VERIFY_IS_APPROX( (A(std::array<int,3>{{1,3,5}}, std::array<int,4>{{9,6,3,0}})), A(seqN(1,3,2), seqN(9,4,-3)) );

#if EIGEN_HAS_STATIC_ARRAY_TEMPLATE
  VERIFY_IS_APPROX( A({3, 1, 6, 5}, all), A(std::array<int,4>{{3, 1, 6, 5}}, all) );
  VERIFY_IS_APPROX( A(all,{3, 1, 6, 5}), A(all,std::array<int,4>{{3, 1, 6, 5}}) );
  VERIFY_IS_APPROX( A({1,3,5},{3, 1, 6, 5}), A(std::array<int,3>{{1,3,5}},std::array<int,4>{{3, 1, 6, 5}}) );

  VERIFY_IS_EQUAL( A({1,3,5},{3, 1, 6, 5}).RowsAtCompileTime, 3 );
  VERIFY_IS_EQUAL( A({1,3,5},{3, 1, 6, 5}).ColsAtCompileTime, 4 );

  VERIFY_IS_APPROX( a({3, 1, 6, 5}), a(std::array<int,4>{{3, 1, 6, 5}}) );
  VERIFY_IS_EQUAL( a({1,3,5}).SizeAtCompileTime, 3 );

  VERIFY_IS_APPROX( b({3, 1, 6, 5}), b(std::array<int,4>{{3, 1, 6, 5}}) );
  VERIFY_IS_EQUAL( b({1,3,5}).SizeAtCompileTime, 3 );
#endif

#endif

  // check mat(i,j) with weird types for i and j
  {
    VERIFY_IS_APPROX( A(B.RowsAtCompileTime-1, 1), A(3,1) );
    VERIFY_IS_APPROX( A(B.RowsAtCompileTime, 1), A(4,1) );
    VERIFY_IS_APPROX( A(B.RowsAtCompileTime-1, B.ColsAtCompileTime-1), A(3,3) );
    VERIFY_IS_APPROX( A(B.RowsAtCompileTime, B.ColsAtCompileTime), A(4,4) );
    const Index I_ = 3, J_ = 4;
    VERIFY_IS_APPROX( A(I_,J_), A(3,4) );
  }

  // check extended block API
  {
    VERIFY( is_same_eq( A.block<3,4>(1,1), A.block(1,1,fix<3>,fix<4>)) );
    VERIFY( is_same_eq( A.block<3,4>(1,1,3,4), A.block(1,1,fix<3>(),fix<4>(4))) );
    VERIFY( is_same_eq( A.block<3,Dynamic>(1,1,3,4), A.block(1,1,fix<3>,4)) );
    VERIFY( is_same_eq( A.block<Dynamic,4>(1,1,3,4), A.block(1,1,fix<Dynamic>(3),fix<4>)) );
    VERIFY( is_same_eq( A.block(1,1,3,4), A.block(1,1,fix<Dynamic>(3),fix<Dynamic>(4))) );

    VERIFY( is_same_eq( A.topLeftCorner<3,4>(), A.topLeftCorner(fix<3>,fix<4>)) );
    VERIFY( is_same_eq( A.bottomLeftCorner<3,4>(), A.bottomLeftCorner(fix<3>,fix<4>)) );
    VERIFY( is_same_eq( A.bottomRightCorner<3,4>(), A.bottomRightCorner(fix<3>,fix<4>)) );
    VERIFY( is_same_eq( A.topRightCorner<3,4>(), A.topRightCorner(fix<3>,fix<4>)) );

    VERIFY( is_same_eq( A.leftCols<3>(), A.leftCols(fix<3>)) );
    VERIFY( is_same_eq( A.rightCols<3>(), A.rightCols(fix<3>)) );
    VERIFY( is_same_eq( A.middleCols<3>(1), A.middleCols(1,fix<3>)) );

    VERIFY( is_same_eq( A.topRows<3>(), A.topRows(fix<3>)) );
    VERIFY( is_same_eq( A.bottomRows<3>(), A.bottomRows(fix<3>)) );
    VERIFY( is_same_eq( A.middleRows<3>(1), A.middleRows(1,fix<3>)) );

    VERIFY( is_same_eq( a.segment<3>(1), a.segment(1,fix<3>)) );
    VERIFY( is_same_eq( a.head<3>(), a.head(fix<3>)) );
    VERIFY( is_same_eq( a.tail<3>(), a.tail(fix<3>)) );

    const ArrayXXi& cA(A);
    VERIFY( is_same_eq( cA.block<Dynamic,4>(1,1,3,4), cA.block(1,1,fix<Dynamic>(3),fix<4>)) );

    VERIFY( is_same_eq( cA.topLeftCorner<3,4>(), cA.topLeftCorner(fix<3>,fix<4>)) );
    VERIFY( is_same_eq( cA.bottomLeftCorner<3,4>(), cA.bottomLeftCorner(fix<3>,fix<4>)) );
    VERIFY( is_same_eq( cA.bottomRightCorner<3,4>(), cA.bottomRightCorner(fix<3>,fix<4>)) );
    VERIFY( is_same_eq( cA.topRightCorner<3,4>(), cA.topRightCorner(fix<3>,fix<4>)) );

    VERIFY( is_same_eq( cA.leftCols<3>(), cA.leftCols(fix<3>)) );
    VERIFY( is_same_eq( cA.rightCols<3>(), cA.rightCols(fix<3>)) );
    VERIFY( is_same_eq( cA.middleCols<3>(1), cA.middleCols(1,fix<3>)) );

    VERIFY( is_same_eq( cA.topRows<3>(), cA.topRows(fix<3>)) );
    VERIFY( is_same_eq( cA.bottomRows<3>(), cA.bottomRows(fix<3>)) );
    VERIFY( is_same_eq( cA.middleRows<3>(1), cA.middleRows(1,fix<3>)) );
  }

  // Check compilation of enums as index type:
  a(XX) = 1;
  A(XX,YY) = 1;
  // Anonymous enums only work with C++11
#if EIGEN_HAS_CXX11
  enum { X=0, Y=1 };
  a(X) = 1;
  A(X,Y) = 1;
  A(XX,Y) = 1;
  A(X,YY) = 1;
#endif

  // Check compilation of varying integer types as index types:
  Index i = n/2;
  short i_short(i);
  std::size_t i_sizet(i);
  VERIFY_IS_EQUAL( a(i), a.coeff(i_short) );
  VERIFY_IS_EQUAL( a(i), a.coeff(i_sizet) );

  VERIFY_IS_EQUAL( A(i,i), A.coeff(i_short, i_short) );
  VERIFY_IS_EQUAL( A(i,i), A.coeff(i_short, i) );
  VERIFY_IS_EQUAL( A(i,i), A.coeff(i, i_short) );
  VERIFY_IS_EQUAL( A(i,i), A.coeff(i, i_sizet) );
  VERIFY_IS_EQUAL( A(i,i), A.coeff(i_sizet, i) );
  VERIFY_IS_EQUAL( A(i,i), A.coeff(i_sizet, i_short) );
  VERIFY_IS_EQUAL( A(i,i), A.coeff(5, i_sizet) );

  // Regression test for Max{Rows,Cols}AtCompileTime
  {
    Matrix3i A3 = Matrix3i::Random();
    ArrayXi ind(5); ind << 1,1,1,1,1;
    VERIFY_IS_EQUAL( A3(ind,ind).eval(), MatrixXi::Constant(5,5,A3(1,1)) );
  }

  // Regression for bug 1736
  {
    VERIFY_IS_APPROX(A(all, eii).col(0).eval(), A.col(eii(0)));
    A(all, eii).col(0) = A.col(eii(0));
  }

  // bug 1815: IndexedView should allow linear access
  {
    VERIFY( MATCH( b(eii)(0), "3" ) );
    VERIFY( MATCH( a(eii)(0), "3" ) );
    VERIFY( MATCH( A(1,eii)(0), "103"));
    VERIFY( MATCH( A(eii,1)(0), "301"));
    VERIFY( MATCH( A(1,all)(1), "101"));
    VERIFY( MATCH( A(all,1)(1), "101"));
  }

#if EIGEN_HAS_CXX11
  //Bug IndexView with a single static row should be RowMajor:
  {
    // A(1, seq(0,2,1)).cwiseAbs().colwise().replicate(2).eval();
    STATIC_CHECK(( (internal::evaluator<decltype( A(1,seq(0,2,1)) )>::Flags & RowMajorBit) == RowMajorBit ));
  }
#endif

}

EIGEN_DECLARE_TEST(indexed_view)
{
//   for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( check_indexed_view() );
    CALL_SUBTEST_2( check_indexed_view() );
    CALL_SUBTEST_3( check_indexed_view() );
//   }

  // static checks of some internals:
  STATIC_CHECK(( internal::is_valid_index_type<int>::value ));
  STATIC_CHECK(( internal::is_valid_index_type<unsigned int>::value ));
  STATIC_CHECK(( internal::is_valid_index_type<short>::value ));
  STATIC_CHECK(( internal::is_valid_index_type<std::ptrdiff_t>::value ));
  STATIC_CHECK(( internal::is_valid_index_type<std::size_t>::value ));
  STATIC_CHECK(( !internal::valid_indexed_view_overload<int,int>::value ));
  STATIC_CHECK(( !internal::valid_indexed_view_overload<int,std::ptrdiff_t>::value ));
  STATIC_CHECK(( !internal::valid_indexed_view_overload<std::ptrdiff_t,int>::value ));
  STATIC_CHECK(( !internal::valid_indexed_view_overload<std::size_t,int>::value ));
}
