// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018-2019 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iterator>
#include <numeric>
#include "main.h"

template< class Iterator >
std::reverse_iterator<Iterator>
make_reverse_iterator( Iterator i )
{
  return std::reverse_iterator<Iterator>(i);
}

#if !EIGEN_HAS_CXX11
template<class ForwardIt>
ForwardIt is_sorted_until(ForwardIt firstIt, ForwardIt lastIt)
{
    if (firstIt != lastIt) {
        ForwardIt next = firstIt;
        while (++next != lastIt) {
            if (*next < *firstIt)
                return next;
            firstIt = next;
        }
    }
    return lastIt;
}
template<class ForwardIt>
bool is_sorted(ForwardIt firstIt, ForwardIt lastIt)
{
    return ::is_sorted_until(firstIt, lastIt) == lastIt;
}
#else
using std::is_sorted;
#endif

template<typename XprType>
bool is_pointer_based_stl_iterator(const internal::pointer_based_stl_iterator<XprType> &) { return true; }

template<typename XprType>
bool is_generic_randaccess_stl_iterator(const internal::generic_randaccess_stl_iterator<XprType> &) { return true; }

template<typename Xpr>
void check_begin_end_for_loop(Xpr xpr)
{
  const Xpr& cxpr(xpr);
  Index i = 0;

  i = 0;
  for(typename Xpr::iterator it = xpr.begin(); it!=xpr.end(); ++it) { VERIFY_IS_EQUAL(*it,xpr[i++]); }

  i = 0;
  for(typename Xpr::const_iterator it = xpr.cbegin(); it!=xpr.cend(); ++it) { VERIFY_IS_EQUAL(*it,xpr[i++]); }

  i = 0;
  for(typename Xpr::const_iterator it = cxpr.begin(); it!=cxpr.end(); ++it) { VERIFY_IS_EQUAL(*it,xpr[i++]); }

  i = 0;
  for(typename Xpr::const_iterator it = xpr.begin(); it!=xpr.end(); ++it) { VERIFY_IS_EQUAL(*it,xpr[i++]); }

  {
    // simple API check
    typename Xpr::const_iterator cit = xpr.begin();
    cit = xpr.cbegin();

    #if EIGEN_HAS_CXX11
    auto tmp1 = xpr.begin();
    VERIFY(tmp1==xpr.begin());
    auto tmp2 = xpr.cbegin();
    VERIFY(tmp2==xpr.cbegin());
    #endif
  }

  VERIFY( xpr.end() -xpr.begin()  == xpr.size() );
  VERIFY( xpr.cend()-xpr.begin()  == xpr.size() );
  VERIFY( xpr.end() -xpr.cbegin() == xpr.size() );
  VERIFY( xpr.cend()-xpr.cbegin() == xpr.size() );

  if(xpr.size()>0) {
    VERIFY(xpr.begin() != xpr.end());
    VERIFY(xpr.begin() < xpr.end());
    VERIFY(xpr.begin() <= xpr.end());
    VERIFY(!(xpr.begin() == xpr.end()));
    VERIFY(!(xpr.begin() > xpr.end()));
    VERIFY(!(xpr.begin() >= xpr.end()));
    
    VERIFY(xpr.cbegin() != xpr.end());
    VERIFY(xpr.cbegin() < xpr.end());
    VERIFY(xpr.cbegin() <= xpr.end());
    VERIFY(!(xpr.cbegin() == xpr.end()));
    VERIFY(!(xpr.cbegin() > xpr.end()));
    VERIFY(!(xpr.cbegin() >= xpr.end()));

    VERIFY(xpr.begin() != xpr.cend());
    VERIFY(xpr.begin() < xpr.cend());
    VERIFY(xpr.begin() <= xpr.cend());
    VERIFY(!(xpr.begin() == xpr.cend()));
    VERIFY(!(xpr.begin() > xpr.cend()));
    VERIFY(!(xpr.begin() >= xpr.cend()));
  }
}

template<typename Scalar, int Rows, int Cols>
void test_stl_iterators(int rows=Rows, int cols=Cols)
{
  typedef Matrix<Scalar,Rows,1> VectorType;
  #if EIGEN_HAS_CXX11
  typedef Matrix<Scalar,1,Cols> RowVectorType;
  #endif
  typedef Matrix<Scalar,Rows,Cols,ColMajor> ColMatrixType;
  typedef Matrix<Scalar,Rows,Cols,RowMajor> RowMatrixType;
  VectorType v = VectorType::Random(rows);
  const VectorType& cv(v);
  ColMatrixType A = ColMatrixType::Random(rows,cols);
  const ColMatrixType& cA(A);
  RowMatrixType B = RowMatrixType::Random(rows,cols);
  
  Index i, j;

  // Check we got a fast pointer-based iterator when expected
  {
    VERIFY( is_pointer_based_stl_iterator(v.begin()) );
    VERIFY( is_pointer_based_stl_iterator(v.end()) );
    VERIFY( is_pointer_based_stl_iterator(cv.begin()) );
    VERIFY( is_pointer_based_stl_iterator(cv.end()) );

    j = internal::random<Index>(0,A.cols()-1);
    VERIFY( is_pointer_based_stl_iterator(A.col(j).begin()) );
    VERIFY( is_pointer_based_stl_iterator(A.col(j).end()) );
    VERIFY( is_pointer_based_stl_iterator(cA.col(j).begin()) );
    VERIFY( is_pointer_based_stl_iterator(cA.col(j).end()) );

    i = internal::random<Index>(0,A.rows()-1);
    VERIFY( is_pointer_based_stl_iterator(A.row(i).begin()) );
    VERIFY( is_pointer_based_stl_iterator(A.row(i).end()) );
    VERIFY( is_pointer_based_stl_iterator(cA.row(i).begin()) );
    VERIFY( is_pointer_based_stl_iterator(cA.row(i).end()) );

    VERIFY( is_pointer_based_stl_iterator(A.reshaped().begin()) );
    VERIFY( is_pointer_based_stl_iterator(A.reshaped().end()) );
    VERIFY( is_pointer_based_stl_iterator(cA.reshaped().begin()) );
    VERIFY( is_pointer_based_stl_iterator(cA.reshaped().end()) );

    VERIFY( is_pointer_based_stl_iterator(B.template reshaped<AutoOrder>().begin()) );
    VERIFY( is_pointer_based_stl_iterator(B.template reshaped<AutoOrder>().end()) );

    VERIFY( is_generic_randaccess_stl_iterator(A.template reshaped<RowMajor>().begin()) );
    VERIFY( is_generic_randaccess_stl_iterator(A.template reshaped<RowMajor>().end()) );
  }

  {
    check_begin_end_for_loop(v);
    check_begin_end_for_loop(A.col(internal::random<Index>(0,A.cols()-1)));
    check_begin_end_for_loop(A.row(internal::random<Index>(0,A.rows()-1)));
    check_begin_end_for_loop(v+v);
  }

#if EIGEN_HAS_CXX11
  // check swappable
  {
    using std::swap;
    // pointer-based
    {
      VectorType v_copy = v;
      auto a = v.begin();
      auto b = v.end()-1;
      swap(a,b);
      VERIFY_IS_EQUAL(v,v_copy);
      VERIFY_IS_EQUAL(*b,*v.begin());
      VERIFY_IS_EQUAL(*b,v(0));
      VERIFY_IS_EQUAL(*a,v.end()[-1]);
      VERIFY_IS_EQUAL(*a,v(last));
    }

    // generic
    {
      RowMatrixType B_copy = B;
      auto Br = B.reshaped();
      auto a = Br.begin();
      auto b = Br.end()-1;
      swap(a,b);
      VERIFY_IS_EQUAL(B,B_copy);
      VERIFY_IS_EQUAL(*b,*Br.begin());
      VERIFY_IS_EQUAL(*b,Br(0));
      VERIFY_IS_EQUAL(*a,Br.end()[-1]);
      VERIFY_IS_EQUAL(*a,Br(last));
    }
  }

  // check non-const iterator with for-range loops
  {
    i = 0;
    for(auto x : v) { VERIFY_IS_EQUAL(x,v[i++]); }

    j = internal::random<Index>(0,A.cols()-1);
    i = 0;
    for(auto x : A.col(j)) { VERIFY_IS_EQUAL(x,A(i++,j)); }

    i = 0;
    for(auto x : (v+A.col(j))) { VERIFY_IS_APPROX(x,v(i)+A(i,j)); ++i; }

    j = 0;
    i = internal::random<Index>(0,A.rows()-1);
    for(auto x : A.row(i)) { VERIFY_IS_EQUAL(x,A(i,j++)); }

    i = 0;
    for(auto x : A.reshaped()) { VERIFY_IS_EQUAL(x,A(i++)); }
  }

  // same for const_iterator
  {
    i = 0;
    for(auto x : cv) { VERIFY_IS_EQUAL(x,v[i++]); }

    i = 0;
    for(auto x : cA.reshaped()) { VERIFY_IS_EQUAL(x,A(i++)); }

    j = 0;
    i = internal::random<Index>(0,A.rows()-1);
    for(auto x : cA.row(i)) { VERIFY_IS_EQUAL(x,A(i,j++)); }
  }

  // check reshaped() on row-major
  {
    i = 0;
    Matrix<Scalar,Dynamic,Dynamic,ColMajor> Bc = B;
    for(auto x : B.reshaped()) { VERIFY_IS_EQUAL(x,Bc(i++)); }
  }

  // check write access
  {
    VectorType w(v.size());
    i = 0;
    for(auto& x : w) { x = v(i++); }
    VERIFY_IS_EQUAL(v,w);
  }

  // check for dangling pointers
  {
    // no dangling because pointer-based
    {
      j = internal::random<Index>(0,A.cols()-1);
      auto it = A.col(j).begin();
      for(i=0;i<rows;++i) {
        VERIFY_IS_EQUAL(it[i],A(i,j));
      }
    }

    // no dangling because pointer-based
    {
      i = internal::random<Index>(0,A.rows()-1);
      auto it = A.row(i).begin();
      for(j=0;j<cols;++j) { VERIFY_IS_EQUAL(it[j],A(i,j)); }
    }

    {
      j = internal::random<Index>(0,A.cols()-1);
      // this would produce a dangling pointer:
      // auto it = (A+2*A).col(j).begin(); 
      // we need to name the temporary expression:
      auto tmp = (A+2*A).col(j);
      auto it = tmp.begin();
      for(i=0;i<rows;++i) {
        VERIFY_IS_APPROX(it[i],3*A(i,j));
      }
    }
  }

  {
    // check basic for loop on vector-wise iterators
    j=0;
    for (auto it = A.colwise().cbegin(); it != A.colwise().cend(); ++it, ++j) {
      VERIFY_IS_APPROX( it->coeff(0), A(0,j) );
      VERIFY_IS_APPROX( (*it).coeff(0), A(0,j) );
    }
    j=0;
    for (auto it = A.colwise().begin(); it != A.colwise().end(); ++it, ++j) {
      (*it).coeffRef(0) = (*it).coeff(0); // compilation check
      it->coeffRef(0) = it->coeff(0);     // compilation check
      VERIFY_IS_APPROX( it->coeff(0), A(0,j) );
      VERIFY_IS_APPROX( (*it).coeff(0), A(0,j) );
    }

    // check valuetype gives us a copy
    j=0;
    for (auto it = A.colwise().cbegin(); it != A.colwise().cend(); ++it, ++j) {
      typename decltype(it)::value_type tmp = *it;
      VERIFY_IS_NOT_EQUAL( tmp.data() , it->data() );
      VERIFY_IS_APPROX( tmp, A.col(j) );
    }
  }

#endif

  if(rows>=3) {
    VERIFY_IS_EQUAL((v.begin()+rows/2)[1], v(rows/2+1));

    VERIFY_IS_EQUAL((A.rowwise().begin()+rows/2)[1], A.row(rows/2+1));
  }

  if(cols>=3) {
    VERIFY_IS_EQUAL((A.colwise().begin()+cols/2)[1], A.col(cols/2+1));
  }

  // check std::sort
  {
    // first check that is_sorted returns false when required
    if(rows>=2)
    {
      v(1) = v(0)-Scalar(1);
      #if EIGEN_HAS_CXX11
      VERIFY(!is_sorted(std::begin(v),std::end(v)));
      #else
      VERIFY(!is_sorted(v.cbegin(),v.cend()));
      #endif
    }

    // on a vector
    {
      std::sort(v.begin(),v.end());
      VERIFY(is_sorted(v.begin(),v.end()));
      VERIFY(!::is_sorted(make_reverse_iterator(v.end()),make_reverse_iterator(v.begin())));
    }

    // on a column of a column-major matrix -> pointer-based iterator and default increment
    {
      j = internal::random<Index>(0,A.cols()-1);
      // std::sort(begin(A.col(j)),end(A.col(j))); // does not compile because this returns const iterators
      typename ColMatrixType::ColXpr Acol = A.col(j);
      std::sort(Acol.begin(),Acol.end());
      VERIFY(is_sorted(Acol.cbegin(),Acol.cend()));
      A.setRandom();

      std::sort(A.col(j).begin(),A.col(j).end());
      VERIFY(is_sorted(A.col(j).cbegin(),A.col(j).cend()));
      A.setRandom();
    }

    // on a row of a rowmajor matrix -> pointer-based iterator and runtime increment
    {
      i = internal::random<Index>(0,A.rows()-1);
      typename ColMatrixType::RowXpr Arow = A.row(i);
      VERIFY_IS_EQUAL( std::distance(Arow.begin(),Arow.end()), cols);
      std::sort(Arow.begin(),Arow.end());
      VERIFY(is_sorted(Arow.cbegin(),Arow.cend()));
      A.setRandom();

      std::sort(A.row(i).begin(),A.row(i).end());
      VERIFY(is_sorted(A.row(i).cbegin(),A.row(i).cend()));
      A.setRandom();
    }

    // with a generic iterator
    {
      Reshaped<RowMatrixType,RowMatrixType::SizeAtCompileTime,1> B1 = B.reshaped();
      std::sort(B1.begin(),B1.end());
      VERIFY(is_sorted(B1.cbegin(),B1.cend()));
      B.setRandom();

      // assertion because nested expressions are different
      // std::sort(B.reshaped().begin(),B.reshaped().end());
      // VERIFY(is_sorted(B.reshaped().cbegin(),B.reshaped().cend()));
      // B.setRandom();
    }
  }

  // check with partial_sum
  {
    j = internal::random<Index>(0,A.cols()-1);
    typename ColMatrixType::ColXpr Acol = A.col(j);
    std::partial_sum(Acol.begin(), Acol.end(), v.begin());
    VERIFY_IS_APPROX(v(seq(1,last)), v(seq(0,last-1))+Acol(seq(1,last)));

    // inplace
    std::partial_sum(Acol.begin(), Acol.end(), Acol.begin());
    VERIFY_IS_APPROX(v, Acol);
  }

  // stress random access as required by std::nth_element
  if(rows>=3)
  {
    v.setRandom();
    VectorType v1 = v;
    std::sort(v1.begin(),v1.end());
    std::nth_element(v.begin(), v.begin()+rows/2, v.end());
    VERIFY_IS_APPROX(v1(rows/2), v(rows/2));

    v.setRandom();
    v1 = v;
    std::sort(v1.begin()+rows/2,v1.end());
    std::nth_element(v.begin()+rows/2, v.begin()+rows/4, v.end());
    VERIFY_IS_APPROX(v1(rows/4), v(rows/4));
  }

#if EIGEN_HAS_CXX11
  // check rows/cols iterators with range-for loops
  {
    j = 0;
    for(auto c : A.colwise()) { VERIFY_IS_APPROX(c.sum(), A.col(j).sum()); ++j; }
    j = 0;
    for(auto c : B.colwise()) { VERIFY_IS_APPROX(c.sum(), B.col(j).sum()); ++j; }

    j = 0;
    for(auto c : B.colwise()) {
      i = 0;
      for(auto& x : c) {
        VERIFY_IS_EQUAL(x, B(i,j));
        x = A(i,j);
        ++i;
      }
      ++j;
    }
    VERIFY_IS_APPROX(A,B);
    B.setRandom();
    
    i = 0;
    for(auto r : A.rowwise()) { VERIFY_IS_APPROX(r.sum(), A.row(i).sum()); ++i; }
    i = 0;
    for(auto r : B.rowwise()) { VERIFY_IS_APPROX(r.sum(), B.row(i).sum()); ++i; }
  }


  // check rows/cols iterators with STL algorithms
  {
    RowVectorType row = RowVectorType::Random(cols);
    A.rowwise() = row;
    VERIFY( std::all_of(A.rowwise().begin(),  A.rowwise().end(),  [&row](typename ColMatrixType::RowXpr x) { return internal::isApprox(x.squaredNorm(),row.squaredNorm()); }) );
    VERIFY( std::all_of(A.rowwise().rbegin(), A.rowwise().rend(), [&row](typename ColMatrixType::RowXpr x) { return internal::isApprox(x.squaredNorm(),row.squaredNorm()); }) );

    VectorType col = VectorType::Random(rows);
    A.colwise() = col;
    VERIFY( std::all_of(A.colwise().begin(),   A.colwise().end(),   [&col](typename ColMatrixType::ColXpr x) { return internal::isApprox(x.squaredNorm(),col.squaredNorm()); }) );
    VERIFY( std::all_of(A.colwise().rbegin(),  A.colwise().rend(),  [&col](typename ColMatrixType::ColXpr x) { return internal::isApprox(x.squaredNorm(),col.squaredNorm()); }) );
    VERIFY( std::all_of(A.colwise().cbegin(),  A.colwise().cend(),  [&col](typename ColMatrixType::ConstColXpr x) { return internal::isApprox(x.squaredNorm(),col.squaredNorm()); }) );
    VERIFY( std::all_of(A.colwise().crbegin(), A.colwise().crend(), [&col](typename ColMatrixType::ConstColXpr x) { return internal::isApprox(x.squaredNorm(),col.squaredNorm()); }) );

    i = internal::random<Index>(0,A.rows()-1);
    A.setRandom();
    A.row(i).setZero();
    VERIFY_IS_EQUAL( std::find_if(A.rowwise().begin(),  A.rowwise().end(),  [](typename ColMatrixType::RowXpr x) { return x.squaredNorm() == Scalar(0); })-A.rowwise().begin(),  i );
    VERIFY_IS_EQUAL( std::find_if(A.rowwise().rbegin(), A.rowwise().rend(), [](typename ColMatrixType::RowXpr x) { return x.squaredNorm() == Scalar(0); })-A.rowwise().rbegin(), (A.rows()-1) - i );

    j = internal::random<Index>(0,A.cols()-1);
    A.setRandom();
    A.col(j).setZero();
    VERIFY_IS_EQUAL( std::find_if(A.colwise().begin(),  A.colwise().end(),  [](typename ColMatrixType::ColXpr x) { return x.squaredNorm() == Scalar(0); })-A.colwise().begin(),  j );
    VERIFY_IS_EQUAL( std::find_if(A.colwise().rbegin(), A.colwise().rend(), [](typename ColMatrixType::ColXpr x) { return x.squaredNorm() == Scalar(0); })-A.colwise().rbegin(), (A.cols()-1) - j );
  }

  {
    using VecOp = VectorwiseOp<ArrayXXi, 0>;
    STATIC_CHECK(( internal::is_same<VecOp::const_iterator, decltype(std::declval<const VecOp&>().cbegin())>::value ));
    STATIC_CHECK(( internal::is_same<VecOp::const_iterator, decltype(std::declval<const VecOp&>().cend  ())>::value ));
    #if EIGEN_COMP_CXXVER>=14
      STATIC_CHECK(( internal::is_same<VecOp::const_iterator, decltype(std::cbegin(std::declval<const VecOp&>()))>::value ));
      STATIC_CHECK(( internal::is_same<VecOp::const_iterator, decltype(std::cend  (std::declval<const VecOp&>()))>::value ));
    #endif
  }

#endif
}


#if EIGEN_HAS_CXX11
// When the compiler sees expression IsContainerTest<C>(0), if C is an
// STL-style container class, the first overload of IsContainerTest
// will be viable (since both C::iterator* and C::const_iterator* are
// valid types and NULL can be implicitly converted to them).  It will
// be picked over the second overload as 'int' is a perfect match for
// the type of argument 0.  If C::iterator or C::const_iterator is not
// a valid type, the first overload is not viable, and the second
// overload will be picked.
template <class C,
          class Iterator = decltype(::std::declval<const C&>().begin()),
          class = decltype(::std::declval<const C&>().end()),
          class = decltype(++::std::declval<Iterator&>()),
          class = decltype(*::std::declval<Iterator>()),
          class = typename C::const_iterator>
bool IsContainerType(int /* dummy */) { return true; }

template <class C>
bool IsContainerType(long /* dummy */) { return false; }

template <typename Scalar, int Rows, int Cols>
void test_stl_container_detection(int rows=Rows, int cols=Cols)
{
  typedef Matrix<Scalar,Rows,1> VectorType;
  typedef Matrix<Scalar,Rows,Cols,ColMajor> ColMatrixType;
  typedef Matrix<Scalar,Rows,Cols,RowMajor> RowMatrixType;

  ColMatrixType A = ColMatrixType::Random(rows, cols);
  RowMatrixType B = RowMatrixType::Random(rows, cols);

  Index i = 1;

  using ColMatrixColType = decltype(A.col(i));
  using ColMatrixRowType = decltype(A.row(i));
  using RowMatrixColType = decltype(B.col(i));
  using RowMatrixRowType = decltype(B.row(i));

  // Vector and matrix col/row are valid Stl-style container.
  VERIFY_IS_EQUAL(IsContainerType<VectorType>(0), true);
  VERIFY_IS_EQUAL(IsContainerType<ColMatrixColType>(0), true);
  VERIFY_IS_EQUAL(IsContainerType<ColMatrixRowType>(0), true);
  VERIFY_IS_EQUAL(IsContainerType<RowMatrixColType>(0), true);
  VERIFY_IS_EQUAL(IsContainerType<RowMatrixRowType>(0), true);

  // But the matrix itself is not a valid Stl-style container.
  VERIFY_IS_EQUAL(IsContainerType<ColMatrixType>(0), rows == 1 || cols == 1);
  VERIFY_IS_EQUAL(IsContainerType<RowMatrixType>(0), rows == 1 || cols == 1);
}
#endif

EIGEN_DECLARE_TEST(stl_iterators)
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(( test_stl_iterators<double,2,3>() ));
    CALL_SUBTEST_1(( test_stl_iterators<float,7,5>() ));
    CALL_SUBTEST_1(( test_stl_iterators<int,Dynamic,Dynamic>(internal::random<int>(5,10), internal::random<int>(5,10)) ));
    CALL_SUBTEST_1(( test_stl_iterators<int,Dynamic,Dynamic>(internal::random<int>(10,200), internal::random<int>(10,200)) ));
  }
  
#if EIGEN_HAS_CXX11
  CALL_SUBTEST_1(( test_stl_container_detection<float,1,1>() ));
  CALL_SUBTEST_1(( test_stl_container_detection<float,5,5>() ));
#endif  
}
