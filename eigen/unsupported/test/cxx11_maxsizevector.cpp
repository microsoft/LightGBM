#include "main.h"

#include <exception>  // std::exception

#include <unsupported/Eigen/CXX11/Tensor>

struct Foo
{
  static Index object_count;
  static Index object_limit;
  EIGEN_ALIGN_TO_BOUNDARY(128) int dummy;

  Foo(int x=0) : dummy(x)
  {
#ifdef EIGEN_EXCEPTIONS
    // TODO: Is this the correct way to handle this?
    if (Foo::object_count > Foo::object_limit) { std::cout << "\nThrow!\n"; throw Foo::Fail(); }
#endif
    std::cout << '+';
    ++Foo::object_count;
    eigen_assert((internal::UIntPtr(this) & (127)) == 0);
  }
  Foo(const Foo&)
  {
    std::cout << 'c';
    ++Foo::object_count;
    eigen_assert((internal::UIntPtr(this) & (127)) == 0);
  }

  ~Foo()
  {
    std::cout << '~';
    --Foo::object_count;
  }

  class Fail : public std::exception {};
};

Index Foo::object_count = 0;
Index Foo::object_limit = 0;



EIGEN_DECLARE_TEST(cxx11_maxsizevector)
{
  typedef MaxSizeVector<Foo> VectorX;
  Foo::object_count = 0;
  for(int r = 0; r < g_repeat; r++) {
    Index rows = internal::random<Index>(3,30);
    Foo::object_limit = internal::random<Index>(0, rows - 2);
    std::cout << "object_limit = " << Foo::object_limit << std::endl;
    bool exception_raised = false;
#ifdef EIGEN_EXCEPTIONS
    try
    {
#endif
      std::cout <<       "\nVectorX m(" << rows << ");\n";
      VectorX vect(rows);
      for(int i=0; i<rows; ++i)
          vect.push_back(Foo());
#ifdef EIGEN_EXCEPTIONS
      VERIFY(false);  // not reached if exceptions are enabled
    }
    catch (const Foo::Fail&) { exception_raised = true; }
    VERIFY(exception_raised);
#endif
    VERIFY_IS_EQUAL(Index(0), Foo::object_count);

    {
      Foo::object_limit = rows+1;
      VectorX vect2(rows, Foo());
      VERIFY_IS_EQUAL(Foo::object_count, rows);
    }
    VERIFY_IS_EQUAL(Index(0), Foo::object_count);
    std::cout << '\n';
  }
}
