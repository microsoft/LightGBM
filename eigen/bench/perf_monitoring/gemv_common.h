#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <functional>
#include "eigen_src/Eigen/Core"
#include "../BenchTimer.h"
using namespace Eigen;

#ifndef SCALAR
#error SCALAR must be defined
#endif

typedef SCALAR Scalar;

typedef Matrix<Scalar,Dynamic,Dynamic> Mat;
typedef Matrix<Scalar,Dynamic,1>       Vec;

template<typename Func>
EIGEN_DONT_INLINE
double bench(long m, long n, Func &f)
{
  Mat A(m,n);
  Vec B(n);
  Vec C(m);
  A.setRandom();
  B.setRandom();
  C.setRandom();

  BenchTimer t;

  double up = 1e8/sizeof(Scalar);
  double tm0 = 4, tm1 = 10;
  if(NumTraits<Scalar>::IsComplex)
  {
    up /= 4;
    tm0 = 2;
    tm1 = 4;
  }

  double flops = 2. * m * n;
  long rep = std::max(1., std::min(100., up/flops) );
  long tries = std::max(tm0, std::min(tm1, up/flops) );

  BENCH(t, tries, rep, f(A,B,C));

  return 1e-9 * rep * flops / t.best();
}

template<typename Func>
int main_gemv(int argc, char **argv, Func& f)
{
  std::vector<double> results;

  std::string filename = std::string("gemv_settings.txt");
  if(argc>1)
    filename = std::string(argv[1]);
  std::ifstream settings(filename);
  long m, n;
  while(settings >> m >> n)
  {
    //std::cerr << "  Testing " << m << " " << n << std::endl;
    results.push_back( bench(m, n, f) );
  }

  std::cout << RowVectorXd::Map(results.data(), results.size());

  return 0;
}
