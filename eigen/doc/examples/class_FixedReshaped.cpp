#include <Eigen/Core>
#include <iostream>
using namespace Eigen;
using namespace std;

template<typename Derived>
Eigen::Reshaped<Derived, 4, 2>
reshape_helper(MatrixBase<Derived>& m)
{
  return Eigen::Reshaped<Derived, 4, 2>(m.derived());
}

int main(int, char**)
{
  MatrixXd m(2, 4);
  m << 1, 2, 3, 4,
       5, 6, 7, 8;
  MatrixXd n = reshape_helper(m);
  cout << "matrix m is:" << endl << m << endl;
  cout << "matrix n is:" << endl << n << endl;
  return 0;
}
