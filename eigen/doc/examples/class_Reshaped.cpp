#include <Eigen/Core>
#include <iostream>
using namespace std;
using namespace Eigen;

template<typename Derived>
const Reshaped<const Derived>
reshape_helper(const MatrixBase<Derived>& m, int rows, int cols)
{
  return Reshaped<const Derived>(m.derived(), rows, cols);
}

int main(int, char**)
{
  MatrixXd m(3, 4);
  m << 1, 4, 7, 10,
       2, 5, 8, 11,
       3, 6, 9, 12;
  cout << m << endl;
  Ref<const MatrixXd> n = reshape_helper(m, 2, 6);
  cout << "Matrix m is:" << endl << m << endl;
  cout << "Matrix n is:" << endl << n << endl;
}
