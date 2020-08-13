#include "gemm_common.h"
#include <Eigen/Cholesky>

EIGEN_DONT_INLINE
void llt(const Mat &A, const Mat &B, Mat &C)
{
  C = A;
  C.diagonal().array() += 1000;
  Eigen::internal::llt_inplace<Mat::Scalar, Lower>::blocked(C);
}

int main(int argc, char **argv)
{
  return main_gemm(argc, argv, llt);
}
