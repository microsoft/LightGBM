#include "gemm_common.h"

EIGEN_DONT_INLINE
void gemm(const Mat &A, const Mat &B, Mat &C)
{
  C.noalias() += A * B;
}

int main(int argc, char **argv)
{
  return main_gemm(argc, argv, gemm);
}
