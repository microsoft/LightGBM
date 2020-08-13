#include "gemv_common.h"

EIGEN_DONT_INLINE
void trmv(const Mat &A, Vec &B, const Vec &C)
{
  B.noalias() += A.transpose().triangularView<Lower>() * C;
}

int main(int argc, char **argv)
{
  return main_gemv(argc, argv, trmv);
}
