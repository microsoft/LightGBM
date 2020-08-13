#include "gemv_common.h"

EIGEN_DONT_INLINE
void trmv(const Mat &A, const Vec &B, Vec &C)
{
  C.noalias() += A.triangularView<Lower>() * B;
}

int main(int argc, char **argv)
{
  return main_gemv(argc, argv, trmv);
}
