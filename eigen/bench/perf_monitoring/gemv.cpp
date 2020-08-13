#include "gemv_common.h"

EIGEN_DONT_INLINE
void gemv(const Mat &A, const Vec &B, Vec &C)
{
  C.noalias() += A * B;
}

int main(int argc, char **argv)
{
  return main_gemv(argc, argv, gemv);
}
