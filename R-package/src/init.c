
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

void R_init_lib_lightgbm(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, NULL, NULL, NULL);
  R_useDynamicSymbols(dll, TRUE);
}
