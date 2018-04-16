/* lightgbmlib.i */
%module lightgbmlib
%ignore LGBM_BoosterSaveModelToString;
%{
/* Includes the header in the wrapper code */
#include "../include/LightGBM/export.h"
#include "../include/LightGBM/utils/log.h"
#include "../include/LightGBM/c_api.h"
%}

/* header files */
%include "../include/LightGBM/export.h"
%include "../include/LightGBM/c_api.h"
%include "cpointer.i"
%include "carrays.i"

%inline %{
  char * LGBM_BoosterSaveModelToStringSWIG(BoosterHandle handle,
					   int num_iteration,
					   int64_t buffer_len,
					   int64_t* out_len) {
    char* dst = new char[buffer_len];
    int result = LGBM_BoosterSaveModelToString(handle, num_iteration, buffer_len, out_len, dst);
    if (result != 0) {
      return nullptr;
    }
    return dst;
  }
%}

%pointer_functions(int, intp)
%pointer_functions(long, longp)
%pointer_functions(double, doublep)
%pointer_functions(float, floatp)
%pointer_functions(int64_t, int64_tp)
%pointer_functions(int32_t, int32_tp)

%pointer_cast(int64_t *, long *, int64_t_to_long_ptr)
%pointer_cast(int64_t *, double *, int64_t_to_double_ptr)
%pointer_cast(int32_t *, int *, int32_t_to_int_ptr)
%pointer_cast(long *, int64_t *, long_to_int64_t_ptr)
%pointer_cast(double *, int64_t *, double_to_int64_t_ptr)
%pointer_cast(double *, void *, double_to_voidp_ptr)
%pointer_cast(int *, int32_t *, int_to_int32_t_ptr)
%pointer_cast(float *, void *, float_to_voidp_ptr)

%array_functions(double, doubleArray)
%array_functions(float, floatArray)
%array_functions(int, intArray)
%array_functions(long, longArray)

/* Custom pointer manipulation template */
%define %pointer_manipulation(TYPE,NAME)
%{
  static TYPE *new_##NAME() { %}
  %{  TYPE* NAME = new TYPE; return NAME; %}
  %{}

  static void delete_##NAME(TYPE *self) { %}
  %{  if (self) delete self; %}
  %{}
  %}

TYPE *new_##NAME();
void  delete_##NAME(TYPE *self);

%enddef

%define %pointer_dereference(TYPE,NAME)
%{
  static TYPE NAME ##_value(TYPE *self) {
    TYPE NAME = *self;
    return NAME;
  }
%}

TYPE NAME##_value(TYPE *self);

%enddef

%define %pointer_handle(TYPE,NAME)
%{
  static TYPE* NAME ##_handle() { %}
  %{ TYPE* NAME = new TYPE; *NAME = (TYPE)operator new(sizeof(int*)); return NAME; %}
  %{}
%}

TYPE *NAME##_handle();

%enddef

%pointer_manipulation(void*, voidpp)

/* Allow dereferencing of void** to void* */
%pointer_dereference(void*, voidpp)

/* Allow retrieving handle to void** */
%pointer_handle(void*, voidpp)

