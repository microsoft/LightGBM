/**
 * Wrap ChunkedArray.hpp class for SWIG usage.
 *
 * Author: Alberto Ferreira
 */

%{
#include "../include/LightGBM/utils/ChunkedArray.hpp"
%}

%include "../include/LightGBM/utils/ChunkedArray.hpp"

%template(int32ChunkedArray) ChunkedArray<int32_t>;
/* Unfortunately, for the time being,
 * SWIG has issues generating the overloads to coalesce_to()
 * for larger integral types
 * so we won't support that for now:
 */
//%template(int64ChunkedArray) ChunkedArray<int64_t>;
%template(floatChunkedArray) ChunkedArray<float>;
%template(doubleChunkedArray) ChunkedArray<double>;
