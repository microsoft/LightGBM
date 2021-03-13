/**
 * Wrap ChunkedArray.hpp class for SWIG usage.
 */

%{
#include "../include/LightGBM/utils/ChunkedArray.hpp"
%}

%include "../include/LightGBM/utils/ChunkedArray.hpp"

%template(int32ChunkedArray) ChunkedArray<int32_t>;
%template(int64ChunkedArray) ChunkedArray<int64_t>;
%template(floatChunkedArray) ChunkedArray<float>;
%template(doubleChunkedArray) ChunkedArray<double>;
