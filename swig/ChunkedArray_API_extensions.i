/**
 * Wrap ChunkedArray.hpp class for SWIG usage.
 */

%{
#include "../include/LightGBM/utils/ChunkedArray.hpp"
%}

%include "../include/LightGBM/utils/ChunkedArray.hpp"

%template(intChunkedArray) ChunkedArray<int32_t>;
%template(floatChunkedArray) ChunkedArray<float>;
%template(doubleChunkedArray) ChunkedArray<double>;
