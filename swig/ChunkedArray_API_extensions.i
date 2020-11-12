/**
 * Wrap ChunkedArray.hpp class for SWIG usage.
 */

%{
#include "../swig/ChunkedArray.hpp"
%}

%include "../swig/ChunkedArray.hpp"

%template(intChunkedArray) ChunkedArray<int32_t>;
%template(floatChunkedArray) ChunkedArray<float>;
%template(doubleChunkedArray) ChunkedArray<double>;