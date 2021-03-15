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
 * defining a new integral template specialization
 * causes type-inference conflicts in SWIG with the int32 version
 * of the coalesce_to() overload. I think this is due to some
 * SWIG limitation when implementing the wrappers:
 */
%template(int64ChunkedArray) ChunkedArray<int64_t>;
%template(floatChunkedArray) ChunkedArray<float>;
%template(doubleChunkedArray) ChunkedArray<double>;
