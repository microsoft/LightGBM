#!/bin/sh

echo "running cpplint"
cpplint \
    --filter=-build/c++11,-build/include_subdir,-build/header_guard,-whitespace/line_length \
    --recursive ./src ./include ./R-package ./swig ./tests \
|| exit -1
echo "done running cpplint"

echo "running cmakelint"
cmake_files=$(
    find . -name CMakeLists.txt -o -path "./cmake/*.cmake" \
    | grep -v external_libs
)
cmakelint \
    --linelength=120 \
    --filter=-convention/filename,-package/stdargs,-readability/wonkycase \
    ${cmake_files} \
|| exit -1
echo "done running cmakelint"
