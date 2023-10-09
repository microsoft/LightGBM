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

echo "checking that all OpenMP pragmas specify num_threads()"
get_omp_pragmas_without_num_threads() {
    grep \
        -n \
        -R \
        --include='*.c' \
        --include='*.cc' \
        --include='*.cpp' \
        --include='*.h' \
        --include='*.hpp' \
        'pragma omp parallel' \
    | grep -v ' num_threads'
}
PROBLEMATIC_LINES=$(
    get_omp_pragmas_without_num_threads
)
if test "${PROBLEMATIC_LINES}" != ""; then
    get_omp_pragmas_without_num_threads
    echo "All '#pragma omp parallel' uses should specify num_threads() explicitly."
    exit -1
fi
