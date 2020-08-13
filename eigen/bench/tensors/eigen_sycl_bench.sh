rm -f tensor_benchmark_sycl
: "${COMPUTECPP_PACKAGE_ROOT_DIR:?Need to set COMPUTECPP_PACKAGE_ROOT_DIR}"
echo "COMPUTECPP_PACKAGE_ROOT_DIR is set to: "$COMPUTECPP_PACKAGE_ROOT_DIR
${COMPUTECPP_PACKAGE_ROOT_DIR}/bin/compute++ \
tensor_benchmarks_sycl.cc \
benchmark_main.cc \
-I ../../ \
-I ${COMPUTECPP_PACKAGE_ROOT_DIR}/include/ \
-std=c++11 \
-march=native \
-O3 \
-DNDEBUG \
-DEIGEN_MPL2_ONLY \
-DEIGEN_USE_SYCL=1 \
-DEIGEN_SYCL_LOCAL_MEM=1 \
-no-serial-memop \
-mllvm \
-inline-threshold=10000 \
-fsycl-ih-last \
-sycl-driver \
-Xclang -cl-mad-enable \
-lOpenCL \
-lComputeCpp \
-lpthread \
-o \
tensor_benchmark_sycl\
${@:1}

export LD_LIBRARY_PATH=${COMPUTECPP_PACKAGE_ROOT_DIR}/lib:$LD_LIBRARY_PATH
./tensor_benchmark_sycl
