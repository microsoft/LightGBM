FROM quay.io/pypa/manylinux_2_28_aarch64

# use 'bash' for RUN steps
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# install packages
RUN <<EOF
yum update -y

yum install --nodocs -y \
   epel-release

yum install --nodocs -y \
   clang-devel \
   hwloc-devel \
   llvm-devel \
   llvm-static \
   ocl-icd-devel \
   opencl-headers \
   sudo

yum module install -y \
   llvm-toolset

yum clean all

rm -rf /var/cache/yum
EOF

RUN <<EOF
# install a newer CMake than what the package manager has
curl -sL https://cmake.org/files/v4.2/cmake-4.2.1-linux-aarch64.sh -o cmake.sh
chmod +x cmake.sh
./cmake.sh --prefix=/usr/local --exclude-subdir
rm -f ./cmake.sh

# build PoCL
#
# NOTE: If this is updated, check if CL_TARGET_OPENCL_VERSION in cmake/IntegratedOpenCL.cmake
#       needs to be updated (see comments there fore links).
git clone \
   --depth 1 \
   --branch v7.1 \
   https://github.com/pocl/pocl.git

# explanations for some flags:
#
#   * -DCMAKE_{C,CXX}_COMPILER: DEVTOOLSET_ROOTPATH is where manylinux puts the gcc toolset
#   * -DENABLE_ICD=ON: select appropriate OpenCL platform at runtime (https://github.com/pocl/pocl/blob/013d2f19f4e8f2e0fd9aedcb70117d6dcc737aa9/doc/sphinx/source/using.rst#installable-client-driver-icd)
#   * -DLLC_HOST_CPU="generic": passed to clang's -march/-mcpu flag.
#
cmake \
   -B pocl/build \
   -S pocl \
   -DCMAKE_BUILD_TYPE=release \
   -DCMAKE_C_COMPILER="clang" \
   -DCMAKE_CXX_COMPILER="clang++" \
   -DENABLE_DOXYGEN=OFF \
   -DENABLE_ICD=ON \
   -DENABLE_LLVM=ON \
   -DENABLE_EXAMPLES=OFF \
   -DENABLE_HOST_CPU_DEVICES=ON \
   -DENABLE_HWLOC=ON \
   -DENABLE_POCLCC=ON \
   -DENABLE_SPIRV=ON \
   -DENABLE_TESTS=OFF \
   -DENABLE_VALGRIND=OFF \
   -DINSTALL_OPENCL_HEADERS=OFF \
   -DLLC_HOST_CPU=generic \
   -DPOCL_DEBUG_MESSAGES=OFF \
   -DPOCL_INSTALL_ICD_VENDORDIR=/etc/OpenCL/vendors

cmake --build pocl/build -j4

cmake \
   --install pocl/build \
   --component dev \
   --component icd
EOF
