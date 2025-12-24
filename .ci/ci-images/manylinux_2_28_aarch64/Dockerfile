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
   gcc-c++ \
   hwloc-devel \
   llvm-devel \
   llvm-static \
   ocl-icd-devel \
   sudo

yum module install --nodocs -y \
   llvm-toolset

yum clean all

rm -rf /var/cache/yum
EOF

RUN <<EOF
# install a newer CMake than what the package manager has
curl -sL https://cmake.org/files/v3.23/cmake-3.23.1-linux-aarch64.sh -o cmake.sh
chmod +x cmake.sh
./cmake.sh --prefix=/usr/local --exclude-subdir
rm -f ./cmake.sh

# build PoCL
git clone \
   --depth 1 \
   --branch v7.1 \
   https://github.com/pocl/pocl.git

cmake \
   -B pocl/build \
   -S pocl \
   -DCMAKE_BUILD_TYPE=release \
   -DCMAKE_C_COMPILER="${DEVTOOLSET_ROOTPATH}/usr/bin/gcc" \
   -DCMAKE_CXX_COMPILER="${DEVTOOLSET_ROOTPATH}/usr/bin/g++" \
   -DENABLE_DOXYGEN=OFF \
   -DENABLE_EXAMPLES=OFF \
   -DENABLE_HOST_CPU_DEVICES=ON \
   -DENABLE_HWLOC=ON \
   -DENABLE_POCLCC=OFF \
   -DENABLE_SPIRV=OFF \
   -DENABLE_TESTS=OFF \
   -DENABLE_VALGRIND=OFF \
   -DINSTALL_OPENCL_HEADERS=OFF \
   -DLLC_HOST_CPU=generic \
   -DPOCL_DEBUG_MESSAGES=OFF \
   -DPOCL_INSTALL_ICD_VENDORDIR=/etc/OpenCL/vendors

cmake --build pocl/build -j4

cmake --install pocl/build
EOF
