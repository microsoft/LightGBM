FROM quay.io/pypa/manylinux_2_28_aarch64

RUN yum update -y \
 && yum install -y \
    epel-release \
    gcc-c++ \
    hwloc-devel \
    ocl-icd-devel \
    sudo \
 && yum module install -y \
      llvm-toolset \
 && yum clean all \
 && rm -rf /var/cache/yum

RUN git clone --depth 1 --branch v1.8 https://github.com/pocl/pocl.git \
 && cmake \
    -B pocl/build \
    -S pocl \
    -DCMAKE_BUILD_TYPE=release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_C_FLAGS=-std=gnu99 \
    -DPOCL_INSTALL_ICD_VENDORDIR=/etc/OpenCL/vendors \
    -DPOCL_DEBUG_MESSAGES=OFF \
    -DINSTALL_OPENCL_HEADERS=OFF \
    -DENABLE_SPIR=OFF \
    -DENABLE_POCLCC=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_EXAMPLES=OFF \
    -DLLC_HOST_CPU=generic \
 && cmake --build pocl/build -j4 \
 && sudo cmake --install pocl/build
