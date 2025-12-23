FROM quay.io/pypa/manylinux_2_28_x86_64

# ensure that libraries like libc++ built in this image can be found by the linker
ENV LD_LIBRARY_PATH="/usr/local/lib64:${LD_LIBRARY_PATH}:/usr/local/lib"

RUN yum update -y \
 && yum install -y \
    sudo \
 && yum clean all

# Install CMake
RUN curl -sL https://cmake.org/files/v3.23/cmake-3.23.1-linux-x86_64.sh -o cmake.sh \
 && chmod +x cmake.sh \
 && ./cmake.sh --prefix=/usr/local --exclude-subdir \
 && rm -f ./cmake.sh

# building clang
#
# [why v11.1.0?]
#
# - LightGBM is incompatible with libomp v12 and v13 (https://github.com/microsoft/LightGBM/issues/4229)
# - PoCL v1.8 requires LLVM+clang, but only supports v6-v13 (https://github.com/pocl/pocl/blob/3f420ef735672e439097d020db605778dbc4a6a1/cmake/LLVM.cmake#L209)
# - so v11.x is the latest version that will work with the these constraints, and v11.1.0 was the last release in the v11 series
#
# [why from source?]
#
# - manylinux images use a custom sysroot foor the gcc toolchain, which doesn't play well with the precompiled llvm+clang available from https://github.com/llvm/llvm-project/releases
# - there is not a precompiled llvm+clang available for x86_64 architecture + non-Ubuntu distributions
#
# [why are we building libcxx and lld?]
#
# - in case compilation with clang (e.g. of PoCL v1.8) uses libc++ instead of libstdc++
#
# https://clang.llvm.org/get_started.html
# https://llvm.org/docs/GettingStarted.html#compiling-the-llvm-suite-source-code
ARG CLANG_VER=11.1.0
RUN mkdir /usr/local/src/clang \
 && cd /usr/local/src/clang \
 && git clone --depth 1 --branch llvmorg-$CLANG_VER https://github.com/llvm/llvm-project.git \
 && cd llvm-project \
 && mkdir build \
 && cd build \
 && cmake \
     -DCMAKE_BUILD_TYPE=Release \
     -DGCC_INSTALL_PREFIX=${DEVTOOLSET_ROOTPATH} \
     -DLLVM_BUILD_BENCHMARKS=OFF \
     -DLLVM_BUILD_DOCS=OFF \
     -DLLVM_BUILD_TESTS=OFF \
     -DLLVM_INCLUDE_BENCHMARKS=OFF \
     -DLLVM_INCLUDE_DOCS=OFF \
     -DLLVM_INCLUDE_TESTS=OFF \
     -DLLVM_ENABLE_BINDINGS=OFF \
     -DLLVM_ENABLE_DOXYGEN=OFF \
     -DLLVM_ENABLE_OCAMLDOC=OFF \
     -DLLVM_ENABLE_PROJECTS="clang;openmp" \
     -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" \
     -DLLVM_ENABLE_SPHINX=OFF \
     -DLLVM_TARGETS_TO_BUILD="X86" \
     -G "Unix Makefiles" \
     ../llvm \
 && make -j2 \
 && make install \
 # `make install` places libc++ into a different directory,
 # symlinking it into /usr/local/lib so the linker can find it
 && cp -s /usr/local/lib/x86_64-unknown-linux-gnu/c++/* /usr/local/lib/ \
 # manylinux images already come with an ldconfig rule pointing at /usr/local/lib,
 # but just doing this explicitly to be safe in case they remove that in the future
 && echo /usr/local/lib > /etc/ld.so.conf.d/llvm.conf \
 && ldconfig \
 && cd "${HOME}" \
 && rm -rf /usr/local/src/clang

# Install PoCL
RUN git clone --depth 1 --branch v1.8 https://github.com/pocl/pocl.git \
 && yum update -y \
 && yum install -y \
    ocl-icd-devel \
    opencl-headers \
 && cmake \
    -B pocl/build \
    -S pocl \
    -DCMAKE_BUILD_TYPE=release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_FLAGS=-stdlib=libstdc++ \
    -DPOCL_INSTALL_ICD_VENDORDIR=/etc/OpenCL/vendors \
    -DPOCL_DEBUG_MESSAGES=OFF \
    -DSTATIC_LLVM=ON \
    -DINSTALL_OPENCL_HEADERS=OFF \
    -DENABLE_SPIR=OFF \
    -DENABLE_POCLCC=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_EXAMPLES=OFF \
 && cmake --build pocl/build -j4 \
 && cmake --install pocl/build \
 && rm -f ./compile_test*.bc \
 && rm -f ./compile_test*.o \
 && rm -rf ./pocl \
 && yum clean all

# Install Java
RUN yum install -y \
      java-1.8.0-openjdk-devel.x86_64 \
 && yum clean all

ENV JAVA_HOME_8_X64=/usr/lib/jvm/java
ENV JAVA_HOME=$JAVA_HOME_8_X64

# Install SWIG
RUN curl -sLk https://sourceforge.net/projects/swig/files/swig/swig-4.0.2/swig-4.0.2.tar.gz/download -o swig.tar.gz \
 && tar -xzf swig.tar.gz \
 && cd swig-4.0.2 \
 && ./configure --prefix=/usr/local --without-pcre \
 && make \
 && make install \
 && cd .. \
 && rm -f ./swig.tar.gz \
 && rm -rf ./swig-4.0.2

# Install miniforge
RUN curl -sL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" -o miniforge.sh \
 && chmod +x miniforge.sh \
 && ./miniforge.sh -b -p /opt/miniforge \
 && rm -f ./miniforge.sh \
 && /opt/miniforge/bin/conda clean -a -y \
 && chmod -R 777 /opt/miniforge

ENV CONDA=/opt/miniforge/
