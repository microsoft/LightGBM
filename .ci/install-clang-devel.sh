#!/bin/bash

set -ux pipefail

CLANG_MAJOR_VERSION=${1}

# remove clang stuff that comes installed in the image
apt-get autoremove -y --purge \
    clang-* \
    libclang-* \
    libunwind-* \
    llvm-*

# replace it all with clang-${CLANG_MAJOR_VERSION}
apt-get update -y
apt-get install --no-install-recommends -y \
    gnupg \
    lsb-release \
    software-properties-common \
    wget

wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -

add-apt-repository "deb http://apt.llvm.org/unstable/ llvm-toolchain main"
apt-get update -y
apt-get install -y --no-install-recommends \
    clang-${CLANG_MAJOR_VERSION} \
    clangd-${CLANG_MAJOR_VERSION} \
    clang-format-${CLANG_MAJOR_VERSION} \
    clang-tidy-${CLANG_MAJOR_VERSION} \
    clang-tools-${CLANG_MAJOR_VERSION} \
    lldb-${CLANG_MAJOR_VERSION} \
    lld-${CLANG_MAJOR_VERSION} \
    llvm-${CLANG_MAJOR_VERSION}-dev \
    llvm-${CLANG_MAJOR_VERSION}-tools \
    libomp-${CLANG_MAJOR_VERSION}-dev \
    libc++-${CLANG_MAJOR_VERSION}-dev \
    libc++abi-${CLANG_MAJOR_VERSION}-dev \
    libclang-common-${CLANG_MAJOR_VERSION}-dev \
    libclang-${CLANG_MAJOR_VERSION}-dev \
    libclang-cpp${CLANG_MAJOR_VERSION}-dev \
    libunwind-${CLANG_MAJOR_VERSION}-dev

# overwrite everything in /usr/bin with the new v15 versions
cp --remove-destination /usr/lib/llvm-${CLANG_MAJOR_VERSION}/bin/* /usr/bin/

echo ""
echo "done install clang"
clang --version
echo ""
