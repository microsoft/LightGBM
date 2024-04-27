#!/bin/bash

# [description]
#
#   Installs a development version of clang and the other LLVM tools.
#

set -e -E -u -o pipefail

CLANG_VERSION=${1}

apt-get autoremove -y --purge \
    clang-* \
    libclang-* \
    libunwind-* \
    llvm-*

apt-get update -y
apt-get install --no-install-recommends -y \
    gnupg \
    lsb-release \
    software-properties-common \
    wget

wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -

# ref: https://apt.llvm.org/
add-apt-repository -y "deb http://apt.llvm.org/unstable/ llvm-toolchain main"
add-apt-repository -y "deb-src http://apt.llvm.org/unstable/ llvm-toolchain main"
add-apt-repository -y "deb http://apt.llvm.org/unstable/ llvm-toolchain-${CLANG_VERSION} main" || true
add-apt-repository -y "deb-src http://apt.llvm.org/unstable/ llvm-toolchain-${CLANG_VERSION} main" || true
apt-get update -y

apt-get install -y --no-install-recommends \
    clang-${CLANG_VERSION} \
    clangd-${CLANG_VERSION} \
    clang-format-${CLANG_VERSION} \
    clang-tidy-${CLANG_VERSION} \
    clang-tools-${CLANG_VERSION} \
    lldb-${CLANG_VERSION} \
    lld-${CLANG_VERSION} \
    llvm-${CLANG_VERSION}-dev \
    llvm-${CLANG_VERSION}-tools \
    libomp-${CLANG_VERSION}-dev \
    libc++-${CLANG_VERSION}-dev \
    libc++abi-${CLANG_VERSION}-dev \
    libclang-common-${CLANG_VERSION}-dev \
    libclang-${CLANG_VERSION}-dev \
    libclang-cpp${CLANG_VERSION}-dev \
    libunwind-${CLANG_VERSION}-dev

# overwriting the stuff in /usr/bin is simpler and more reliable than
# updating PATH, LD_LIBRARY_PATH, etc.
cp --remove-destination /usr/lib/llvm-${CLANG_VERSION}/bin/* /usr/bin/

# per https://www.stats.ox.ac.uk/pub/bdr/Rconfig/r-devel-linux-x86_64-fedora-clang
#
# clang was built to use libc++: for a version built to default to libstdc++
# (as shipped by Fedora/Debian/Ubuntu), add -stdlib=libc++ to CXX
# and install the libcxx-devel/libc++-dev package.
mkdir -p "${HOME}/.R"

cat << EOF > "${HOME}/.R/Makevars"
CXX += -stdlib=libc++
CXX11 += -stdlib=libc++
CXX14 += -stdlib=libc++
CXX17 += -stdlib=libc++
CXX20 += -stdlib=libc++
EOF

echo ""
echo "done installing clang"
clang --version
echo ""
