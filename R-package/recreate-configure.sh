#!/bin/bash

# recreates 'configure' from 'configure.ac'
# this script should run on Ubuntu 18.04
AUTOCONF_VERSION=$(cat R-package/AUTOCONF_UBUNTU_VERSION)

echo "Creating 'configure' script with Autoconf ${AUTOCONF_VERSION}"

apt update
apt-get install \
    --no-install-recommends \
    -y \
        autoconf=${AUTOCONF_VERSION}

cp VERSION.txt R-package/inst/
autoconf \
    --output R-package/configure \
    R-package/configure.ac

rm -r autom4te.cache || echo "no autoconf cache found"

echo "done creating 'configure' script"
