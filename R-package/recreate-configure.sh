#!/bin/bash

# recreates 'configure' from 'configure.ac'
# this script should run on Ubuntu 20.04
AUTOCONF_VERSION=$(cat R-package/AUTOCONF_UBUNTU_VERSION)

# R packages cannot have versions like 3.0.0rc1, but
# 3.0.0-1 is acceptable
LGB_VERSION=$(cat VERSION.txt | sed "s/rc/-/g")

# this script changes configure.ac. Copying to a temporary file
# so changes to configure.ac don't get committed in git
TMP_CONFIGURE_AC=".configure.ac"

echo "Creating 'configure' script with Autoconf ${AUTOCONF_VERSION}"

apt update
apt-get install \
    --no-install-recommends \
    -y \
        autoconf=${AUTOCONF_VERSION}

cd R-package

cp configure.ac ${TMP_CONFIGURE_AC}
sed -i.bak -e "s/~~VERSION~~/${LGB_VERSION}/" ${TMP_CONFIGURE_AC}

autoconf \
    --output configure \
    ${TMP_CONFIGURE_AC} \
    || exit -1

rm ${TMP_CONFIGURE_AC}

rm -r autom4te.cache || echo "no autoconf cache found"

echo "done creating 'configure' script"
