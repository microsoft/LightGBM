#!/bin/bash

sh build-cran-package.sh || exit -1

apt-get install --no-install-recommends -y \
  libcurl4-openssl-dev \
  libxml2-dev \
  libssl-dev

log_file="$GITHUB_WORKSPACE/rhub_logs.txt"
Rscript -e "install.packages('rhub', dependencies = c('Depends', 'Imports', 'LinkingTo'), repos = 'https://cran.r-project.org', Ncpus = parallel::detectCores())" || exit -1

Rscript $GITHUB_WORKSPACE/.ci/run_rhub_solaris_checks.R $(pwd)/lightgbm_*.tar.gz $log_file || exit -1
