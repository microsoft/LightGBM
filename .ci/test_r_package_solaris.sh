#!/bin/bash

apt-get install --no-install-recommends -y \
  libcurl4-openssl-dev \
  libxml2-dev \
  libssl-dev

Rscript -e "install.packages('rhub', dependencies = c('Depends', 'Imports', 'LinkingTo'), repos = 'https://cran.r-project.org', Ncpus = parallel::detectCores())" || exit -1

sh build-cran-package.sh || exit -1

log_file="rhub_logs.txt"
Rscript ./.ci/run_rhub_solaris_checks.R lightgbm_*.tar.gz $log_file || exit -1
