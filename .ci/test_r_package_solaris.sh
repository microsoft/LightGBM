#!/bin/bash

apt-get install --no-install-recommends -y \
  libcurl4-openssl-dev \
  libxml2-dev \
  libssl-dev

# installation of dependencies needs to happen before building the package,
# since `R CMD build` needs to install the package to build vignettes
Rscript -e "install.packages(c('R6', 'data.table', 'jsonlite', 'knitr', 'Matrix', 'RhpcBLASctl', 'rmarkdown', 'rhub', 'testthat'), dependencies = c('Depends', 'Imports', 'LinkingTo'), repos = 'https://cran.r-project.org', Ncpus = parallel::detectCores())" || exit -1

sh build-cran-package.sh || exit -1

log_file="rhub_logs.txt"
Rscript ./.ci/run_rhub_solaris_checks.R lightgbm_*.tar.gz $log_file || exit -1
