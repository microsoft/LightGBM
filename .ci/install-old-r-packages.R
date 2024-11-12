# [description]
#
#    Installs a pinned set of packages that worked together
#    as of the last R 3.6 release.
#

cran_archive <- "https://cran.r-project.org/src/contrib/Archive"

packages <- c(
    "brio/brio_1.1.4.tar.gz"
    , "cli/cli_3.6.2.tar.gz"
    , "diffobj/diffobj_0.3.4.tar.gz"
    , "evaluate/evaluate_0.23.tar.gz"
    , "glue/glue_1.7.0.tar.gz"
    , "lattice/lattice_0.20-41.tar.gz"
    , "lifecycle/lifecycle_1.0.3.tar.gz"
    , "pillar/pillar_1.8.1.tar.gz"
    , "R6/R6_2.5.0.tar.gz"
    , "rematch2/rematch2_2.1.1.tar.gz"
    , "rlang/rlang_1.1.3.tar.gz"
    , "testthat/testthat_3.2.1.tar.gz"
    , "tibble/tibble_3.2.0.tar.gz"
    , "waldo/waldo_0.5.3.tar.gz"
    , "vctrs/vctrs_0.6.4.tar.gz"
)

install.packages(
    pkgs = paste(cran_archive, packages, sep = "/")
    , repos = NULL
)
