# [description]
#
#    Installs a pinned set of packages that worked together
#    as of the last R 3.6 release.
#

.install_packages <- function(packages) {
    install.packages(
        pkgs = c(
            paste("https://cran.r-project.org/src/contrib/Archive", packages, sep = "/")
            , "https://cran.r-project.org/src/contrib/praise_1.0.0.tar.gz"
        )
        , dependencies = FALSE
        , lib = Sys.getenv("R_LIBS")
        , repos = NULL
    )
}

# when confronted with a bunch of URLs like this, install.packages() sometimes
# struggles to determine install order... so install packages in batches here,
# starting from the root of the dependency graph and working up
print("---- group 1 ----")
.install_packages(c(
    "brio/brio_1.1.4.tar.gz"
    , "cli/cli_3.6.2.tar.gz"
    , "crayon/crayon_1.5.2.tar.gz"
    , "digest/digest_0.6.36.tar.gz"
    , "evaluate/evaluate_0.23.tar.gz"
    , "fansi/fansi_1.0.5.tar.gz"
    , "fs/fs_1.6.4.tar.gz"
    , "glue/glue_1.7.0.tar.gz"
    , "jsonlite/jsonlite_1.8.8.tar.gz"
    , "lattice/lattice_0.20-41.tar.gz"
    , "magrittr/magrittr_2.0.2.tar.gz"
    , "pkgconfig/pkgconfig_2.0.2.tar.gz"
    , "R6/R6_2.5.0.tar.gz"
    , "rlang/rlang_1.1.3.tar.gz"
    , "rprojroot/rprojroot_2.0.3.tar.gz"
    , "utf8/utf8_1.2.3.tar.gz"
    , "withr/withr_3.0.1.tar.gz"
))

print("---- group 2 ----")
.install_packages(c(
    "diffobj/diffobj_0.3.4.tar.gz"
))

print("---- group 3 ----")
.install_packages(c(
    "callr/callr_3.7.5.tar.gz"
    , "desc/desc_1.4.2.tar.gz"
    , "lifecycle/lifecycle_1.0.3.tar.gz"
    , "pkgbuild/pkgbuild_1.4.4.tar.gz"
    , "pkgload/pkgload_1.3.4.tar.gz"
    , "pillar/pillar_1.8.1.tar.gz"
    , "processx/processx_3.8.3.tar.gz"
    , "ps/ps_1.8.0.tar.gz"
    , "rematch2/rematch2_2.1.1.tar.gz"
    , "testthat/testthat_3.2.1.tar.gz"
    , "tibble/tibble_3.2.0.tar.gz"
    , "vctrs/vctrs_0.6.4.tar.gz"
    , "waldo/waldo_0.5.3.tar.gz"
))
