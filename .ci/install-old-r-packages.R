# [description]
#
#    Installs a pinned set of packages that worked together
#    as of the last R 3.6 release.
#

.install_packages <- function(packages) {
    install.packages(  # nolint: undesirable_function
        pkgs = paste(  # nolint: paste
            "https://cran.r-project.org/src/contrib/Archive"
            , packages
            , sep = "/"
        )
        , dependencies = FALSE
        , lib = Sys.getenv("R_LIBS")
        , repos = NULL
    )
}

# when confronted with a bunch of URLs like this, install.packages() sometimes
# struggles to determine install order... so install packages in batches here,
# starting from the root of the dependency graph and working up

# there was only a single release of {praise}, so there is no contrib/Archive URL for it
install.packages(  # nolint: undesirable_function
    pkgs = "https://cran.r-project.org/src/contrib/praise_1.0.0.tar.gz"
    , dependencies = FALSE
    , lib = Sys.getenv("R_LIBS")
    , repos = NULL
)

.install_packages(c(
    "brio/brio_1.1.4.tar.gz"              # nolint: non_portable_path
    , "cli/cli_3.6.2.tar.gz"              # nolint: non_portable_path
    , "crayon/crayon_1.5.2.tar.gz"        # nolint: non_portable_path
    , "digest/digest_0.6.36.tar.gz"       # nolint: non_portable_path
    , "evaluate/evaluate_0.23.tar.gz"     # nolint: non_portable_path
    , "fansi/fansi_1.0.5.tar.gz"          # nolint: non_portable_path
    , "fs/fs_1.6.4.tar.gz"                # nolint: non_portable_path
    , "glue/glue_1.7.0.tar.gz"            # nolint: non_portable_path
    , "jsonlite/jsonlite_1.8.8.tar.gz"    # nolint: non_portable_path
    , "lattice/lattice_0.20-41.tar.gz"    # nolint: non_portable_path
    , "magrittr/magrittr_2.0.2.tar.gz"    # nolint: non_portable_path
    , "pkgconfig/pkgconfig_2.0.2.tar.gz"  # nolint: non_portable_path
    , "ps/ps_1.8.0.tar.gz"                # nolint: non_portable_path
    , "R6/R6_2.5.0.tar.gz"                # nolint: non_portable_path
    , "rlang/rlang_1.1.3.tar.gz"          # nolint: non_portable_path
    , "rprojroot/rprojroot_2.0.3.tar.gz"  # nolint: non_portable_path
    , "utf8/utf8_1.2.3.tar.gz"            # nolint: non_portable_path
    , "withr/withr_3.0.1.tar.gz"          # nolint: non_portable_path
))

.install_packages(c(
    "desc/desc_1.4.2.tar.gz"              # nolint: non_portable_path
    , "diffobj/diffobj_0.3.4.tar.gz"      # nolint: non_portable_path
    , "lifecycle/lifecycle_1.0.3.tar.gz"  # nolint: non_portable_path
    , "processx/processx_3.8.3.tar.gz"    # nolint: non_portable_path
))

.install_packages(c(
    "callr/callr_3.7.5.tar.gz"    # nolint: non_portable_path
    , "vctrs/vctrs_0.6.4.tar.gz"  # nolint: non_portable_path
))

.install_packages(c(
    "pillar/pillar_1.8.1.tar.gz"    # nolint: non_portable_path
    , "tibble/tibble_3.2.0.tar.gz"  # nolint: non_portable_path
))

.install_packages(c(
    "pkgbuild/pkgbuild_1.4.4.tar.gz"    # nolint: non_portable_path
    , "rematch2/rematch2_2.1.1.tar.gz"  # nolint: non_portable_path
    , "waldo/waldo_0.5.3.tar.gz"        # nolint: non_portable_path
))

.install_packages(c(
    "pkgload/pkgload_1.3.4.tar.gz"      # nolint: non_portable_path
    , "testthat/testthat_3.2.1.tar.gz"  # nolint: non_portable_path
))
