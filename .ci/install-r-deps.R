args <- commandArgs(trailingOnly = TRUE)

# which dependencies to install
ALL_DEPS     <- "--all" %in% args
BUILD_DEPS   <- ALL_DEPS || ("--build" %in% args)
ROXYGEN_DEPS <- ALL_DEPS || ("--roxygen" %in% args)
TEST_DEPS    <- ALL_DEPS || ("--test" %in% args)

# force downloading of binary packages on macOS
COMPILE_FROM_SOURCE <- "both"
PACKAGE_TYPE <- getOption("pkgType")

# CRAN has precompiled binaries for macOS and Windows... prefer those,
# for faster installation.
if (Sys.info()[["sysname"]] == "Darwin" || .Platform$OS.type == "windows") {
    COMPILE_FROM_SOURCE <- "never"
    PACKAGE_TYPE <- "binary"
}
options(
    install.packages.check.source = "no"
    , install.packages.compile.from.source = COMPILE_FROM_SOURCE
)

# always use the same CRAN mirror
CRAN_MIRROR <- Sys.getenv("CRAN_MIRROR", unset = "https://cran.r-project.org")

# we always want these
deps_to_install <- c(
    "data.table"
    , "jsonlite"
    , "Matrix"
    , "R6"
)

if (isTRUE(BUILD_DEPS)) {
    deps_to_install <- c(
        deps_to_install
        , "knitr"
        , "markdown"
    )
}

if (isTRUE(ROXYGEN_DEPS)) {
    deps_to_install <- c(
        deps_to_install
        , "roxygen"
    )
}

if (isTRUE(TEST_DEPS)) {
    deps_to_install <- c(
        deps_to_install
        , "RhpcBLASctl"
        , "testthat"
    )
}

# in some builds, {Matrix} is pre-installed to pin to an old version,
# so we don't want to overwrite that
if (requireNamespace("Matrix")) {
    deps_to_install <- setdiff(deps_to_install, "Matrix")
}

msg <- sprintf(
    "[install-r-deps] installing R packages: %s\n"
    , toString(sort(deps_to_install))
)
cat(msg)

install.packages(  # nolint[undesirable_function]
    pkgs = deps_to_install
    , dependencies = c("Depends", "Imports", "LinkingTo")
    , lib = Sys.getenv("R_LIB_PATH", unset = .libPaths()[[1L]])
    , repos = CRAN_MIRROR
    , type = PACKAGE_TYPE
    , Ncpus = parallel::detectCores()
)
