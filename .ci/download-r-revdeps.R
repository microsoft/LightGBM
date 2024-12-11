loadNamespace("crandep")

PKG_DIR <- commandArgs(trailing = TRUE)[[1L]]

.log <- function(msg) {
    cat(sprintf("[download-revdeps] %s\n", msg))
}

# get all of lightgbm's reverse dependencies
depDF <- crandep::get_dep(
    name = "lightgbm"
    , type = "all"
    , reverse = TRUE
)
reverse_deps <- depDF[["to"]]
.log(sprintf("found %i reverse deps", length(reverse_deps)))

# skip some dependencies with known issues:
#
#   * 'misspi' (https://github.com/microsoft/LightGBM/issues/6741)
deps_to_skip <- "misspi"
.log(sprintf("excluding %i reverse deps: %s", length(deps_to_skip), toString(deps_to_skip)))
reverse_deps <- reverse_deps[!reverse_deps %in% deps_to_skip]

.log(sprintf("checking the following packages: %s", toString(reverse_deps)))

# get the superset of all their dependencies
# (all of the packages needed to run 'R CMD check' on lightgbm's reverse deps)
their_deps <- unlist(
    unname(
        sapply(
            X = reverse_deps
            , FUN = function(pkg) {
                return(
                    crandep::get_dep(
                        name = pkg
                        , type = "all"
                        , reverse = FALSE
                    )[["to"]]
                )
            }
        )
    )
)

all_deps <- sort(unique(c(their_deps, reverse_deps)))

# don't try to install 'lightgbm', or packages that ship with the R standard library
all_deps <- all_deps[!all_deps %in% c("grid", "methods", "lightgbm", "parallel", "stats", "utils")]

.log(sprintf("packages required to run these checks: %i", length(all_deps)))

.log("installing all packages required to check reverse dependencies")

# install only the strong dependencies of all those packages
install.packages(  # nolint: undesirable_function
    pkgs = all_deps
    , repos = "https://cran.r-project.org"
    , dependencies = c("Depends", "Imports", "LinkingTo")
    , type = "both"
    , Ncpus = parallel::detectCores()
)

# get source tarballs, to be checked with 'R CMD check'
print(sprintf("--- downloading reverse dependencies to check (%i)", length(reverse_deps)))
download.packages(
    pkgs = reverse_deps
    , destdir = PKG_DIR
    , repos = "https://cran.r-project.org"
    , type = "source"
)
