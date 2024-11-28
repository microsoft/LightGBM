loadNamespace("crandep")

.log <- function(msg){
    cat(sprintf("[download-revdeps] %s", msg))
}

PKG_DIR <- "/tmp/packages"

# get all of lightgbm's reverse dependencies
depDF <- crandep::get_dep(
    name = "lightgbm"
    , type = "all"
    , reverse = TRUE
)
reverse_deps <- depDF[["to"]]
.log(sprintf("found %i reverse deps:", length(reverse_deps)))
.log(toString(reverse_deps))

# get the superset of all their dependencies
# (all of the packages needed to run 'R CMD check' on lightgbm's reverse deps)
their_deps <- unlist(
    unname(
        sapply(
            X = reverse_deps
            , FUN = function(pkg){
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

# don't try to install 'lightgbm', or packages expected to ship with the R standard library
all_deps <- all_deps[!all_deps %in% c("grid", "methods", "lightgbm", "parallel", "stats", "utils")]

.log(sprintf("packages required to run these checks: %i", length(all_deps)))

.log("installing all packages required to check reverse dependencies")

# install only the strong dependencies of all those packages
install.packages(
    pkgs = all_deps
    , repos = "https://cran.r-project.org"
    , dependencies = c("Depends", "Imports", "LinkingTo")
)

# remove 'lightgbm' and its direct reverse dependencies
.log("removing 'lightgbm' and its reverse dependencies")
remove.packages(
    pkgs = c("lightgbm", reverse_deps)
)

print(sprintf("--- downloading reverse dependencies (%i)", length(reverse_deps)))

download.packages(
    pkgs = reverse_deps
    , destdir = PKG_DIR
    , repos = "https://cran.r-project.org"
)
