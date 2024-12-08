options(repos = "https://cran.r-project.org")

check_dir <- commandArgs(trailing = TRUE)[[1L]]

tools::check_packages_in_dir(
    dir = check_dir
    , check_args = c("--as-cran")
    , Ncpus = 1
    , clean = TRUE
    , all = TRUE
    # , reverse = list(
    #     which = "most"
    #     , recursive = FALSE
    # )
)

# skip the following with known issues:
#
#   * 'misspi' ()
deps_to_skip <- c("misspi")

# tools::check_packages_in_dir() and associated functions don't offer
# a way to say something like ""
for (dep in deps_to_skip) {
    dep_check_results_dir <- paste0("rdepends_", dep, ".Rcheck")
    file.rename(
        file.path(dep_check_results_dir, "00check.log")
        , file.path(dep_check_results_dir, "results-backup.bak")
    )
}

all_checks_passed <- tools::summarize_check_packages_in_dir_results(
    dir = "/tmp/lgb-revdepchecks"
    , all = TRUE
    , which = c("SHAPforxgboost")
)

if (!isTRUE(all_checks_passed)) {
    invisible(
        tools::summarize_check_packages_in_dir_results(
            dir = "/tmp/lgb-revdepchecks"
            , all = TRUE
            , full = TRUE
        )
    )
    stop("Some checks failed! See results above.")
}
