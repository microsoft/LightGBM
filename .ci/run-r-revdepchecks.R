options(repos = "https://cran.r-project.org")

check_dir <- commandArgs(trailing = TRUE)[[1L]]

tools::check_packages_in_dir(
    dir = check_dir
    , check_args = c("--as-cran")
    , Ncpus = 1
    , clean = TRUE
    , all = TRUE
    , reverse = list(
        which = "most"
        , recursive = FALSE
    )
)

all_checks_passed <- tools::summarize_check_packages_in_dir_results(
    dir = "/tmp/lgb-revdepchecks"
    , all = TRUE
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
