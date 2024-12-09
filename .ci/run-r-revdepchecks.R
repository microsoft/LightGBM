options(repos = "https://cran.r-project.org")

check_dir <- commandArgs(trailing = TRUE)[[1L]]

tools::check_packages_in_dir(
    dir = check_dir
    , check_args = c("--run-dontrun", "--run-donttest")
    , clean = TRUE
    , all = TRUE
    # only check one package at a time, to avoid oversubscribing CPUs
    , Ncpus = 1L
    # only test the libraries found in `check_dir`
    , reverse = FALSE
)

all_checks_passed <- tools::summarize_check_packages_in_dir_results(
    dir = check_dir
    , all = TRUE
)

if (!isTRUE(all_checks_passed)) {
    invisible(
        tools::summarize_check_packages_in_dir_results(
            dir = check_dir
            , all = TRUE
            , full = TRUE
        )
    )
    stop("Some checks failed! See results above.")
}
