build_docs <- function() {
    loadNamespace("pkgdown")
    loadNamespace("roxygen2")

    description_path <- file.path("R-package", "DESCRIPTION")
    description_contents <- readLines(description_path)
    on.exit(writeLines(description_contents, description_path), add = TRUE)

    lgb_version <- readLines("VERSION.txt", n = 1L)
    lgb_version <- gsub("rc", "-", lgb_version, fixed = TRUE)
    docs_description <- gsub(
        "~~VERSION~~"
        , lgb_version
        , description_contents
        , fixed = TRUE
    )
    docs_description <- gsub(
        "~~DATE~~"
        , as.character(Sys.Date())
        , docs_description
        , fixed = TRUE
    )
    if (any(grepl("~~(VERSION|DATE)~~", docs_description))) {
        stop("Failed to replace placeholders in R-package/DESCRIPTION")
    }
    writeLines(docs_description, description_path)

    roxygen2::roxygenize(
        "R-package/"
        , load = "installed"
    )

    pkgdown::build_site(
        "R-package/"
        , lazy = FALSE
        , install = FALSE
        , devel = FALSE
        , examples = TRUE
        , run_dont_run = TRUE
        , seed = 42L
        , preview = FALSE
        , new_process = TRUE
    )
}

build_docs()
