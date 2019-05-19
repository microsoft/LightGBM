library(pkgdown)

setwd("lightgbm_r")
if (!dir.exists("docs")) {
  dir.create("docs")
}

devtools::document()
clean_site()
init_site()
build_home(preview = FALSE, quiet = FALSE)
build_reference(lazy = FALSE,
                document = FALSE,
                examples = TRUE,
                run_dont_run = FALSE,
                seed = 42,
                preview = FALSE)
