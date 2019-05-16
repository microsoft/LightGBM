library(pkgdown)

setwd("lightgbm_r")
if (!dir.exists("docs")) {
  dir.create("docs")
}

devtools::document()
clean_site()
init_site()
build_home(preview = FALSE, quiet = FALSE)
build_reference(lazy = FALSE, document = FALSE,
                examples = TRUE, run_dont_run = FALSE,
                seed = 42, preview = FALSE)
# # to-do
# build_articles(preview = FALSE)
# build_tutorials(preview = FALSE)
# build_news(preview = FALSE)

# # doesn't work
# pkgdown::build_site(pkg = ".", examples = FALSE, document = TRUE,
#                     run_dont_run = TRUE, seed = 1014, lazy = FALSE,
#                     override = list(), preview = NA, new_process = FALSE)
