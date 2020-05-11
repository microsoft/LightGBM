library(pkgdown)
library(roxygen2)

setwd("lightgbm_r")
if (!dir.exists("docs")) {
  dir.create("docs")
}

roxygen2::roxygenize(load = 'installed')
pkgdown::clean_site()
pkgdown::init_site()
pkgdown::build_home(preview = FALSE, quiet = FALSE)
pkgdown::build_reference(
    lazy = FALSE
    , document = FALSE
    , examples = TRUE
    , run_dont_run = FALSE
    , seed = 42L
    , preview = FALSE
)
