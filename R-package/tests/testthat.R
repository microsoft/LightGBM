library(testthat)
library(lightgbm)

options(
    "lightgbm.cran.testing.threads" = 2.0
)

test_check(
    package = "lightgbm"
    , stop_on_failure = TRUE
    , stop_on_warning = FALSE
    , reporter = testthat::SummaryReporter$new()
)
