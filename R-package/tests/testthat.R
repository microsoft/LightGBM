library(testthat)
library(lightgbm)

test_check(
    package = "lightgbm"
    , stop_on_failure = TRUE
    , stop_on_warning = FALSE
)
