library(testthat)
library(lightgbm)

data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
train <- agaricus.train
test <- agaricus.test

test_check(
    package = "lightgbm"
    , stop_on_failure = TRUE
    , stop_on_warning = FALSE
)
