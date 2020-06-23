
context("feature penalties")

data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
train <- agaricus.train
test <- agaricus.test

test_that("Feature penalties work properly", {
  # Fit a series of models with varying penalty on most important variable
  var_name <- "odor=none"
  var_index <- which(train$data@Dimnames[[2L]] == var_name)

  bst <- lapply(seq(1.0, 0.0, by = -0.1), function(x) {
    feature_penalties <- rep(1.0, ncol(train$data))
    feature_penalties[var_index] <- x
    lightgbm(
      data = train$data
      , label = train$label
      , num_leaves = 5L
      , learning_rate = 0.05
      , nrounds = 5L
      , objective = "binary"
      , feature_penalty = paste0(feature_penalties, collapse = ",")
      , metric = "binary_error"
      , verbose = -1L
      , save_name = tempfile(fileext = ".model")
    )
  })

  var_gain <- lapply(bst, function(x) lgb.importance(x)[Feature == var_name, Gain])
  var_cover <- lapply(bst, function(x) lgb.importance(x)[Feature == var_name, Cover])
  var_freq <- lapply(bst, function(x) lgb.importance(x)[Feature == var_name, Frequency])

  # Ensure that feature gain, cover, and frequency decreases with stronger penalties
  expect_true(all(diff(unlist(var_gain)) <= 0.0))
  expect_true(all(diff(unlist(var_cover)) <= 0.0))
  expect_true(all(diff(unlist(var_freq)) <= 0.0))

  expect_lt(min(diff(unlist(var_gain))), 0.0)
  expect_lt(min(diff(unlist(var_cover))), 0.0)
  expect_lt(min(diff(unlist(var_freq))), 0.0)

  # Ensure that feature is not used when feature_penalty = 0
  expect_length(var_gain[[length(var_gain)]], 0L)
})

context("parameter aliases")

test_that(".PARAMETER_ALIASES() returns a named list of character vectors, where names are unique", {
  param_aliases <- .PARAMETER_ALIASES()
  expect_identical(class(param_aliases), "list")
  expect_true(is.character(names(param_aliases)))
  expect_true(is.character(param_aliases[["boosting"]]))
  expect_true(is.character(param_aliases[["early_stopping_round"]]))
  expect_true(is.character(param_aliases[["num_iterations"]]))
  expect_true(is.character(param_aliases[["pre_partition"]]))
  expect_true(length(names(param_aliases)) == length(param_aliases))
  expect_true(all(sapply(param_aliases, is.character)))
  expect_true(length(unique(names(param_aliases))) == length(param_aliases))
})

test_that("training should warn if you use 'dart' boosting, specified with 'boosting' or aliases", {
  for (boosting_param in .PARAMETER_ALIASES()[["boosting"]]) {
    expect_warning({
      result <- lightgbm(
        data = train$data
        , label = train$label
        , num_leaves = 5L
        , learning_rate = 0.05
        , nrounds = 5L
        , objective = "binary"
        , metric = "binary_error"
        , verbose = -1L
        , params = stats::setNames(
          object = "dart"
          , nm = boosting_param
        )
        , save_name = tempfile(fileext = ".model")
      )
    }, regexp = "Early stopping is not available in 'dart' mode")
  }
})
