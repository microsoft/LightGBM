data(agaricus.train, package='lightgbm')
data(agaricus.test, package='lightgbm')
train <- agaricus.train
test <- agaricus.test

test_that("Feature penalties work properly", {
  # Fit a series of models with varying penalty on most important variable
  var_name <- "odor=none"
  var_index <- which(train$data@Dimnames[[2]] == var_name)

  bst <- lapply(seq(1, 0, by = -0.1), function(x) {
    feature_penalties <- rep(1, ncol(train$data))
    feature_penalties[var_index] <- x
    lightgbm(
      data = train$data,
      label = train$label,
      num_leaves = 5,
      learning_rate = 0.05,
      nrounds = 20,
      objective = "binary",
      feature_penalty = paste0(feature_penalties, collapse = ","),
      metric="binary_error",
      verbose = -1
    )
  })

  var_gain <- lapply(bst, function(x) lgb.importance(x)[Feature == var_name, Gain])
  var_cover <- lapply(bst, function(x) lgb.importance(x)[Feature == var_name, Cover])
  var_freq <- lapply(bst, function(x) lgb.importance(x)[Feature == var_name, Frequency])

  # Ensure that feature gain, cover, and frequency decreases with stronger penalties
  expect_true(all(diff(unlist(var_gain)) <= 0))
  expect_true(all(diff(unlist(var_cover)) <= 0))
  expect_true(all(diff(unlist(var_freq)) <= 0))

  expect_lt(min(diff(unlist(var_gain))), 0)
  expect_lt(min(diff(unlist(var_cover))), 0)
  expect_lt(min(diff(unlist(var_freq))), 0)

  # Ensure that feature is not used when feature_penalty = 0
  expect_length(var_gain[[length(var_gain)]], 0)
})
