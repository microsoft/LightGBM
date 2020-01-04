context("lightgbm()")

data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
train <- agaricus.train
test <- agaricus.test

windows_flag <- grepl("Windows", Sys.info()[["sysname"]])

test_that("train and predict binary classification", {
  nrounds <- 10L
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , num_leaves = 5L
    , nrounds = nrounds
    , objective = "binary"
    , metric = "binary_error"
  )
  expect_false(is.null(bst$record_evals))
  record_results <- lgb.get.eval.result(bst, "train", "binary_error")
  expect_lt(min(record_results), 0.02)

  pred <- predict(bst, test$data)
  expect_equal(length(pred), 1611L)

  pred1 <- predict(bst, train$data, num_iteration = 1L)
  expect_equal(length(pred1), 6513L)
  err_pred1 <- sum((pred1 > 0.5) != train$label) / length(train$label)
  err_log <- record_results[1L]
  expect_lt(abs(err_pred1 - err_log), 10e-6)
})


test_that("train and predict softmax", {
  lb <- as.numeric(iris$Species) - 1L

  bst <- lightgbm(
    data = as.matrix(iris[, -5L])
    , label = lb
    , num_leaves = 4L
    , learning_rate = 0.1
    , nrounds = 20L
    , min_data = 20L
    , min_hess = 20.0
    , objective = "multiclass"
    , metric = "multi_error"
    , num_class = 3L
  )

  expect_false(is.null(bst$record_evals))
  record_results <- lgb.get.eval.result(bst, "train", "multi_error")
  expect_lt(min(record_results), 0.03)

  pred <- predict(bst, as.matrix(iris[, -5L]))
  expect_equal(length(pred), nrow(iris) * 3L)
})


test_that("use of multiple eval metrics works", {
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , num_leaves = 4L
    , learning_rate = 1.0
    , nrounds = 10L
    , objective = "binary"
    , metric = list("binary_error", "auc", "binary_logloss")
  )
  expect_false(is.null(bst$record_evals))
})

test_that("lightgbm() rejects negative or 0 value passed to nrounds", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", metric = "l2,l1")
  for (nround_value in c(-10L, 0L)) {
    expect_error({
      bst <- lightgbm(
        data = dtrain
        , params = params
        , nrounds = nround_value
      )
    }, "nrounds should be greater than zero")
  }
})


test_that("training continuation works", {
  testthat::skip("This test is currently broken. See issue #2468 for details.")
  dtrain <- lgb.Dataset(
    train$data
    , label = train$label
    , free_raw_data = FALSE
  )
  watchlist <- list(train = dtrain)
  param <- list(
    objective = "binary"
    , metric = "binary_logloss"
    , num_leaves = 5L
    , learning_rate = 1.0
  )

  # for the reference, use 10 iterations at once:
  bst <- lgb.train(param, dtrain, nrounds = 10L, watchlist)
  err_bst <- lgb.get.eval.result(bst, "train", "binary_logloss", 10L)
  # first 5 iterations:
  bst1 <- lgb.train(param, dtrain, nrounds = 5L, watchlist)
  # test continuing from a model in file
  lgb.save(bst1, "lightgbm.model")
  # continue for 5 more:
  bst2 <- lgb.train(param, dtrain, nrounds = 5L, watchlist, init_model = bst1)
  err_bst2 <- lgb.get.eval.result(bst2, "train", "binary_logloss", 10L)
  expect_lt(abs(err_bst - err_bst2), 0.01)

  bst2 <- lgb.train(param, dtrain, nrounds = 5L, watchlist, init_model = "lightgbm.model")
  err_bst2 <- lgb.get.eval.result(bst2, "train", "binary_logloss", 10L)
  expect_lt(abs(err_bst - err_bst2), 0.01)
})

context("lgb.cv()")

test_that("cv works", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", metric = "l2,l1")
  bst <- lgb.cv(
    params
    , dtrain
    , 10L
    , nfold = 5L
    , min_data = 1L
    , learning_rate = 1.0
    , early_stopping_rounds = 10L
  )
  expect_false(is.null(bst$record_evals))
})

test_that("lgb.cv() rejects negative or 0 value passed to nrounds", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", metric = "l2,l1")
  for (nround_value in c(-10L, 0L)) {
    expect_error({
      bst <- lgb.cv(
        params
        , dtrain
        , nround_value
        , nfold = 5L
        , min_data = 1L
      )
    }, "nrounds should be greater than zero")
  }
})

test_that("lgb.cv() throws an informative error is 'data' is not an lgb.Dataset and labels are not given", {
  bad_values <- list(
    4L
    , "hello"
    , list(a = TRUE, b = seq_len(10L))
    , data.frame(x = seq_len(5L), y = seq_len(5L))
    , data.table::data.table(x = seq_len(5L),  y = seq_len(5L))
    , matrix(data = seq_len(10L), 2L, 5L)
  )
  for (val in bad_values) {
    expect_error({
      bst <- lgb.cv(
        params = list(objective = "regression", metric = "l2,l1")
        , data = val
        , 10L
        , nfold = 5L
        , min_data = 1L
      )
    }, regexp = "'label' must be provided for lgb.cv if 'data' is not an 'lgb.Dataset'", fixed = TRUE)
  }
})

context("lgb.train()")

test_that("lgb.train() rejects negative or 0 value passed to nrounds", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", metric = "l2,l1")
  for (nround_value in c(-10L, 0L)) {
    expect_error({
      bst <- lgb.train(
        params
        , dtrain
        , nround_value
      )
    }, "nrounds should be greater than zero")
  }
})

test_that("lgb.train() throws an informative error if 'data' is not an lgb.Dataset", {
  bad_values <- list(
    4L
    , "hello"
    , list(a = TRUE, b = seq_len(10L))
    , data.frame(x = seq_len(5L), y = seq_len(5L))
    , data.table::data.table(x = seq_len(5L),  y = seq_len(5L))
    , matrix(data = seq_len(10L), 2L, 5L)
  )
  for (val in bad_values) {
    expect_error({
      bst <- lgb.train(
        params = list(objective = "regression", metric = "l2,l1")
        , data = val
        , 10L
      )
    }, regexp = "data must be an lgb.Dataset instance", fixed = TRUE)
  }
})

test_that("lgb.train() throws an informative error if 'valids' is not a list of lgb.Dataset objects", {
  valids <- list(
    "valid1" = data.frame(x = rnorm(5L), y = rnorm(5L))
    , "valid2" = data.frame(x = rnorm(5L), y = rnorm(5L))
  )
  expect_error({
    bst <- lgb.train(
      params = list(objective = "regression", metric = "l2,l1")
      , data = lgb.Dataset(train$data, label = train$label)
      , 10L
      , valids = valids
    )
  }, regexp = "valids must be a list of lgb.Dataset elements")
})

test_that("lgb.train() errors if 'valids' is a list of lgb.Dataset objects but some do not have names", {
  valids <- list(
    "valid1" = lgb.Dataset(matrix(rnorm(10L), 5L, 2L))
    , lgb.Dataset(matrix(rnorm(10L), 2L, 5L))
  )
  expect_error({
    bst <- lgb.train(
      params = list(objective = "regression", metric = "l2,l1")
      , data = lgb.Dataset(train$data, label = train$label)
      , 10L
      , valids = valids
    )
  }, regexp = "each element of valids must have a name")
})

test_that("lgb.train() throws an informative error if 'valids' contains lgb.Dataset objects but none have names", {
  valids <- list(
    lgb.Dataset(matrix(rnorm(10L), 5L, 2L))
    , lgb.Dataset(matrix(rnorm(10L), 2L, 5L))
  )
  expect_error({
    bst <- lgb.train(
      params = list(objective = "regression", metric = "l2,l1")
      , data = lgb.Dataset(train$data, label = train$label)
      , 10L
      , valids = valids
    )
  }, regexp = "each element of valids must have a name")
})
