context("lightgbm()")

data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
train <- agaricus.train
test <- agaricus.test

TOLERANCE <- 1e-6
set.seed(708L)

# [description] Every time this function is called, it adds 0.1
#               to an accumulator then returns the current value.
#               This is used to mock the situation where an evaluation
#               metric increases every iteration
ACCUMULATOR_NAME <- "INCREASING_METRIC_ACUMULATOR"
assign(ACCUMULATOR_NAME, 0.0, envir = .GlobalEnv)
.increasing_metric <- function(preds, dtrain) {
  if (!exists(ACCUMULATOR_NAME, envir = .GlobalEnv)) {
    assign(ACCUMULATOR_NAME, 0.0, envir = .GlobalEnv)
  }
  assign(
    x = ACCUMULATOR_NAME
    , value = get(ACCUMULATOR_NAME, envir = .GlobalEnv) + 0.1
    , envir = .GlobalEnv
  )
  return(list(
    name = "increasing_metric"
    , value = get(ACCUMULATOR_NAME, envir = .GlobalEnv)
    , higher_better = TRUE
  ))
}

# [description] Evaluation function that always returns the
#               same value
CONSTANT_METRIC_VALUE <- 0.2
.constant_metric <- function(preds, dtrain) {
  return(list(
    name = "constant_metric"
    , value = CONSTANT_METRIC_VALUE
    , higher_better = FALSE
  ))
}

# sample datasets to test early stopping
DTRAIN_RANDOM_REGRESSION <- lgb.Dataset(
  data = as.matrix(rnorm(100L), ncol = 1L, drop = FALSE)
  , label = rnorm(100L)
)
DVALID_RANDOM_REGRESSION <- lgb.Dataset(
  data = as.matrix(rnorm(50L), ncol = 1L, drop = FALSE)
  , label = rnorm(50L)
)
DTRAIN_RANDOM_CLASSIFICATION <- lgb.Dataset(
  data = as.matrix(rnorm(120L), ncol = 1L, drop = FALSE)
  , label = sample(c(0L, 1L), size = 120L, replace = TRUE)
)
DVALID_RANDOM_CLASSIFICATION <- lgb.Dataset(
  data = as.matrix(rnorm(37L), ncol = 1L, drop = FALSE)
  , label = sample(c(0L, 1L), size = 37L, replace = TRUE)
)

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
  expect_lt(abs(err_pred1 - err_log), TOLERANCE)
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
  metrics <- list("binary_error", "auc", "binary_logloss")
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , num_leaves = 4L
    , learning_rate = 1.0
    , nrounds = 10L
    , objective = "binary"
    , metric = metrics
  )
  expect_false(is.null(bst$record_evals))
  expect_named(
    bst$record_evals[["train"]]
    , unlist(metrics)
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
})

test_that("lgb.Booster.upper_bound() and lgb.Booster.lower_bound() work as expected for binary classification", {
  set.seed(708L)
  nrounds <- 10L
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , num_leaves = 5L
    , nrounds = nrounds
    , objective = "binary"
    , metric = "binary_error"
  )
  expect_true(abs(bst$lower_bound() - -1.590853) < TOLERANCE)
  expect_true(abs(bst$upper_bound() - 1.871015) <  TOLERANCE)
})

test_that("lgb.Booster.upper_bound() and lgb.Booster.lower_bound() work as expected for regression", {
  set.seed(708L)
  nrounds <- 10L
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , num_leaves = 5L
    , nrounds = nrounds
    , objective = "regression"
    , metric = "l2"
  )
  expect_true(abs(bst$lower_bound() - 0.1513859) < TOLERANCE)
  expect_true(abs(bst$upper_bound() - 0.9080349) < TOLERANCE)
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

test_that("lightgbm() performs evaluation on validation sets if they are provided", {
  set.seed(708L)
  dvalid1 <- lgb.Dataset(
    data = train$data
    , labels = train$label
  )
  dvalid2 <- lgb.Dataset(
    data = train$data
    , labels = train$label
  )
  nrounds <- 10L
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , num_leaves = 5L
    , nrounds = nrounds
    , objective = "binary"
    , metric = "binary_error"
    , valids = list(
      "valid1" = dvalid1
      , "valid2" = dvalid2
    )
  )

  expect_named(
    bst$record_evals
    , c("train", "valid1", "valid2", "start_iter")
    , ignore.order = TRUE
    , ignore.case = FALSE
  )
  for (valid_name in c("train", "valid1", "valid2")) {
    eval_results <- bst$record_evals[[valid_name]][["binary_error"]]
    expect_length(eval_results[["eval"]], nrounds)
  }
  expect_true(abs(bst$record_evals[["train"]][["binary_error"]][["eval"]][[1L]] - 0.02226317) < TOLERANCE)
  expect_true(abs(bst$record_evals[["valid1"]][["binary_error"]][["eval"]][[1L]] - 0.4825733) < TOLERANCE)
  expect_true(abs(bst$record_evals[["valid2"]][["binary_error"]][["eval"]][[1L]] - 0.4825733) < TOLERANCE)
})


context("training continuation")

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

test_that("lgb.train() works as expected with multiple eval metrics", {
  metrics <- c("binary_error", "auc", "binary_logloss")
  bst <- lgb.train(
    data = lgb.Dataset(
      train$data
      , label = train$label
    )
    , learning_rate = 1.0
    , nrounds = 10L
    , params = list(
      objective = "binary"
      , metric = metrics
    )
    , valids = list(
      "train" = lgb.Dataset(
        train$data
        , label = train$label
      )
    )
  )
  expect_false(is.null(bst$record_evals))
  expect_named(
    bst$record_evals[["train"]]
    , unlist(metrics)
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
})

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

test_that("lgb.train() works with force_col_wise and force_row_wise", {
  set.seed(1234L)
  nrounds <- 10L
  dtrain <- lgb.Dataset(
    train$data
    , label = train$label
  )
  params <- list(
    objective = "binary"
    , metric = "binary_error"
    , force_col_wise = TRUE
  )
  bst_col_wise <- lgb.train(
    params = params
    , data = dtrain
    , nrounds = nrounds
  )

  params <- list(
    objective = "binary"
    , metric = "binary_error"
    , force_row_wise = TRUE
  )
  bst_row_wise <- lgb.train(
    params = params
    , data = dtrain
    , nrounds = nrounds
  )

  expected_error <- 0.003070782
  expect_equal(bst_col_wise$eval_train()[[1L]][["value"]], expected_error)
  expect_equal(bst_row_wise$eval_train()[[1L]][["value"]], expected_error)

  # check some basic details of the boosters just to be sure force_col_wise
  # and force_row_wise are not causing any weird side effects
  for (bst in list(bst_row_wise, bst_col_wise)) {
    expect_equal(bst$current_iter(), nrounds)
    parsed_model <- jsonlite::fromJSON(bst$dump_model())
    expect_equal(parsed_model$objective, "binary sigmoid:1")
    expect_false(parsed_model$average_output)
  }
})

test_that("lgb.train() works as expected with sparse features", {
  set.seed(708L)
  num_obs <- 70000L
  trainDF <- data.frame(
    y = sample(c(0L, 1L), size = num_obs, replace = TRUE)
    , x = sample(c(1.0:10.0, rep(NA_real_, 50L)), size = num_obs, replace = TRUE)
  )
  dtrain <- lgb.Dataset(
    data = as.matrix(trainDF[["x"]], drop = FALSE)
    , label = trainDF[["y"]]
  )
  nrounds <- 1L
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , min_data = 1L
      , min_data_in_bin = 1L
    )
    , data = dtrain
    , nrounds = nrounds
  )

  expect_true(lgb.is.Booster(bst))
  expect_equal(bst$current_iter(), nrounds)
  parsed_model <- jsonlite::fromJSON(bst$dump_model())
  expect_equal(parsed_model$objective, "binary sigmoid:1")
  expect_false(parsed_model$average_output)
  expected_error <- 0.6931268
  expect_true(abs(bst$eval_train()[[1L]][["value"]] - expected_error) < TOLERANCE)
})

test_that("lgb.train() works with early stopping for classification", {
  trainDF <- data.frame(
    "feat1" = rep(c(5.0, 10.0), 500L)
    , "target" = rep(c(0L, 1L), 500L)
  )
  validDF <- data.frame(
    "feat1" = rep(c(5.0, 10.0), 50L)
    , "target" = rep(c(0L, 1L), 50L)
  )
  dtrain <- lgb.Dataset(
    data = as.matrix(trainDF[["feat1"]], drop = FALSE)
    , label = trainDF[["target"]]
  )
  dvalid <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
  )
  nrounds <- 10L

  ################################
  # train with no early stopping #
  ################################
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "binary_error"
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )

  # a perfect model should be trivial to obtain, but all 10 rounds
  # should happen
  expect_equal(bst$best_score, 0.0)
  expect_equal(bst$best_iter, 1L)
  expect_equal(length(bst$record_evals[["valid1"]][["binary_error"]][["eval"]]), nrounds)

  #############################
  # train with early stopping #
  #############################
  early_stopping_rounds <- 5L
  bst  <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "binary_error"
      , early_stopping_rounds = early_stopping_rounds
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )

  # a perfect model should be trivial to obtain, and only 6 rounds
  # should have happen (1 with improvement, 5 consecutive with no improvement)
  expect_equal(bst$best_score, 0.0)
  expect_equal(bst$best_iter, 1L)
  expect_equal(
    length(bst$record_evals[["valid1"]][["binary_error"]][["eval"]])
    , early_stopping_rounds + 1L
  )

})

test_that("lgb.train() works with early stopping for regression", {
  set.seed(708L)
  trainDF <- data.frame(
    "feat1" = rep(c(10.0, 100.0), 500L)
    , "target" = rep(c(-50.0, 50.0), 500L)
  )
  validDF <- data.frame(
    "feat1" = rep(50.0, 4L)
    , "target" = rep(50.0, 4L)
  )
  dtrain <- lgb.Dataset(
    data = as.matrix(trainDF[["feat1"]], drop = FALSE)
    , label = trainDF[["target"]]
  )
  dvalid <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
  )
  nrounds <- 10L

  ################################
  # train with no early stopping #
  ################################
  bst <- lgb.train(
    params = list(
      objective = "regression"
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )

  # the best possible model should come from the first iteration, but
  # all 10 training iterations should happen
  expect_equal(bst$best_score, 55.0)
  expect_equal(bst$best_iter, 1L)
  expect_equal(length(bst$record_evals[["valid1"]][["rmse"]][["eval"]]), nrounds)

  #############################
  # train with early stopping #
  #############################
  early_stopping_rounds <- 5L
  bst  <- lgb.train(
    params = list(
      objective = "regression"
      , metric = "rmse"
      , min_data_in_bin = 5L
      , early_stopping_rounds = early_stopping_rounds
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )

  # the best model should be from the first iteration, and only 6 rounds
  # should have happen (1 with improvement, 5 consecutive with no improvement)
  expect_equal(bst$best_score, 55.0)
  expect_equal(bst$best_iter, 1L)
  expect_equal(
    length(bst$record_evals[["valid1"]][["rmse"]][["eval"]])
    , early_stopping_rounds + 1L
  )
})

test_that("lgb.train() does not stop early if early_stopping_rounds is not given", {
  set.seed(708L)

  increasing_metric_starting_value <- get(
    ACCUMULATOR_NAME
    , envir = .GlobalEnv
  )
  nrounds <- 10L
  metrics <- list(
    .constant_metric
    , .increasing_metric
  )
  bst <- lgb.train(
    params = list(
      objective = "regression"
      , metric = "None"
    )
    , data = DTRAIN_RANDOM_REGRESSION
    , nrounds = nrounds
    , valids = list("valid1" = DVALID_RANDOM_REGRESSION)
    , eval = metrics
  )

  # Only the two functions provided to "eval" should have been evaluated
  expect_equal(length(bst$record_evals[["valid1"]]), 2L)

  # all 10 iterations should have happen, and the best_iter should be
  # the first one (based on constant_metric)
  best_iter <- 1L
  expect_equal(bst$best_iter, best_iter)

  # best_score should be taken from the first metric
  expect_equal(
    bst$best_score
    , bst$record_evals[["valid1"]][["constant_metric"]][["eval"]][[best_iter]]
  )

  # early stopping should not have happened. Even though constant_metric
  # had 9 consecutive iterations with no improvement, it is ignored because of
  # first_metric_only = TRUE
  expect_equal(
    length(bst$record_evals[["valid1"]][["constant_metric"]][["eval"]])
    , nrounds
  )
  expect_equal(
    length(bst$record_evals[["valid1"]][["increasing_metric"]][["eval"]])
    , nrounds
  )
})

test_that("If first_metric_only is not given or is FALSE, lgb.train() decides to stop early based on all metrics", {
  set.seed(708L)

  early_stopping_rounds <- 3L
  param_variations <- list(
    list(
      objective = "regression"
      , metric = "None"
      , early_stopping_rounds = early_stopping_rounds
    )
    , list(
      objective = "regression"
      , metric = "None"
      , early_stopping_rounds = early_stopping_rounds
      , first_metric_only = FALSE
    )
  )

  for (params in param_variations) {

    nrounds <- 10L
    bst <- lgb.train(
      params = params
      , data = DTRAIN_RANDOM_REGRESSION
      , nrounds = nrounds
      , valids = list(
        "valid1" = DVALID_RANDOM_REGRESSION
      )
      , eval = list(
        .increasing_metric
        , .constant_metric
      )
    )

    # Only the two functions provided to "eval" should have been evaluated
    expect_equal(length(bst$record_evals[["valid1"]]), 2L)

    # early stopping should have happened, and should have stopped early_stopping_rounds + 1 rounds in
    # because constant_metric never improves
    #
    # the best iteration should be the last one, because increasing_metric was first
    # and gets better every iteration
    best_iter <- early_stopping_rounds + 1L
    expect_equal(bst$best_iter, best_iter)

    # best_score should be taken from "increasing_metric" because it was first
    expect_equal(
      bst$best_score
      , bst$record_evals[["valid1"]][["increasing_metric"]][["eval"]][[best_iter]]
    )

    # early stopping should not have happened. even though increasing_metric kept
    # getting better, early stopping should have happened because "constant_metric"
    # did not improve
    expect_equal(
      length(bst$record_evals[["valid1"]][["constant_metric"]][["eval"]])
      , early_stopping_rounds + 1L
    )
    expect_equal(
      length(bst$record_evals[["valid1"]][["increasing_metric"]][["eval"]])
      , early_stopping_rounds + 1L
    )
  }

})

test_that("If first_metric_only is TRUE, lgb.train() decides to stop early based on only the first metric", {
  set.seed(708L)
  nrounds <- 10L
  early_stopping_rounds <- 3L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.train(
    params = list(
      objective = "regression"
      , metric = "None"
      , early_stopping_rounds = early_stopping_rounds
      , first_metric_only = TRUE
    )
    , data = DTRAIN_RANDOM_REGRESSION
    , nrounds = nrounds
    , valids = list(
      "valid1" = DVALID_RANDOM_REGRESSION
    )
    , eval = list(
      .increasing_metric
      , .constant_metric
    )
  )

  # Only the two functions provided to "eval" should have been evaluated
  expect_equal(length(bst$record_evals[["valid1"]]), 2L)

  # all 10 iterations should happen, and the best_iter should be the final one
  expect_equal(bst$best_iter, nrounds)

  # best_score should be taken from "increasing_metric"
  expect_equal(
    bst$best_score
    , increasing_metric_starting_value + 0.1 * nrounds
  )

  # early stopping should not have happened. Even though constant_metric
  # had 9 consecutive iterations with no improvement, it is ignored because of
  # first_metric_only = TRUE
  expect_equal(
    length(bst$record_evals[["valid1"]][["constant_metric"]][["eval"]])
    , nrounds
  )
  expect_equal(
    length(bst$record_evals[["valid1"]][["increasing_metric"]][["eval"]])
    , nrounds
  )
})

test_that("lgb.train() works when a mixture of functions and strings are passed to eval", {
  set.seed(708L)
  nrounds <- 10L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.train(
    params = list(
      objective = "regression"
      , metric = "None"
    )
    , data = DTRAIN_RANDOM_REGRESSION
    , nrounds = nrounds
    , valids = list(
      "valid1" = DVALID_RANDOM_REGRESSION
    )
    , eval = list(
      .increasing_metric
      , "rmse"
      , .constant_metric
      , "l2"
    )
  )

  # all 4 metrics should have been used
  expect_named(
    bst$record_evals[["valid1"]]
    , expected = c("rmse", "l2", "increasing_metric", "constant_metric")
    , ignore.order = TRUE
    , ignore.case = FALSE
  )

  # the difference metrics shouldn't have been mixed up with each other
  results <- bst$record_evals[["valid1"]]
  expect_true(abs(results[["rmse"]][["eval"]][[1L]] - 1.105012) < TOLERANCE)
  expect_true(abs(results[["l2"]][["eval"]][[1L]] - 1.221051) < TOLERANCE)
  expected_increasing_metric <- increasing_metric_starting_value + 0.1
  expect_true(
    abs(
      results[["increasing_metric"]][["eval"]][[1L]] - expected_increasing_metric
    ) < TOLERANCE
  )
  expect_true(abs(results[["constant_metric"]][["eval"]][[1L]] - CONSTANT_METRIC_VALUE) < TOLERANCE)

})

test_that("lgb.train() works when a list of strings or a character vector is passed to eval", {

  # testing list and character vector, as well as length-1 and length-2
  eval_variations <- list(
    c("binary_error", "binary_logloss")
    , "binary_logloss"
    , list("binary_error", "binary_logloss")
    , list("binary_logloss")
  )

  for (eval_variation in eval_variations) {

    set.seed(708L)
    nrounds <- 10L
    increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
    bst <- lgb.train(
      params = list(
        objective = "binary"
        , metric = "None"
      )
      , data = DTRAIN_RANDOM_CLASSIFICATION
      , nrounds = nrounds
      , valids = list(
        "valid1" = DVALID_RANDOM_CLASSIFICATION
      )
      , eval = eval_variation
    )

    # both metrics should have been used
    expect_named(
      bst$record_evals[["valid1"]]
      , expected = unlist(eval_variation)
      , ignore.order = TRUE
      , ignore.case = FALSE
    )

    # the difference metrics shouldn't have been mixed up with each other
    results <- bst$record_evals[["valid1"]]
    if ("binary_error" %in% unlist(eval_variation)) {
      expect_true(abs(results[["binary_error"]][["eval"]][[1L]] - 0.4864865) < TOLERANCE)
    }
    if ("binary_logloss" %in% unlist(eval_variation)) {
      expect_true(abs(results[["binary_logloss"]][["eval"]][[1L]] - 0.6932548) < TOLERANCE)
    }
  }
})

test_that("lgb.train() works when you specify both 'metric' and 'eval' with strings", {
  set.seed(708L)
  nrounds <- 10L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "binary_error"
    )
    , data = DTRAIN_RANDOM_CLASSIFICATION
    , nrounds = nrounds
    , valids = list(
      "valid1" = DVALID_RANDOM_CLASSIFICATION
    )
    , eval = "binary_logloss"
  )

  # both metrics should have been used
  expect_named(
    bst$record_evals[["valid1"]]
    , expected = c("binary_error", "binary_logloss")
    , ignore.order = TRUE
    , ignore.case = FALSE
  )

  # the difference metrics shouldn't have been mixed up with each other
  results <- bst$record_evals[["valid1"]]
  expect_true(abs(results[["binary_error"]][["eval"]][[1L]] - 0.4864865) < TOLERANCE)
  expect_true(abs(results[["binary_logloss"]][["eval"]][[1L]] - 0.6932548) < TOLERANCE)
})

test_that("lgb.train() works when you specify both 'metric' and 'eval' with strings", {
  set.seed(708L)
  nrounds <- 10L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "binary_error"
    )
    , data = DTRAIN_RANDOM_CLASSIFICATION
    , nrounds = nrounds
    , valids = list(
      "valid1" = DVALID_RANDOM_CLASSIFICATION
    )
    , eval = "binary_logloss"
  )

  # both metrics should have been used
  expect_named(
    bst$record_evals[["valid1"]]
    , expected = c("binary_error", "binary_logloss")
    , ignore.order = TRUE
    , ignore.case = FALSE
  )

  # the difference metrics shouldn't have been mixed up with each other
  results <- bst$record_evals[["valid1"]]
  expect_true(abs(results[["binary_error"]][["eval"]][[1L]] - 0.4864865) < TOLERANCE)
  expect_true(abs(results[["binary_logloss"]][["eval"]][[1L]] - 0.6932548) < TOLERANCE)
})

test_that("lgb.train() works when you give a function for eval", {
  set.seed(708L)
  nrounds <- 10L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "None"
    )
    , data = DTRAIN_RANDOM_CLASSIFICATION
    , nrounds = nrounds
    , valids = list(
      "valid1" = DVALID_RANDOM_CLASSIFICATION
    )
    , eval = .constant_metric
  )

  # the difference metrics shouldn't have been mixed up with each other
  results <- bst$record_evals[["valid1"]]
  expect_true(abs(results[["constant_metric"]][["eval"]][[1L]] - CONSTANT_METRIC_VALUE) < TOLERANCE)
})

test_that("lgb.train() supports non-ASCII feature names", {
  testthat::skip("UTF-8 feature names are not fully supported in the R package")
  dtrain <- lgb.Dataset(
    data = matrix(rnorm(400L), ncol =  4L)
    , label = rnorm(100L)
  )
  feature_names <- c("F_零", "F_一", "F_二", "F_三")
  bst <- lgb.train(
    data = dtrain
    , nrounds = 5L
    , obj = "regression"
    , params = list(
      metric = "rmse"
    )
    , colnames = feature_names
  )
  expect_true(lgb.is.Booster(bst))
  dumped_model <- jsonlite::fromJSON(bst$dump_model())
  expect_identical(
    dumped_model[["feature_names"]]
    , feature_names
  )
})
