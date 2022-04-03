VERBOSITY <- as.integer(
  Sys.getenv("LIGHTGBM_TEST_VERBOSITY", "-1")
)

test_that("Predictor$finalize() should not fail", {
    X <- as.matrix(as.integer(iris[, "Species"]), ncol = 1L)
    y <- iris[["Sepal.Length"]]
    dtrain <- lgb.Dataset(X, label = y)
    bst <- lgb.train(
        data = dtrain
        , params = list(
            objective = "regression"
        )
        , verbose = VERBOSITY
        , nrounds = 3L
    )
    model_file <- tempfile(fileext = ".model")
    bst$save_model(filename = model_file)
    predictor <- Predictor$new(modelfile = model_file)

    expect_true(lgb.is.Predictor(predictor))

    expect_false(lgb.is.null.handle(predictor$.__enclos_env__$private$handle))

    predictor$finalize()
    expect_true(lgb.is.null.handle(predictor$.__enclos_env__$private$handle))

    # calling finalize() a second time shouldn't cause any issues
    predictor$finalize()
    expect_true(lgb.is.null.handle(predictor$.__enclos_env__$private$handle))
})

test_that("predictions do not fail for integer input", {
    X <- as.matrix(as.integer(iris[, "Species"]), ncol = 1L)
    y <- iris[["Sepal.Length"]]
    dtrain <- lgb.Dataset(X, label = y)
    fit <- lgb.train(
        data = dtrain
        , params = list(
            objective = "regression"
        )
        , verbose = VERBOSITY
        , nrounds = 3L
    )
    X_double <- X[c(1L, 51L, 101L), , drop = FALSE]
    X_integer <- X_double
    storage.mode(X_double) <- "double"
    pred_integer <- predict(fit, X_integer)
    pred_double <- predict(fit, X_double)
    expect_equal(pred_integer, pred_double)
})

test_that("start_iteration works correctly", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    train <- agaricus.train
    test <- agaricus.test
    dtrain <- lgb.Dataset(
        agaricus.train$data
        , label = agaricus.train$label
    )
    dtest <- lgb.Dataset.create.valid(
        dtrain
        , agaricus.test$data
        , label = agaricus.test$label
    )
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , params = list(
            num_leaves = 4L
            , learning_rate = 0.6
            , objective = "binary"
            , verbosity = VERBOSITY
        )
        , nrounds = 50L
        , valids = list("test" = dtest)
        , early_stopping_rounds = 2L
    )
    expect_true(lgb.is.Booster(bst))
    pred1 <- predict(bst, newdata = test$data, rawscore = TRUE)
    pred_contrib1 <- predict(bst, test$data, predcontrib = TRUE)
    pred2 <- rep(0.0, length(pred1))
    pred_contrib2 <- rep(0.0, length(pred2))
    step <- 11L
    end_iter <- 49L
    if (bst$best_iter != -1L) {
        end_iter <- bst$best_iter - 1L
    }
    start_iters <- seq(0L, end_iter, by = step)
    for (start_iter in start_iters) {
        n_iter <- min(c(end_iter - start_iter + 1L, step))
        inc_pred <- predict(bst, test$data
            , start_iteration = start_iter
            , num_iteration = n_iter
            , rawscore = TRUE
        )
        inc_pred_contrib <- bst$predict(test$data
            , start_iteration = start_iter
            , num_iteration = n_iter
            , predcontrib = TRUE
        )
        pred2 <- pred2 + inc_pred
        pred_contrib2 <- pred_contrib2 + inc_pred_contrib
    }
    expect_equal(pred2, pred1)
    expect_equal(pred_contrib2, pred_contrib1)

    pred_leaf1 <- predict(bst, test$data, predleaf = TRUE)
    pred_leaf2 <- predict(bst, test$data, start_iteration = 0L, num_iteration = end_iter + 1L, predleaf = TRUE)
    expect_equal(pred_leaf1, pred_leaf2)
})

test_that("predict() params should override keyword argument for raw-score predictions", {
  data(agaricus.train, package = "lightgbm")
  X <- agaricus.train$data
  y <- agaricus.train$label
  bst <- lgb.train(
    data = lgb.Dataset(
      data = X
      , label = y
      , params = list(
        verbosity = VERBOSITY
        , data_seed = 708L
        , min_data_in_bin = 5L
      )
    )
    , params = list(
      objective = "binary"
      , min_data_in_leaf = 1L
      , seed = 708L
    )
    , nrounds = 10L
    , verbose = VERBOSITY
  )

  # check that the predictions from predict.lgb.Booster() really look like raw score predictions
  preds_prob <- predict(bst, X)
  preds_raw_s3_keyword <- predict(bst, X, rawscore = TRUE)
  preds_prob_from_raw <- 1.0 / (1.0 + exp(-preds_raw_s3_keyword))
  expect_equal(preds_prob, preds_prob_from_raw, tolerance = 1e-6)
  accuracy <- sum(as.integer(preds_prob_from_raw > 0.5) == y) / length(y)
  expect_equal(accuracy, 1.0)

  # should get the same results from Booster$predict() method
  preds_raw_r6_keyword <- bst$predict(X, rawscore = TRUE)
  expect_equal(preds_raw_s3_keyword, preds_raw_r6_keyword)

  # using a parameter alias of predict_raw_score should result in raw scores being returned
  aliases <- .PARAMETER_ALIASES()[["predict_raw_score"]]
  expect_true(length(aliases) > 1L)
  for (rawscore_alias in aliases) {
    params <- as.list(
      stats::setNames(
        object = TRUE
        , nm = rawscore_alias
      )
    )
    preds_raw_s3_param <- predict(bst, X, params = params)
    preds_raw_r6_param <- bst$predict(X, params = params)
    expect_equal(preds_raw_s3_keyword, preds_raw_s3_param)
    expect_equal(preds_raw_s3_keyword, preds_raw_r6_param)
  }
})

test_that("predict() params should override keyword argument for leaf-index predictions", {
  data(mtcars)
  X <- as.matrix(mtcars[, which(names(mtcars) != "mpg")])
  y <- as.numeric(mtcars[, "mpg"])
  bst <- lgb.train(
    data = lgb.Dataset(
      data = X
      , label = y
      , params = list(
        min_data_in_bin = 1L
        , verbosity = VERBOSITY
        , data_seed = 708L
      )
    )
    , params = list(
      objective = "regression"
      , min_data_in_leaf = 1L
      , seed = 708L
    )
    , nrounds = 10L
    , verbose = VERBOSITY
  )

  # check that predictions really look like leaf index predictions
  preds_leaf_s3_keyword <- predict(bst, X, predleaf = TRUE)
  expect_true(is.matrix(preds_leaf_s3_keyword))
  expect_equal(dim(preds_leaf_s3_keyword), c(32L, 10L))
  expect_true(min(preds_leaf_s3_keyword) >= 0L)
  max_leaf_node_index <- max(
    model = lgb.model.dt.tree(bst)[["leaf_index"]]
    , na.rm = TRUE
  )
  expect_true(max(preds_leaf_s3_keyword) <= max_leaf_node_index)

  # should get the same results from Booster$predict() method
  preds_leaf_r6_keyword <- bst$predict(X, predleaf = TRUE)
  expect_equal(preds_leaf_s3_keyword, preds_leaf_r6_keyword)

  # using a parameter alias of predict_leaf_index should result in leaf indices being returned
  aliases <- .PARAMETER_ALIASES()[["predict_leaf_index"]]
  expect_true(length(aliases) > 1L)
  for (predleaf_alias in aliases) {
    params <- as.list(
      stats::setNames(
        object = TRUE
        , nm = predleaf_alias
      )
    )
    preds_leaf_s3_param <- predict(bst, X, params = params)
    preds_leaf_r6_param <- bst$predict(X, params = params)
    expect_equal(preds_leaf_s3_keyword, preds_leaf_s3_param)
    expect_equal(preds_leaf_s3_keyword, preds_leaf_r6_param)
  }
})

test_that("predict() params should override keyword argument for feature contributions", {
  data(mtcars)
  X <- as.matrix(mtcars[, which(names(mtcars) != "mpg")])
  y <- as.numeric(mtcars[, "mpg"])
  bst <- lgb.train(
    data = lgb.Dataset(
      data = X
      , label = y
      , params = list(
        min_data_in_bin = 1L
        , verbosity = VERBOSITY
        , data_seed = 708L
      )
    )
    , params = list(
      objective = "regression"
      , min_data_in_leaf = 1L
      , seed = 708L
    )
    , nrounds = 10L
    , verbose = VERBOSITY
  )

  # check that predictions really look like feature contributions
  preds_contrib_s3_keyword <- predict(bst, X, predcontrib = TRUE)
  num_features <- ncol(X)
  shap_base_value <- preds_contrib_s3_keyword[, ncol(preds_contrib_s3_keyword)]
  expect_true(is.matrix(preds_contrib_s3_keyword))
  expect_equal(dim(preds_contrib_s3_keyword), c(nrow(X), num_features + 1L))
  expect_equal(length(unique(shap_base_value)), 1L)
  expect_equal(mean(y), shap_base_value[1L])
  expect_equal(predict(bst, X), rowSums(preds_contrib_s3_keyword))

  # should get the same results from Booster$predict() method
  preds_contrib_r6_keyword <- bst$predict(X, predcontrib = TRUE)
  expect_equal(preds_contrib_s3_keyword, preds_contrib_r6_keyword)

  # using a parameter alias of predict_contrib should result in feature contributions being returned
  aliases <- .PARAMETER_ALIASES()[["predict_contrib"]]
  expect_true(length(aliases) > 1L)
  for (predcontrib_alias in aliases) {
    params <- as.list(
      stats::setNames(
        object = TRUE
        , nm = predcontrib_alias
      )
    )
    preds_contrib_s3_param <- predict(bst, X, params = params)
    preds_contrib_r6_param <- bst$predict(X, params = params)
    expect_equal(preds_contrib_s3_keyword, preds_contrib_s3_param)
    expect_equal(preds_contrib_s3_keyword, preds_contrib_r6_param)
  }
})

test_that("predictions for regression and binary classification are returned as vectors", {
    data(mtcars)
    X <- as.matrix(mtcars[, -1L])
    y <- as.numeric(mtcars[, 1L])
    dtrain <- lgb.Dataset(
      X
      , label = y
      , params = list(
        max_bins = 5L
        , min_data_in_bin = 1L
      )
    )
    model <- lgb.train(
      data = dtrain
      , obj = "regression"
      , nrounds = 5L
      , verbose = VERBOSITY
      , params = list(
        min_data_in_leaf = 1L
      )
    )
    pred <- predict(model, X)
    expect_true(is.vector(pred))
    expect_equal(length(pred), nrow(X))
    pred <- predict(model, X, rawscore = TRUE)
    expect_true(is.vector(pred))
    expect_equal(length(pred), nrow(X))

    data(agaricus.train, package = "lightgbm")
    X <- agaricus.train$data
    y <- agaricus.train$label
    dtrain <- lgb.Dataset(X, label = y)
    model <- lgb.train(
      data = dtrain
      , obj = "binary"
      , nrounds = 5L
      , verbose = VERBOSITY
    )
    pred <- predict(model, X)
    expect_true(is.vector(pred))
    expect_equal(length(pred), nrow(X))
    pred <- predict(model, X, rawscore = TRUE)
    expect_true(is.vector(pred))
    expect_equal(length(pred), nrow(X))
})

test_that("predictions for multiclass classification are returned as matrix", {
    data(iris)
    X <- as.matrix(iris[, -5L])
    y <- as.numeric(iris$Species) - 1.0
    dtrain <- lgb.Dataset(X, label = y)
    model <- lgb.train(
      data = dtrain
      , obj = "multiclass"
      , nrounds = 5L
      , verbose = VERBOSITY
      , params = list(num_class = 3L)
    )
    pred <- predict(model, X)
    expect_true(is.matrix(pred))
    expect_equal(nrow(pred), nrow(X))
    expect_equal(ncol(pred), 3L)
    pred <- predict(model, X, rawscore = TRUE)
    expect_true(is.matrix(pred))
    expect_equal(nrow(pred), nrow(X))
    expect_equal(ncol(pred), 3L)
})
