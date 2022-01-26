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
    pred1 <- predict(bst, data = test$data, rawscore = TRUE)
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

test_that("Single-row predictions are identical to multi-row ones", {
    data(mtcars)
    X <- as.matrix(mtcars[, -1L])
    y <- mtcars[, 1L]
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bin = 5L))
    params <- list(min_data_in_leaf = 2L)
    model <- lgb.train(
      params = params
     , data = dtrain
     , obj = "regression"
     , nrounds = 5L
     , verbose = -1L
    )

    x1 <- X[1L, , drop = FALSE]
    x11 <- X[11L, , drop = FALSE]
    x1_spv <- as(x1, "sparseVector")
    x11_spv <- as(x11, "sparseVector")
    x1_csr <- as(x1, "RsparseMatrix")
    x11_csr <- as(x11, "RsparseMatrix")

    pred_all <- predict(model, X)
    pred1_wo_config <- predict(model, x1)
    pred11_wo_config <- predict(model, x11)
    pred1_spv_wo_config <- predict(model, x1_spv)
    pred11_spv_wo_config <- predict(model, x11_spv)
    pred1_csr_wo_config <- predict(model, x1_csr)
    pred11_csr_wo_config <- predict(model, x11_csr)

    lgb.configure_fast_predict(model)
    pred1_w_config <- predict(model, x1)
    pred11_w_config <- predict(model, x11)

    model <- lgb.train(
      params = params
     , data = dtrain
     , obj = "regression"
     , nrounds = 5L
     , verbose = -1L
    )
    lgb.configure_fast_predict(model, csr = TRUE)
    pred1_spv_w_config <- predict(model, x1_spv)
    pred11_spv_w_config <- predict(model, x11_spv)
    pred1_csr_w_config <- predict(model, x1_csr)
    pred11_csr_w_config <- predict(model, x11_csr)

    expect_equal(pred1_wo_config, pred_all[1L])
    expect_equal(pred11_wo_config, pred_all[11L])
    expect_equal(pred1_spv_wo_config, pred_all[1L])
    expect_equal(pred11_spv_wo_config, pred_all[11L])
    expect_equal(pred1_csr_wo_config, pred_all[1L])
    expect_equal(pred11_csr_wo_config, pred_all[11L])

    expect_equal(pred1_w_config, pred_all[1L])
    expect_equal(pred11_w_config, pred_all[11L])
    expect_equal(pred1_spv_w_config, pred_all[1L])
    expect_equal(pred11_spv_w_config, pred_all[11L])
    expect_equal(pred1_csr_w_config, pred_all[1L])
    expect_equal(pred11_csr_w_config, pred_all[11L])
})

test_that("Fast-predict configuration accepts non-default prediction types", {
    data(mtcars)
    X <- as.matrix(mtcars[, -1L])
    y <- mtcars[, 1L]
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bin = 5L))
    params <- list(min_data_in_leaf = 2L)
    model <- lgb.train(
      params = params
     , data = dtrain
     , obj = "regression"
     , nrounds = 5L
     , verbose = -1L
    )

    x1 <- X[1L, , drop = FALSE]
    x11 <- X[11L, , drop = FALSE]

    pred_all <- predict(model, X, predleaf = TRUE)
    pred1_wo_config <- predict(model, x1, predleaf = TRUE)
    pred11_wo_config <- predict(model, x11, predleaf = TRUE)
    expect_equal(pred1_wo_config, pred_all[1L, , drop = FALSE])
    expect_equal(pred11_wo_config, pred_all[11L, , drop = FALSE])

    lgb.configure_fast_predict(model, predleaf = TRUE)
    pred1_w_config <- predict(model, x1, predleaf = TRUE)
    pred11_w_config <- predict(model, x11, predleaf = TRUE)
    expect_equal(pred1_w_config, pred_all[1L, , drop = FALSE])
    expect_equal(pred11_w_config, pred_all[11L, , drop = FALSE])
})

test_that("Fast-predict configuration does not block other prediction types", {
    data(mtcars)
    X <- as.matrix(mtcars[, -1L])
    y <- mtcars[, 1L]
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bin = 5L))
    params <- list(min_data_in_leaf = 2L)
    model <- lgb.train(
      params = params
     , data = dtrain
     , obj = "regression"
     , nrounds = 5L
     , verbose = -1L
    )

    x1 <- X[1L, , drop = FALSE]
    x11 <- X[11L, , drop = FALSE]

    pred_all <- predict(model, X)
    pred_all_leaf <- predict(model, X, predleaf = TRUE)

    lgb.configure_fast_predict(model)
    pred1_w_config <- predict(model, x1)
    pred11_w_config <- predict(model, x11)
    pred1_leaf_w_config <- predict(model, x1, predleaf = TRUE)
    pred11_leaf_w_config <- predict(model, x11, predleaf = TRUE)

    expect_equal(pred1_w_config, pred_all[1L])
    expect_equal(pred11_w_config, pred_all[11L])
    expect_equal(pred1_leaf_w_config, pred_all_leaf[1L, , drop = FALSE])
    expect_equal(pred11_leaf_w_config, pred_all_leaf[11L, , drop = FALSE])
})
