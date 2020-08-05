context("Predictor")

test_that("predictions do not fail for integer input", {
    X <- as.matrix(as.integer(iris[, "Species"]), ncol = 1L)
    y <- iris[["Sepal.Length"]]
    dtrain <- lgb.Dataset(X, label = y)
    fit <- lgb.train(
        data = dtrain
        , objective = "regression"
        , verbose = -1L
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
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , num_leaves = 4L
        , learning_rate = 0.1
        , nrounds = 100L
        , objective = "binary"
        , save_name = tempfile(fileext = ".model")
    )
    expect_true(lgb.is.Booster(bst))
    pred1 <- predict(bst, test$data, rawscore=TRUE)
    pred_contrib1 <- predict(bst, test$data, predcontrib=TRUE)
    pred2 <- rep(0L, length(pred1))
    pred_contrib2 <- rep(0L, length((pred2)))
    step <- 11L
    start_iters <- seq(0L, 49L, by=step)
    for (start_iter in start_iters) {
        inc_pred <- predict(bst, test$data, start_iteration=start_iter, num_iteration=step, rawscore=TRUE)
        inc_pred_contrib <- predict(bst, test$data, start_iteration=start_iter, num_iteration=step, predcontrib=TRUE)
        pred2 <- pred2 + inc_pred
        pred_contrib2 <- pred_contrib2 + inc_pred_contrib
    }
    expect_equal(pred2, pred1)
    expect_equal(pred_contrib2, pred_contrib1)
})