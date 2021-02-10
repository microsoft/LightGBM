context("lgb.unloader")

test_that("lgb.unloader works as expected", {
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(train$data, label = train$label)
    bst <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
        )
        , data = dtrain
        , nrounds = 1L
        , min_data = 1L
        , learning_rate = 1.0
    )
    expect_true(exists("bst"))
    result <- lgb.unloader(restore = TRUE, wipe = TRUE, envir = environment())
    expect_false(exists("bst"))
    expect_null(result)
})

test_that("lgb.unloader finds all boosters and removes them", {
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(train$data, label = train$label)
    bst1 <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
        )
        , data = dtrain
        , nrounds = 1L
        , min_data = 1L
        , learning_rate = 1.0
    )
    bst2 <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
        )
        , data = dtrain
        , nrounds = 1L
        , min_data = 1L
        , learning_rate = 1.0
    )
    expect_true(exists("bst1"))
    expect_true(exists("bst2"))
    lgb.unloader(restore = TRUE, wipe = TRUE, envir = environment())
    expect_false(exists("bst1"))
    expect_false(exists("bst2"))
})
