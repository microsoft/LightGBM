VERBOSITY <- as.integer(
    Sys.getenv("LIGHTGBM_TEST_VERBOSITY", "-1")
)

test_that("lgb.unloader works as expected", {
    if (require(covr)) {  # nolint
        testthat::skip_if(
            condition = covr::in_covr()
            , message = "lgb.unloader() tests are skipped when calculating test coverage"
        )
    }
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(train$data, label = train$label)
    bst <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
            , min_data = 1L
            , learning_rate = 1.0
            , verbosity = VERBOSITY
        )
        , data = dtrain
        , nrounds = 1L
    )
    expect_true(exists("bst"))
    result <- lgb.unloader(restore = TRUE, wipe = TRUE, envir = environment())
    expect_false(exists("bst"))
    expect_null(result)
})

test_that("lgb.unloader finds all boosters and removes them", {
    if (require(covr)) {  # nolint
        testthat::skip_if(
            condition = covr::in_covr()
            , message = "lgb.unloader() tests are skipped when calculating test coverage"
        )
    }
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(train$data, label = train$label)
    bst1 <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
            , min_data = 1L
            , learning_rate = 1.0
            , verbosity = VERBOSITY
        )
        , data = dtrain
        , nrounds = 1L
    )
    bst2 <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
            , min_data = 1L
            , learning_rate = 1.0
            , verbosity = VERBOSITY
        )
        , data = dtrain
        , nrounds = 1L
    )
    expect_true(exists("bst1"))
    expect_true(exists("bst2"))
    lgb.unloader(restore = TRUE, wipe = TRUE, envir = environment())
    expect_false(exists("bst1"))
    expect_false(exists("bst2"))
})
