context("lgb.check.r6.class")

test_that("lgb.check.r6.class() should return FALSE for NULL input", {
    expect_false(lgb.check.r6.class(NULL, "lgb.Dataset"))
})

test_that("lgb.check.r6.class() should return FALSE for non-R6 inputs", {
    x <- 5L
    class(x) <- "lgb.Dataset"
    expect_false(lgb.check.r6.class(x, "lgb.Dataset"))
})

test_that("lgb.check.r6.class() should correctly identify lgb.Dataset", {

    data("agaricus.train", package = "lightgbm")
    train <- agaricus.train
    ds <- lgb.Dataset(train$data, label = train$label)
    expect_true(lgb.check.r6.class(ds, "lgb.Dataset"))
    expect_false(lgb.check.r6.class(ds, "lgb.Predictor"))
    expect_false(lgb.check.r6.class(ds, "lgb.Booster"))
})
