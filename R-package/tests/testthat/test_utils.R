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

context("lgb.params2str")

test_that("lgb.params2str() works as expected for empty lists", {
    out_str <- lgb.params2str(
        params = list()
    )
    expect_identical(class(out_str), "raw")
    expect_equal(out_str, lgb.c_str(""))
})

test_that("lgb.params2str() works as expected for a key in params with multiple different-length elements", {
    metrics <- c("a", "ab", "abc", "abcdefg")
    params <- list(
        objective = "magic"
        , metric = metrics
        , nrounds = 10L
        , learning_rate = 0.0000001
    )
    out_str <- lgb.params2str(
        params = params
    )
    expect_identical(class(out_str), "raw")
    out_as_char <- rawToChar(out_str)
    expect_identical(
        out_as_char
        , "objective=magic metric=a,ab,abc,abcdefg nrounds=10 learning_rate=0.0000001"
    )
})

context("lgb.last_error")

test_that("lgb.last_error() throws an error if there are no errors", {
    expect_error({
        lgb.last_error()
    }, regexp = "Everything is fine")
})

test_that("lgb.last_error() correctly returns errors from the C++ side", {
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dvalid1 <- lgb.Dataset(
        data = train$data
        , label = as.matrix(rnorm(5L))
    )
    expect_error({
        dvalid1$construct()
    }, regexp = "[LightGBM] [Fatal] Length of label is not same with #data", fixed = TRUE)
})
