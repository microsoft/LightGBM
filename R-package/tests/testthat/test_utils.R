context("lgb.encode.char")

test_that("lgb.encode.char throws an informative error if it is passed a non-raw input", {
    x <- "some-string"
    expect_error({
        lgb.encode.char(x)
    }, regexp = "Can only encode from raw type")
})

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
    testthat::skip(paste0(
        "Skipping this test because it causes valgrind to think "
        , "there is a memory leak, and needs to be rethought"
    ))
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

context("lgb.check.eval")

test_that("lgb.check.eval works as expected with no metric", {
    params <- lgb.check.eval(
        params = list(device = "cpu")
        , eval = "binary_error"
    )
    expect_named(params, c("device", "metric"))
    expect_identical(params[["metric"]], list("binary_error"))
})

test_that("lgb.check.eval adds eval to metric in params", {
    params <- lgb.check.eval(
        params = list(metric = "auc")
        , eval = "binary_error"
    )
    expect_named(params, "metric")
    expect_identical(params[["metric"]], list("auc", "binary_error"))
})

test_that("lgb.check.eval adds eval to metric in params if two evaluation names are provided", {
    params <- lgb.check.eval(
        params = list(metric = "auc")
        , eval = c("binary_error", "binary_logloss")
    )
    expect_named(params, "metric")
    expect_identical(params[["metric"]], list("auc", "binary_error", "binary_logloss"))
})

test_that("lgb.check.eval adds eval to metric in params if a list is provided", {
    params <- lgb.check.eval(
        params = list(metric = "auc")
        , eval = list("binary_error", "binary_logloss")
    )
    expect_named(params, "metric")
    expect_identical(params[["metric"]], list("auc", "binary_error", "binary_logloss"))
})

test_that("lgb.check.eval drops duplicate metrics and preserves order", {
    params <- lgb.check.eval(
        params = list(metric = "l1")
        , eval = list("l2", "rmse", "l1", "rmse")
    )
    expect_named(params, "metric")
    expect_identical(params[["metric"]], list("l1", "l2", "rmse"))
})

context("lgb.check.wrapper_param")

test_that("lgb.check.wrapper_param() uses passed-in keyword arg if no alias found in params", {
    kwarg_val <- sample(seq_len(100L), size = 1L)
    params <- lgb.check.wrapper_param(
        main_param_name = "num_iterations"
        , params = list()
        , alternative_kwarg_value = kwarg_val
    )
    expect_equal(params[["num_iterations"]], kwarg_val)
})

test_that("lgb.check.wrapper_param() prefers main parameter to alias and keyword arg", {
    num_iterations <- sample(seq_len(100L), size = 1L)
    kwarg_val <- sample(seq_len(100L), size = 1L)
    params <- lgb.check.wrapper_param(
        main_param_name = "num_iterations"
        , params = list(
            num_iterations = num_iterations
            , num_tree = sample(seq_len(100L), size = 1L)
            , n_estimators = sample(seq_len(100L), size = 1L)
        )
        , alternative_kwarg_value = kwarg_val
    )
    expect_equal(params[["num_iterations"]], num_iterations)

    # aliases should be removed
    expect_identical(params, list(num_iterations = num_iterations))
})

test_that("lgb.check.wrapper_param() prefers alias to keyword arg", {
    n_estimators <- sample(seq_len(100L), size = 1L)
    num_tree <- sample(seq_len(100L), size = 1L)
    kwarg_val <- sample(seq_len(100L), size = 1L)
    params <- lgb.check.wrapper_param(
        main_param_name = "num_iterations"
        , params = list(
            num_tree = num_tree
            , n_estimators = n_estimators
        )
        , alternative_kwarg_value = kwarg_val
    )
    expect_equal(params[["num_iterations"]], num_tree)
    expect_identical(params, list(num_iterations = num_tree))

    # switching the order should switch which one is chosen
    params2 <- lgb.check.wrapper_param(
        main_param_name = "num_iterations"
        , params = list(
            n_estimators = n_estimators
            , num_tree = num_tree
        )
        , alternative_kwarg_value = kwarg_val
    )
    expect_equal(params2[["num_iterations"]], n_estimators)
    expect_identical(params2, list(num_iterations = n_estimators))
})
