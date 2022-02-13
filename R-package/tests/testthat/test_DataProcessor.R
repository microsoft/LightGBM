# Note: the lgb.DataProcessor class is meant to look for symbols in two
# environments above from where it is called. Thus, it should not be called
# or tested directly, only as part of calls to lightgbm().
library(Matrix)
data("iris")
data("mtcars")
data(bank, package = "lightgbm")

test_that("lightgbm() and predict() work with formula interface", {
  model <- lightgbm(
    Species ~ .
    , data = iris
    , nthreads = 1L
    , verbose = -1L
  )
  pred <- predict(model, iris, type = "class")
  expect_true(all(pred == iris$Species))

  expect_s3_class(pred, "factor")
  expect_equal(levels(pred), levels(iris$Species))

  model <- lightgbm(
    Species ~ . + log(Petal.Length) + I(Petal.Length^2.0) - Sepal.Width
    , data = iris
    , nthreads = 1L
    , verbose = -1L
  )
  pred <- predict(model, iris, type = "class")
  expect_true(all(pred == iris$Species))
  expect_equal(5L, .Call(LGBM_BoosterGetNumFeature_R, model$.__enclos_env__$private$handle))
})

test_that("lightgbm() and predict() work with data.frame interface", {
  model <- lightgbm(
    iris
    , Species
    , nthreads = 1L
    , verbose = -1L
  )
  pred <- predict(model, iris, type = "class")
  expect_true(all(pred == iris$Species))
  expect_s3_class(pred, "factor")
  expect_equal(levels(pred), levels(iris$Species))

  model <- lightgbm(
    iris
    , "Species"
    , nthreads = 1L
    , verbose = -1L
  )
  pred <- predict(model, iris, type = "class")
  expect_true(all(pred == iris$Species))
  expect_s3_class(pred, "factor")
  expect_equal(levels(pred), levels(iris$Species))

  model <- lightgbm(
    iris[, -5L]
    , iris$Species
    , nthreads = 1L
    , verbose = -1L
  )
  pred <- predict(model, iris, type = "class")
  expect_true(all(pred == iris$Species))
  expect_s3_class(pred, "factor")
  expect_equal(levels(pred), levels(iris$Species))
})

test_that("lightgbm() and predict() accept data.tables as data.frames", {
  iris_dt <- data.table::as.data.table(iris)
  model <- lightgbm(
    iris_dt
    , Species
    , nthreads = 1L
    , verbose = -1L
  )
  pred <- predict(model, iris_dt, type = "class")
  expect_true(all(pred == iris_dt$Species))
  expect_s3_class(pred, "factor")
  expect_equal(levels(pred), levels(iris_dt$Species))

  model <- lightgbm(
    iris_dt
    , "Species"
    , nthreads = 1L
    , verbose = -1L
  )
  pred <- predict(model, iris_dt, type = "class")
  expect_true(all(pred == iris$Species))
  expect_s3_class(pred, "factor")
  expect_equal(levels(pred), levels(iris_dt$Species))

  model <- lightgbm(
    iris_dt[, -5L]
    , iris_dt$Species
    , nthreads = 1L
    , verbose = -1L
  )
  pred <- predict(model, iris_dt, type = "class")
  expect_true(all(pred == iris_dt$Species))
  expect_s3_class(pred, "factor")
  expect_equal(levels(pred), levels(iris_dt$Species))
})

test_that("lightgbm() and predict() work with matrix interface", {
  model <- lightgbm(
    as.matrix(mtcars)
    , mpg
    , nthreads = 1L
    , verbose = -1L
    , nrounds = 5L
    , params = list(
      max_bins = 5L
      , min_data_in_leaf = 5L
    )
  )
  pred <- predict(model, mtcars)
  expect_true(all(names(pred) == row.names(mtcars)))

  model <- lightgbm(
    as.matrix(mtcars)
    , "mpg"
    , nthreads = 1L
    , verbose = -1L
    , nrounds = 5L
    , params = list(
      max_bins = 5L
      , min_data_in_leaf = 5L
    )
  )
  pred2 <- predict(model, mtcars)
  expect_true(all(pred == pred2))

  model <- lightgbm(
    as.matrix(mtcars[, -1L])
    , mtcars$mpg
    , nthreads = 1L
    , verbose = -1L
    , nrounds = 5L
    , params = list(
      max_bins = 5L
      , min_data_in_leaf = 5L
    )
  )
  pred3 <- predict(model, mtcars)
  expect_true(all(pred == pred3))
})

test_that("lightgbm() and predict() work with dgCMatrix interface", {
  model <- lightgbm(
    as(as.matrix(mtcars), "dgCMatrix")
    , mpg
    , nthreads = 1L
    , verbose = -1L
    , nrounds = 5L
    , params = list(
      max_bins = 5L
      , min_data_in_leaf = 5L
    )
  )
  pred <- predict(model, mtcars)
  expect_true(all(names(pred) == row.names(mtcars)))

  model <- lightgbm(
    as(as.matrix(mtcars), "dgCMatrix")
    , "mpg"
    , nthreads = 1L
    , verbose = -1L
    , nrounds = 5L
    , params = list(
      max_bins = 5L
      , min_data_in_leaf = 5L
    )
  )
  pred2 <- predict(model, mtcars)
  expect_true(all(pred == pred2))

  model <- lightgbm(
    as(as.matrix(mtcars[, -1L]), "dgCMatrix")
    , mtcars$mpg
    , nthreads = 1L
    , verbose = -1L
    , nrounds = 5L
    , params = list(
      max_bins = 5L
      , min_data_in_leaf = 5L
    )
  )
  pred3 <- predict(model, mtcars)
  expect_true(all(pred == pred3))
})

test_that("lightgbm() handles single-column inputs", {
  model <- lightgbm(
    iris[, 1L, drop = FALSE]
    , iris$Species
    , nrounds = 5L
    , nthreads = 1L
    , verbose = -1L
  )
  pred <- predict(model, iris, type = "score")
  expect_equal(nrow(pred), nrow(iris))
  expect_equal(ncol(pred), 3L)
})

test_that("lightbm() data.frame interface handles categorical features", {
  model <- lightgbm(
    bank
    , y
    , nrounds = 5L
    , nthreads = 1L
    , verbose = -1L
  )
  expect_equal(
    model$params$categorical_feature
    , unname(which(sapply(within(bank, rm(y)), is.character)))
  )
})

test_that("lightgbm() accepts dataset parameters", {
  set.seed(123L)
  df <- data.frame(col1 = c(runif(1000L), rep(0.0, 100L)))
  df$col2 <- df$col1
  n_bins <- 5L
  model <- lightgbm(
    df
    , col2
    , nthreads = 1L
    , verbose = -1L
    , params = list(max_bin = n_bins)
  )
  expect_equal(length(table(predict(model, df))), n_bins)

  model <- lightgbm(
    df
    , col2
    , nthreads = 1L
    , verbose = -1L
    , dataset_params = list(max_bin = n_bins)
  )
  expect_equal(length(table(predict(model, df))), n_bins)
})

test_that("lightgbm() accepts NSE for different arguments", {
  iris_dt <- data.table::as.data.table(iris)
  iris_dt[, wcol := 1.0]
  model <- lightgbm(
    iris_dt
    , "Species"
    , weights = wcol
    , nrounds = 5L
    , nthreads = 1L
    , verbose = -1L
  )
  expect_equal(
    ncol(iris_dt) - 2L
    , .Call(LGBM_BoosterGetNumFeature_R, model$.__enclos_env__$private$handle)
  )
})

test_that("lightgbm() does not throw warnings in the presence of NAs", {
  df <- data.frame(
    col1 = rep(c(1.0, 2.0, NA), 100L)
    , col2 = rep(c("a", NA, "b"), 100L)
    , col3 = rep(c(1.0, 2.0, 1.0), 100L)
  )
  expect_warning({
    model <- lightgbm(
      df
      , col3
      , nrounds = 5L
      , nthreads = 1L
      , verbose = -1L
    )
    pred <- predict(model, df)
  }, regexp = NA)
})

test_that("lightgbm() adjusts objective according to data", {
  model <- lightgbm(
    mpg ~ .
    , data = mtcars
    , nrounds = 5L
    , nthreads = 1L
    , verbose = -1L
  )
  expect_equal(model$params$objective, "regression")

  model <- lightgbm(
    y ~ .
    , data = bank
    , nrounds = 5L
    , nthreads = 1L
    , verbose = -1L
  )
  expect_equal(model$params$objective, "binary")

  model <- lightgbm(
    Species ~ .
    , data = iris
    , nrounds = 5L
    , nthreads = 1L
    , verbose = -1L
  )
  expect_equal(model$params$objective, "multiclass")
  expect_equal(model$params$num_class, length(levels(iris$Species)))

  data("agaricus.train")
  model <- lightgbm(
    agaricus.train$data
    , agaricus.train$label
    , objective = "poisson"
    , nrounds = 5L
    , nthreads = 1L
    , verbose = -1L
  )
  expect_equal(model$params$objective, "poisson")
})

test_that("predict() from lightgbm() names columns correctly", {
  model <- lightgbm(
    Species ~ .
    , data = iris
    , nrounds = 1L
    , nthreads = 1L
    , verbose = -1L
  )
  pred_score <- predict(model, iris, type = "score")
  pred_class <- predict(model, iris, type = "class")
  pred_leaf <- predict(model, iris, type = "leaf")
  pred_contrib <- predict(model, iris, type = "contrib")

  expect_equal(colnames(pred_score), levels(iris$Species))
  expect_equal(levels(pred_class), levels(iris$Species))
  expect_null(colnames(pred_leaf))
  expect_null(colnames(pred_contrib))

  model <- lightgbm(
    mpg ~ .
    , data = mtcars
    , nrounds = 10L
    , nthreads = 1L
    , verbose = -1L
    , params = list(
      max_bin = 5L
      , min_data_in_leaf = 5L
    )
  )
  pred_score <- predict(model, mtcars, type = "score")
  expect_error(pred_class <- predict(model, mtcars, type = "class"))
  pred_leaf <- predict(model, mtcars, type = "leaf")
  pred_contrib <- predict(model, mtcars, type = "contrib")

  expect_true(is.numeric(pred_score))
  expect_null(dim(pred_score))
  expect_null(colnames(pred_score))

  expect_equal(ncol(pred_leaf), 10L)
  expect_null(colnames(pred_leaf))
  expect_equal(
    colnames(pred_contrib)
    , c(names(mtcars)[names(mtcars) != "mpg"], "(Intercept)")
  )

  model <- lightgbm(
    mpg ~ cyl + wt
    , data = mtcars
    , nrounds = 10L
    , nthreads = 1L
    , verbose = -1L
    , params = list(
      max_bin = 5L
      , min_data_in_leaf = 5L
    )
  )
  pred_contrib <- predict(model, mtcars, type = "contrib")
  expect_equal(
    colnames(pred_contrib)
    , c("cyl", "wt", "(Intercept)")
  )
})
