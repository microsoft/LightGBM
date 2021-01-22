library(testthat)
library(lightgbm)

data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
train <- agaricus.train
test <- agaricus.test

test_check(
    package = "lightgbm"
    , stop_on_failure = TRUE
    , stop_on_warning = FALSE
)

#test_that(paste0("CTR for R package works"), {
  # test cat_converters
#  set.seed(1L)
#  dtrain <- lgb.Dataset(train$data, label = train$label)
#  dtest <- lgb.Dataset(test$data, label = test$label, reference = dtrain)
#  print(dtrain$dim())
#  print(dtest$dim())
  # ``` cat_converters = "" ```   is equal to   ``` cat_converters = "raw" ```
#  params <- list(objective = "binary", categorical_feature = c(1L, 2L, 3L, 4L), cat_converters = "")
#  bst <- lightgbm(
#    data = dtrain
#    , params = params
#    , nrounds = 10L
#    , verbose = 2L
#    , valids = list("valid1" = dtest)
#  )
#  pred1 <- bst$predict(test$data)

#  dtrain <- lgb.Dataset(train$data, label = train$label,
#    categorical_feature = c(1L, 2L, 3L, 4L, 5L), cat_converters = "raw")
#  dtest <- lgb.Dataset(test$data, label = test$label,
#    categorical_feature = c(1L, 2L, 3L, 4L, 5L), reference = dtrain)
#  params <- list(objective = "binary")
#  bst <- lightgbm(
#    data = dtrain
#    , params = params
#    , nrounds = 10L
#    , verbose = 2L
#    , valids = list("valid1" = dtest)
#  )
#  pred2 <- bst$predict(test$data)
#  expect_equal(pred1, pred2)

  #dtrain <- lgb.Dataset(train$data, label = train$label,
  #  categorical_feature = c(1L, 2L, 3L, 4L, 5L))
  #dtest <- lgb.Dataset(test$data, label = test$label,
  #  categorical_feature = c(1L, 2L, 3L, 4L, 5L), reference = dtrain)
  #params <- list(objective = "binary", cat_converters = "ctr,count,raw")
  #bst <- lightgbm(
  #  data = dtrain
  #  , params = params
  #  , nrounds = 10L
  #  , verbose = 2L
  #  , valids = list("valid1" = dtest)
  #)
  #pred3 <- bst$predict(test$data)

  #err_pred1 <- sum((pred1 > 0.5) != test$label) / length(test$label)
  #err_pred3 <- sum((pred3 > 0.5) != test$label) / length(test$label)
  #expect_lt(err_pred3, err_pred1)


  # test gbdt model with cat_converters
  #model_file <- tempfile(fileext = ".model")
  #lgb.save(bst, model_file)
  # finalize the booster and destroy it so you know we aren't cheating
  #bst$finalize()
  #expect_null(bst$.__enclos_env__$private$handle)
  #rm(bst)

  #bst2 <- lgb.load(
  #    filename = model_file
  #)
  #pred4 <- predict(bst2, test$data)
  #expect_equal(pred3, pred4)


  # test Dataset binary store with cat_converters
  #tmp_file <- tempfile(pattern = "lgb.Dataset_CTR_")
  #lgb.Dataset.save(
  #  dataset = dtrain
  #  , fname = tmp_file
  #)
  #dtrain_read_in <- lgb.Dataset(data = tmp_file)

  #tmp_file <- tempfile(pattern = "lgb.Dataset_CTR_")
  #lgb.Dataset.save(
  #  dataset = dtest
  #  , fname = tmp_file
  #)
  #dtest_read_in <- lgb.Dataset(data = tmp_file)

  #bst <- lightgbm(
  #  data = dtrain_read_in
  #  , params = params
  #  , nrounds = 10L
  #  , verbose = 2L
  #  , valids = list("valid1" = dtest_read_in)
  #)
  #pred5 <- bst$predict(test$data)
  #expect_equal(pred3, pred5)
#})
