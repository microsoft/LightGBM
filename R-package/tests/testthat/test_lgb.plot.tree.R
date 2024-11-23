test_that("lgb.plot.tree works as expected"){
  data(agaricus.train, package = "lightgbm")
  train <- agaricus.train
  dtrain <- lgb.Dataset(train$data, label = train$label)
  data(agaricus.test, package = "lightgbm")
  test <- agaricus.test
  dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
  # define model parameters and build a single tree
  params <- list(
    objective = "regression"
    , metric = "l2"
    , min_data = 1L
    , learning_rate = 1.0
  )
  valids <- list(test = dtest)
  model <- lgb.train(
    params = params
    , data = dtrain
    , nrounds = 1L
    , valids = valids
    , early_stopping_rounds = 1L
  )
  # plot the tree and compare to the tree table
  # trees start from 0 in lgb.model.dt.tree
  tree_table <- lgb.model.dt.tree(model)
  expect_true({
    lgb.plot.tree(model, 0)TRUE
  })
}

test_that("lgb.plot.tree fails when a non existing tree is selected"){
  data(agaricus.train, package = "lightgbm")
  train <- agaricus.train
  dtrain <- lgb.Dataset(train$data, label = train$label)
  data(agaricus.test, package = "lightgbm")
  test <- agaricus.test
  dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
  # define model parameters and build a single tree
  params <- list(
    objective = "regression"
    , metric = "l2"
    , min_data = 1L
    , learning_rate = 1.0
  )
  valids <- list(test = dtest)
  model <- lgb.train(
    params = params
    , data = dtrain
    , nrounds = 1L
    , valids = valids
    , early_stopping_rounds = 1L
  )
  # plot the tree and compare to the tree table
  # trees start from 0 in lgb.model.dt.tree
  tree_table <- lgb.model.dt.tree(model)
  expect_error({
    lgb.plot.tree(model, 999)TRUE
  })
}
