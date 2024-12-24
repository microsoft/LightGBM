test_that("lgb.plot.tree works as expected"){
  data(agaricus.train, package = "lightgbm")
  train <- agaricus.train
  dtrain <- lgb.Dataset(train$data, label = train$label)
  # define model parameters and build a single tree
  model <- lgb.train(
    params = list(
        objective = "regression"
        , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = 1L
    , verbose = .LGB_VERBOSITY
  )
  # plot the tree and compare to the tree table
  # trees start from 0 in lgb.model.dt.tree
  tree_table <- lgb.model.dt.tree(model)
  expect_true({
    lgb.plot.tree(model, 0)
  }, regexp = "lgb.plot.tree: Value of 'tree' should be between 1 and the total number of trees in the model")
}

test_that("lgb.plot.tree fails when a non existing tree is selected"){
  data(agaricus.train, package = "lightgbm")
  train <- agaricus.train
  dtrain <- lgb.Dataset(train$data, label = train$label)
  # define model parameters and build a single tree
  model <- lgb.train(
    params = list(
        objective = "regression"
        , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = 1L
    , verbose = .LGB_VERBOSITY
  )
  # plot the tree and compare to the tree table
  # trees start from 0 in lgb.model.dt.tree
  tree_table <- lgb.model.dt.tree(model)
  expect_error({
    lgb.plot.tree(model, 999)
  }, regexp = "lgb.plot.tree: Value of 'tree' should be between 1 and the total number of trees in the model")
}
