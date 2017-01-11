context('Test models with custom objective')

data(agaricus.train, package='lightgbm')
data(agaricus.test, package='lightgbm')
dtrain <- lgb.Dataset(agaricus.train$data, label = agaricus.train$label)
dtest <- lgb.Dataset(agaricus.test$data, label = agaricus.test$label)
watchlist <- list(eval = dtest, train = dtrain)

logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1 / (1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0))) / length(labels)
  return(list(name = "error", value = err, higher_better=FALSE))
}

param <- list(num_leaves=8, learning_rate=1,
              objective=logregobj, metric="auc")
num_round <- 10

test_that("custom objective works", {
  bst <- lgb.train(param, dtrain, num_round, watchlist, eval = evalerror)
  expect_false(is.null(bst$record_evals))
})
