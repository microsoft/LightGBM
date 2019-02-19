require(lightgbm)
# load in the agaricus dataset
data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
dtrain <- lgb.Dataset(agaricus.train$data, label = agaricus.train$label)
dtest <- lgb.Dataset.create.valid(dtrain, data = agaricus.test$data, label = agaricus.test$label)

nrounds <- 2
param <- list(num_leaves = 4,
              learning_rate = 1,
              objective = "binary")

print("Running cross validation")
# Do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
lgb.cv(param,
       dtrain,
       nrounds,
       nfold = 5,
       eval = "binary_error")

print("Running cross validation, disable standard deviation display")
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
lgb.cv(param,
       dtrain,
       nrounds,
       nfold = 5,
       eval = "binary_error",
       showsd = FALSE)

# You can also do cross validation with cutomized loss function
print("Running cross validation, with cutomsized loss function")

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
  return(list(name = "error", value = err, higher_better = FALSE))
}

# train with customized objective
lgb.cv(params = param,
       data = dtrain,
       nrounds = nrounds,
       obj = logregobj,
       eval = evalerror,
       nfold = 5)
