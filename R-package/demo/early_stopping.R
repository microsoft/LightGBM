require(lightgbm)
require(methods)
# load in the agaricus dataset
data(agaricus.train, package='lightgbm')
data(agaricus.test, package='lightgbm')
dtrain <- lgb.Dataset(agaricus.train$data, label = agaricus.train$label)
dtest <- lgb.Dataset(agaricus.test$data, label = agaricus.test$label)
# note: for customized objective function, we leave objective as default
# note: what we are getting is margin value in prediction
# you must know what you are doing
param <- list(num_leaves=4, learning_rate=1)
valids <- list(eval = dtest)
num_round <- 20
# user define objective function, given prediction, return gradient and second order gradient
# this is loglikelihood loss
logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}
# user defined evaluation function, return a pair metric_name, result, higher_better
# NOTE: when you do customized loss function, the default prediction value is margin
# this may make buildin evalution metric not function properly
# for example, we are doing logistic loss, the prediction is score before logistic transformation
# the buildin evaluation error assumes input is after logistic transformation
# Take this in mind when you use the customization, and maybe you need write customized evaluation function
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  return(list(name = "error", value = err, higher_better=FALSE))
}
print ('start training with early Stopping setting')

bst <- lgb.train(param, dtrain, num_round, valids, 
                 objective = logregobj, eval = evalerror,
                 early_stopping_round = 3)

