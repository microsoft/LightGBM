require(lightgbm)
require(methods)

# load in the agaricus dataset
data(agaricus.train, package='lightgbm')
data(agaricus.test, package='lightgbm')
dtrain <- lgb.Dataset(agaricus.train$data, label = agaricus.train$label)
dtest <- lgb.Dataset(agaricus.test$data, label = agaricus.test$label)

valids <- list(eval = dtest, train = dtrain)
###
# advanced: start from a initial base prediction
#
print('start running example to start from a initial prediction')
# train lightgbm for 1 round
param <- list(num_leaves=4, learning_rate=1, nthread = 2, silent=1, objective='binary')
bst <- lgb.train(param, dtrain, 1, valids=valids)
# Note: we need the margin value instead of transformed prediction in set_init_score

ptrain <- predict(bst, agaricus.train$data, rawscore=TRUE)
ptest  <- predict(bst, agaricus.test$data, rawscore=TRUE)
# set the init_score property of dtrain and dtest
# base margin is the base prediction we will boost from
setinfo(dtrain, "init_score", ptrain)
setinfo(dtest, "init_score", ptest)

print('this is result of boost from initial prediction')
bst <- lgb.train(params = param, data = dtrain, nrounds = 5, valids = valids)
