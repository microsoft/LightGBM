require(lightgbm)
require(methods)

# Load in the agaricus dataset
data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
dtrain <- lgb.Dataset(agaricus.train$data, label = agaricus.train$label)
dtest <- lgb.Dataset.create.valid(dtrain, data = agaricus.test$data, label = agaricus.test$label)

valids <- list(eval = dtest, train = dtrain)
#--------------------Advanced features ---------------------------
# advanced: start from an initial base prediction
print("Start running example to start from an initial prediction")

# Train lightgbm for 1 round
param <- list(num_leaves = 4,
              learning_rate = 1,
              nthread = 2,
              objective = "binary")
bst <- lgb.train(param, dtrain, 1, valids = valids)

# Note: we need the margin value instead of transformed prediction in set_init_score
ptrain <- predict(bst, agaricus.train$data, rawscore = TRUE)
ptest  <- predict(bst, agaricus.test$data, rawscore = TRUE)

# set the init_score property of dtrain and dtest
# base margin is the base prediction we will boost from
setinfo(dtrain, "init_score", ptrain)
setinfo(dtest, "init_score", ptest)

print("This is result of boost from initial prediction")
bst <- lgb.train(params = param,
                 data = dtrain,
                 nrounds = 5,
                 valids = valids)
