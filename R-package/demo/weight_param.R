# This demo R code is to provide a demonstration of hyperparameter adjustment
# when scaling weights for appropriate learning
# As with any optimizers, bad parameters can impair performance

# Load library
library(lightgbm)

# We will train a model with the following scenarii:
# - Run 1: sum of weights equal to 0.06513 without adjusted regularization (not learning)
# - Run 2: sum of weights equal to 0.06513 with adjusted regularization (learning)
# - Run 3: sum of weights equal to 6513 (x 1e5) with adjusted regularization (learning)

# Setup small weights
weights1 <- rep(1.0 / 100000.0, 6513L)
weights2 <- rep(1.0 / 100000.0, 1611L)

# Load data and create datasets
data(agaricus.train, package = "lightgbm")
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label, weight = weights1)
data(agaricus.test, package = "lightgbm")
test <- agaricus.test
dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label, weight = weights2)
valids <- list(test = dtest)

# Run 1: sum of weights equal to 0.06513 without adjusted regularization (not learning)
# It cannot learn because regularization is too large!
# min_sum_hessian alone is bigger than the sum of weights, thus you will never learn anything
params <- list(
    objective = "regression"
    , metric = "l2"
    , device = "cpu"
    , min_sum_hessian = 10.0
    , num_leaves = 7L
    , max_depth = 3L
    , nthread = 1L
)
model <- lgb.train(
    params
    , dtrain
    , 50L
    , valids
    , min_data = 1L
    , learning_rate = 1.0
    , early_stopping_rounds = 10L
)
weight_loss <- as.numeric(model$record_evals$test$l2$eval)
plot(weight_loss) # Shows how poor the learning was: a straight line!

# Run 2: sum of weights equal to 0.06513 with adjusted regularization (learning)
# Adjusted regularization just consisting in multiplicating results by 1e4 (x10000)
# Notice how it learns, there is no issue as we adjusted regularization ourselves
params <- list(
    objective = "regression"
    , metric = "l2"
    , device = "cpu"
    , min_sum_hessian = 1e-4
    , num_leaves = 7L
    , max_depth = 3L
    , nthread = 1L
)
model <- lgb.train(
    params
    , dtrain
    , 50L
    , valids
    , min_data = 1L
    , learning_rate = 1.0
    , early_stopping_rounds = 10L
)
small_weight_loss <- as.numeric(model$record_evals$test$l2$eval)
plot(small_weight_loss) # It learns!

# Run 3: sum of weights equal to 6513 (x 1e5) with adjusted regularization (learning)
# To make it better, we are first cleaning the environment and reloading LightGBM
lgb.unloader(wipe = TRUE)

# And now, we are doing as usual
library(lightgbm)
data(agaricus.train, package = "lightgbm")
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label)
data(agaricus.test, package = "lightgbm")
test <- agaricus.test
dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
valids <- list(test = dtest)

# Setup parameters and run model...
params <- list(
    objective = "regression"
    , metric = "l2"
    , device = "cpu"
    , min_sum_hessian = 10.0
    , num_leaves = 7L
    , max_depth = 3L
    , nthread = 1L
)
model <- lgb.train(
    params
    , dtrain
    , 50L
    , valids
    , min_data = 1L
    , learning_rate = 1.0
    , early_stopping_rounds = 10L
)
large_weight_loss <- as.numeric(model$record_evals$test$l2$eval)
plot(large_weight_loss) # It learns!


# Do you want to compare the learning? They both converge.
plot(small_weight_loss, large_weight_loss)
curve(1.0 * x, from = 0L, to = 0.02, add = TRUE)
