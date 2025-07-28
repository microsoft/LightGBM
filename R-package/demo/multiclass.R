library(lightgbm)

# We load the default iris dataset shipped with R
data(iris)

# We must convert factors to numeric
# They must be starting from number 0 to use multiclass
# For instance: 0, 1, 2, 3, 4, 5...
iris$Species <- as.numeric(as.factor(iris$Species)) - 1L

# We cut the data set into 80% train and 20% validation
# The 10 last samples of each class are for validation

train <- as.matrix(iris[c(1L:40L, 51L:90L, 101L:140L), ])
test <- as.matrix(iris[c(41L:50L, 91L:100L, 141L:150L), ])
dtrain <- lgb.Dataset(data = train[, 1L:4L], label = train[, 5L])
dtest <- lgb.Dataset.create.valid(dtrain, data = test[, 1L:4L], label = test[, 5L])
valids <- list(test = dtest)

# Method 1 of training
params <- list(
    objective = "multiclass"
    , metric = "multi_error"
    , num_class = 3L
    , min_data = 1L
    , learning_rate = 1.0
)
model <- lgb.train(
    params
    , dtrain
    , 100L
    , valids
    , early_stopping_rounds = 10L
)

# We can predict on test data, outputs a 90-length vector
# Order: obs1 class1, obs1 class2, obs1 class3, obs2 class1, obs2 class2, obs2 class3...
my_preds <- predict(model, test[, 1L:4L])

# Method 2 of training, identical
params <- list(
    min_data = 1L
    , learning_rate = 1.0
    , objective = "multiclass"
    , metric = "multi_error"
    , num_class = 3L
)
model <- lgb.train(
    params
    , dtrain
    , 100L
    , valids
    , early_stopping_rounds = 10L
)

# We can predict on test data, identical
my_preds <- predict(model, test[, 1L:4L])

# A (30x3) matrix with the predictions
# class1 class2 class3
#   obs1   obs1   obs1
#   obs2   obs2   obs2
#   ....   ....   ....
my_preds <- predict(model, test[, 1L:4L])

# We can also get the predicted scores before the Sigmoid/Softmax application
my_preds <- predict(model, test[, 1L:4L], type = "raw")

# We can also get the leaf index
my_preds <- predict(model, test[, 1L:4L], type = "leaf")
