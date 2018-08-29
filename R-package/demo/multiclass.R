require(lightgbm)

# We load the default iris dataset shipped with R
data(iris)

# We must convert factors to numeric
# They must be starting from number 0 to use multiclass
# For instance: 0, 1, 2, 3, 4, 5...
iris$Species <- as.numeric(as.factor(iris$Species)) - 1

# We cut the data set into 80% train and 20% validation
# The 10 last samples of each class are for validation

train <- as.matrix(iris[c(1:40, 51:90, 101:140), ])
test <- as.matrix(iris[c(41:50, 91:100, 141:150), ])
dtrain <- lgb.Dataset(data = train[, 1:4], label = train[, 5])
dtest <- lgb.Dataset.create.valid(dtrain, data = test[, 1:4], label = test[, 5])
valids <- list(test = dtest)

# Method 1 of training
params <- list(objective = "multiclass", metric = "multi_error", num_class = 3)
model <- lgb.train(params,
                   dtrain,
                   100,
                   valids,
                   min_data = 1,
                   learning_rate = 1,
                   early_stopping_rounds = 10)

# We can predict on test data, outputs a 90-length vector
# Order: obs1 class1, obs1 class2, obs1 class3, obs2 class1, obs2 class2, obs2 class3...
my_preds <- predict(model, test[, 1:4])

# Method 2 of training, identical
model <- lgb.train(list(),
                   dtrain,
                   100,
                   valids,
                   min_data = 1,
                   learning_rate = 1,
                   early_stopping_rounds = 10,
                   objective = "multiclass",
                   metric = "multi_error",
                   num_class = 3)

# We can predict on test data, identical
my_preds <- predict(model, test[, 1:4])

# A (30x3) matrix with the predictions, use parameter reshape
# class1 class2 class3
#   obs1   obs1   obs1
#   obs2   obs2   obs2
#   ....   ....   ....
my_preds <- predict(model, test[, 1:4], reshape = TRUE)

# We can also get the predicted scores before the Sigmoid/Softmax application
my_preds <- predict(model, test[, 1:4], rawscore = TRUE)

# Raw score predictions as matrix instead of vector
my_preds <- predict(model, test[, 1:4], rawscore = TRUE, reshape = TRUE)

# We can also get the leaf index
my_preds <- predict(model, test[, 1:4], predleaf = TRUE)

# Predict leaf index as matrix instead of vector
my_preds <- predict(model, test[, 1:4], predleaf = TRUE, reshape = TRUE)
