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

# Method 1 of training with built-in multiclass objective
model_builtin <- lgb.train(list(),
                           dtrain,
                           100,
                           valids,
                           min_data = 1,
                           learning_rate = 1,
                           early_stopping_rounds = 10,
                           objective = "multiclass",
                           metric = "multi_logloss",
                           num_class = 3)

preds_builtin <- predict(model_builtin, test[, 1:4], rawscore = TRUE)

# Method 2 of training with custom objective function

# User defined objective function, given prediction, return gradient and second order gradient
custom_multiclass_obj = function(preds, dtrain) {
    labels = getinfo(dtrain, "label")
    
    # preds is a matrix with rows corresponding to samples and colums corresponding to choices
    preds = matrix(preds, nrow = length(labels))
    
    # to prevent overflow, normalize preds by row
    preds = preds - apply(preds, 1, max)
    prob = exp(preds) / rowSums(exp(preds))
    
    # compute gradient
    grad = prob
    grad[cbind(1:length(labels), labels + 1)] = grad[cbind(1:length(labels), labels + 1)] - 1
    
    # compute hessian (approximation)
    hess = 2 * prob * (1 - prob)
    
    return(list(grad = grad, hess = hess))
}

# define custom metric 
custom_multiclass_metric = function(preds, dtrain) {
    labels = getinfo(dtrain, "label")
    preds = matrix(preds, nrow = length(labels))
    preds = preds - apply(preds, 1, max)
    prob = exp(preds) / rowSums(exp(preds))
    
    return(list(name = "error",
                value = -mean(log(prob[cbind(1:length(labels), labels + 1)])),
                higher_better = FALSE))
    
}

model_custom <- lgb.train(list(),
                          dtrain,
                          100,
                          valids,
                          min_data = 1,
                          learning_rate = 1,
                          early_stopping_rounds = 10,
                          objective = custom_multiclass_obj,
                          eval = custom_multiclass_metric,
                          num_class = 3)

preds_custom <- predict(model_custom, test[, 1:4], rawscore = TRUE)

# compare predictions
identical(preds_builtin, preds_custom)

