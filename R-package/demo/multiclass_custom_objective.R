library(lightgbm)

# We load the default iris dataset shipped with R
data(iris)

# We must convert factors to numeric
# They must be starting from number 0 to use multiclass
# For instance: 0, 1, 2, 3, 4, 5...
iris$Species <- as.numeric(as.factor(iris$Species)) - 1L

# Create imbalanced training data (20, 30, 40 examples for classes 0, 1, 2)
train <- as.matrix(iris[c(1L:20L, 51L:80L, 101L:140L), ])
# The 10 last samples of each class are for validation
test <- as.matrix(iris[c(41L:50L, 91L:100L, 141L:150L), ])

dtrain <- lgb.Dataset(data = train[, 1L:4L], label = train[, 5L])
dtest <- lgb.Dataset.create.valid(dtrain, data = test[, 1L:4L], label = test[, 5L])
valids <- list(train = dtrain, test = dtest)

# Method 1 of training with built-in multiclass objective
# Note: need to turn off boost from average to match custom objective
# (https://github.com/microsoft/LightGBM/issues/1846)
model_builtin <- lgb.train(
    list()
    , dtrain
    , boost_from_average = FALSE
    , 100L
    , valids
    , min_data = 1L
    , learning_rate = 1.0
    , early_stopping_rounds = 10L
    , objective = "multiclass"
    , metric = "multi_logloss"
    , num_class = 3L
)

preds_builtin <- predict(model_builtin, test[, 1L:4L], rawscore = TRUE, reshape = TRUE)
probs_builtin <- exp(preds_builtin) / rowSums(exp(preds_builtin))

# Method 2 of training with custom objective function

# User defined objective function, given prediction, return gradient and second order gradient
custom_multiclass_obj <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")

    # preds is a matrix with rows corresponding to samples and columns corresponding to choices
    preds <- matrix(preds, nrow = length(labels))

    # to prevent overflow, normalize preds by row
    preds <- preds - apply(preds, MARGIN = 1L, max)
    prob <- exp(preds) / rowSums(exp(preds))

    # compute gradient
    grad <- prob
    subset_index <- as.matrix(
        data.frame(
            seq_len(length(labels))
            , labels + 1L
            , fix.empty.names = FALSE
        )
        , nrow = length(labels)
        , dimnames = NULL
    )
    grad[subset_index] <- grad[subset_index] - 1L

    # compute hessian (approximation)
    hess <- 2.0 * prob * (1.0 - prob)

    return(list(grad = grad, hess = hess))
}

# define custom metric
custom_multiclass_metric <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    preds <- matrix(preds, nrow = length(labels))
    preds <- preds - apply(preds, 1L, max)
    prob <- exp(preds) / rowSums(exp(preds))

    subset_index <- as.matrix(
        data.frame(
            seq_len(length(labels))
            , labels + 1L
            , fix.empty.names = FALSE
        )
        , nrow = length(labels)
        , dimnames = NULL
    )
    return(list(
        name = "error"
        , value = -mean(log(prob[subset_index]))
        , higher_better = FALSE
    ))
}

model_custom <- lgb.train(
    list()
    , dtrain
    , 100L
    , valids
    , min_data = 1L
    , learning_rate = 1.0
    , early_stopping_rounds = 10L
    , objective = custom_multiclass_obj
    , eval = custom_multiclass_metric
    , num_class = 3L
)

preds_custom <- predict(model_custom, test[, 1L:4L], rawscore = TRUE, reshape = TRUE)
probs_custom <- exp(preds_custom) / rowSums(exp(preds_custom))

# compare predictions
stopifnot(identical(probs_builtin, probs_custom))
stopifnot(identical(preds_builtin, preds_custom))
