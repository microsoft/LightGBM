context("lgb.interpete")

.sigmoid <- function(x){
    1 / (1 + exp(-x))
}
.logit <- function(x){
    log(x / (1 - x))
}

test_that("lgb.intereprete works as expected for binary classification", {
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(train$data, label = train$label)
    setinfo(
        dataset = dtrain
        , "init_score"
        , rep(
            .logit(mean(train$label))
            , length(train$label)
        )
    )
    data(agaricus.test, package = "lightgbm")
    test <- agaricus.test
    params <- list(
        objective = "binary"
        , learning_rate = 0.01
        , num_leaves = 63
        , max_depth = -1
        , min_data_in_leaf = 1
        , min_sum_hessian_in_leaf = 1
    )
    model <- lgb.train(
        params = params
        , data = dtrain
        , nrounds = 10
    )
    num_trees <- 5
    tree_interpretation <- lgb.interprete(
        model = model
        , data = test$data
        , idxset = 1:num_trees
    )
    expect_true(methods::is(tree_interpretation, "list"))
    expect_true(length(tree_interpretation) == num_trees)
    expect_null(names(tree_interpretation))
    expect_true(all(
        sapply(
            X = tree_interpretation
            , FUN = function(treeDT){
                checks <- c(
                    data.table::is.data.table(treeDT)
                    , identical(names(treeDT), c("Feature", "Contribution"))
                    , is.character(treeDT[, Feature])
                    , is.numeric(treeDT[, Contribution])
                )
                return(all(checks))
            }
        )
    ))
})

test_that("lgb.intereprete works as expected for multiclass classification", {
    data(iris)

    # We must convert factors to numeric
    # They must be starting from number 0 to use multiclass
    # For instance: 0, 1, 2, 3, 4, 5...
    iris$Species <- as.numeric(as.factor(iris$Species)) - 1

    # Create imbalanced training data (20, 30, 40 examples for classes 0, 1, 2)
    train <- as.matrix(iris[c(1:20, 51:80, 101:140), ])
    # The 10 last samples of each class are for validation
    test <- as.matrix(iris[c(41:50, 91:100, 141:150), ])
    dtrain <- lgb.Dataset(data = train[, 1:4], label = train[, 5])
    dtest <- lgb.Dataset.create.valid(dtrain, data = test[, 1:4], label = test[, 5])
    params <- list(
        objective = "multiclass"
        , metric = "multi_logloss"
        , num_class = 3
        , learning_rate = 0.00001
    )
    model <- lgb.train(
        params = params
        , data = dtrain
        , nrounds = 10
        , min_data = 1
    )
    num_trees <- 5
    tree_interpretation <- lgb.interprete(
        model = model
        , data = test[, 1:4]
        , idxset = 1:num_trees
    )
    expect_true(methods::is(tree_interpretation, "list"))
    expect_true(length(tree_interpretation) == num_trees)
    expect_null(names(tree_interpretation))
    expect_true(all(
        sapply(
            X = tree_interpretation
            , FUN = function(treeDT){
                checks <- c(
                    data.table::is.data.table(treeDT)
                    , identical(names(treeDT), c("Feature", "Class 0", "Class 1", "Class 2"))
                    , is.character(treeDT[, Feature])
                    , is.numeric(treeDT[, `Class 0`])
                    , is.numeric(treeDT[, `Class 1`])
                    , is.numeric(treeDT[, `Class 2`])
                )
                return(all(checks))
            }
        )
    ))
})
