context("lgb.plot.interpretation")

.sigmoid <- function(x) {
    1 / (1 + exp(-x))
}
.logit <- function(x) {
    log(x / (1 - x))
}

test_that("lgb.plot.interepretation works as expected for binary classification", {
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
    expect_true({
        lgb.plot.interpretation(
            tree_interpretation_dt = tree_interpretation[[1]]
            , top_n = 5
        )
        TRUE
    })

    # should also work when you explicitly pass cex
    plot_res <- lgb.plot.interpretation(
        tree_interpretation_dt = tree_interpretation[[1]]
        , top_n = 5
        , cex = 0.95
    )
    expect_null(plot_res)
})

test_that("lgb.plot.interepretation works as expected for multiclass classification", {
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
    plot_res <- lgb.plot.interpretation(
        tree_interpretation_dt = tree_interpretation[[1]]
        , top_n = 5
    )
    expect_null(plot_res)
})
