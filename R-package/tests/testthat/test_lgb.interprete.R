context("lgb.interpete")

.sigmoid <- function(x){
    1 / (1 + exp(-x))
}
.logit <- function(x){
    log(x / (1 - x))
}
data(agaricus.train, package = "lightgbm")
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label)

test_that("lgb.intereprete works as expected", {
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
                check1 <- data.table::is.data.table(treeDT)
                check2 <- identical(names(treeDT), c("Feature", "Contribution"))
                check3 <- is.character(treeDT[, Feature])
                check4 <- is.numeric(treeDT[, Contribution])
                all(c(check1, check2, check3, check4))
            }
        )
    ))
})
