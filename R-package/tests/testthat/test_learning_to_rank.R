context("Learning to rank")

test_that("learning-to-rank with lgb.train() LambdaRank works as expected", {
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(
        train$data
        , label = train$label
        , group = c(rep(500L, 12L), 513L)
        , weight = inverse.rle(
            list(
                lengths = c(rep(500.0, 12L), 513.0)
                , values = seq_len(13L)
            )
        )
        , init_score = inverse.rle(
            list(
                lengths = c(rep(500.0, 12L), 513)
                , values = seq_len(13L) * 2
            )
        )
    )
    params <- list(
        objective = "lambdarank"
        , metric = "ndcg"
        , ndcg_at = "1,2,3"
        , metric_freq = 1
        , max_position = 3
        , label_gain = "0,1,3,7"
    )
    model <- lgb.train(
        params = params
        , data = dtrain
        , nrounds = 10L
    )
    expect_true(lgb.is.Booster(model))

    dumped_model <- jsonlite::fromJSON(
        model$dump_model()
    )
    expect_equal(dumped_model[['objective']], 'lambdarank')
    expect_equal(fitted_model[['max_feature_idx']], ncol(train$data) - 1)
})

test_that("learning-to-rank with lgb.cv() LambdaRank works as expected", {
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(
        train$data
        , label = train$label
        , group = c(rep(500L, 12L), 513L)
        , weight = inverse.rle(
            list(
                lengths = c(rep(500.0, 12L), 513.0)
                , values = seq_len(13L)
            )
        )
        , init_score = inverse.rle(
            list(
                lengths = c(rep(500.0, 12L), 513)
                , values = seq_len(13L) * 2
            )
        )
    )
    params <- list(
        objective = "lambdarank"
        , metric = "ndcg"
        , ndcg_at = "1,2,3"
        , metric_freq = 1
        , max_position = 3
        , label_gain = "0,1,3,7"
    )
    nfold <- 4
    model <- lgb.cv(
        params = params
        , data = dtrain
        , nrounds = 10
        , nfold = nfold
        , min_data = 1
        , learning_rate = 1
        , early_stopping_rounds = 10
    )
    expect_is(model, 'lgb.CVBooster')
    expect_equal(length(model$boosters), nfold)

    # check details of each booster
    for (bst in model$boosters){
        dumped_model <- jsonlite::fromJSON(
            bst$booster$dump_model()
        )
        expect_equal(dumped_model[['objective']], 'lambdarank')
        expect_equal(dumped_model[['max_feature_idx']], ncol(train$data) - 1)
    }
})
