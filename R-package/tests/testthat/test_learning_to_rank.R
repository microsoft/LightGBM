context("Learning to rank")

test_that("learning-to-rank with lgb.train() works as expected", {
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
                lengths = c(rep(500.0, 12L), 513.0)
                , values = seq_len(13L) * 2.0
            )
        )
    )
    params <- list(
        objective = "lambdarank"
        , metric = "ndcg"
        , ndcg_at = "1,2,3"
        , metric_freq = 1L
        , max_position = 3L
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
    expect_equal(dumped_model[["objective"]], "lambdarank")
    expect_equal(dumped_model[["max_feature_idx"]], ncol(train$data) - 1L)
})

test_that("learning-to-rank with lgb.cv() works as expected", {
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
                lengths = c(rep(500.0, 12L), 513.0)
                , values = seq_len(13L) * 2.0
            )
        )
    )
    params <- list(
        objective = "lambdarank"
        , metric = "ndcg"
        , ndcg_at = "1,2,3"
        , metric_freq = 1L
        , max_position = 3L
        , label_gain = "0,1,3,7"
    )
    nfold <- 4L
    model <- lgb.cv(
        params = params
        , data = dtrain
        , nrounds = 10L
        , nfold = nfold
        , min_data = 1L
        , learning_rate = 1.0
        , early_stopping_rounds = 10L
    )
    expect_is(model, "lgb.CVBooster")
    expect_equal(length(model$boosters), nfold)

    # check details of each booster
    for (bst in model$boosters) {
        dumped_model <- jsonlite::fromJSON(
            bst$booster$dump_model()
        )
        expect_equal(dumped_model[["objective"]], "lambdarank")
        expect_equal(dumped_model[["max_feature_idx"]], ncol(train$data) - 1L)
    }
})
