context("Learning to rank")

test_that("learning-to-rank with lgb.train() works as expected", {
    data(agaricus.train, package = "lightgbm")
    # just keep a few features,to generate an model with imperfect fit
    train <- agaricus.train
    train_data <- train$data[, 1:20]
    dtrain <- lgb.Dataset(
        train_data
        , label = train$label
        , group = c(rep(4L, 1500L), 513L)
    )
    ndcg_at <- "1,2,3"
    eval_names <-  paste0("ndcg@", strsplit(ndcg_at, ",")[[1]])
    params <- list(
        objective = "lambdarank"
        , metric = "ndcg"
        , ndcg_at = ndcg_at
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
    expect_equal(dumped_model[["max_feature_idx"]], ncol(train_data) - 1L)

    # check that evaluation results make sense (0.0 < nDCG < 1.0)
    eval_results <- model$eval_train()
    expect_equal(length(eval_results), length(eval_names))
    for (result in eval_results){
        expect_true(result[["value"]] > 0.0 && result[["value"]] < 1.0)
        expect_true(result[["higher_better"]])
        expect_identical(result[["data_name"]], "training")
    }
    expect_identical(sapply(eval_results, function(x){x$name}), eval_names)
})

test_that("learning-to-rank with lgb.cv() works as expected", {
    data(agaricus.train, package = "lightgbm")
    # just keep a few features,to generate an model with imperfect fit
    train <- agaricus.train
    train_data <- train$data[, 1:20]
    dtrain <- lgb.Dataset(
        train_data
        , label = train$label
        , group = c(rep(4L, 1500L), 513L)
    )
    ndcg_at <- "1,2,3"
    eval_names <-  paste0("ndcg@", strsplit(ndcg_at, ",")[[1]])
    params <- list(
        objective = "lambdarank"
        , metric = "ndcg"
        , ndcg_at = ndcg_at
        , metric_freq = 1L
        , max_position = 3L
        , label_gain = "0,1,3,7"
    )
    nfold <- 4L
    nrounds <- 10L
    cv_bst <- lgb.cv(
        params = params
        , data = dtrain
        , nrounds = nrounds
        , nfold = nfold
        , min_data = 1L
        , learning_rate = 0.01
    )
    expect_is(cv_bst, "lgb.CVBooster")
    expect_equal(length(cv_bst$boosters), nfold)

    # "valid" should contain results for each metric
    eval_results <- cv_bst$record_evals[["valid"]]
    eval_names <-  c("ndcg@1", "ndcg@2", "ndcg@3")
    expect_identical(names(eval_results), eval_names)

    # check that best score and iter make sense (0.0 < nDCG < 1.0)
    best_iter <- cv_bst$best_iter
    best_score <- cv_bst$best_score
    expect_true(best_iter > 0L && best_iter <= nrounds)
    expect_true(best_score > 0.0 && best_score < 1.0)

    # best_score should be set for the first metric
    first_metric <- eval_names[[1L]]
    expect_equal(best_score, eval_results[[first_metric]][["eval"]][[best_iter]])

    for (eval_name in eval_names){
        results_for_this_metric <- eval_results[[eval_name]]

        # each set of metrics should have eval and eval_err
        expect_identical(names(results_for_this_metric), c("eval", "eval_err"))

        # there should be one "eval" and "eval_err" per round
        expect_equal(length(results_for_this_metric[["eval"]]), nrounds)
        expect_equal(length(results_for_this_metric[["eval_err"]]), nrounds)

        # check that evaluation results make sense (0.0 < nDCG < 1.0)
        all_evals <- unlist(results_for_this_metric[["eval"]])
        expect_true(all(all_evals > 0.0 && all_evals < 1.0))
    }

    # check details of each booster
    for (bst in cv_bst$boosters) {
        dumped_model <- jsonlite::fromJSON(
            bst$booster$dump_model()
        )
        expect_equal(dumped_model[["objective"]], "lambdarank")
        expect_equal(dumped_model[["max_feature_idx"]], ncol(train_data) - 1L)
    }
})
