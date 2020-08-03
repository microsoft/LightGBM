context("lgb.get.eval.result")

test_that("lgb.get.eval.result() should throw an informative error if booster is not an lgb.Booster", {
    bad_inputs <- list(
        matrix(1.0:10.0, 2L, 5L)
        , TRUE
        , c("a", "b")
        , NA
        , 10L
        , lgb.Dataset(
            data = matrix(1.0:10.0, 2L, 5L)
            , params = list()
        )
    )
    for (bad_input in bad_inputs) {
        expect_error({
            lgb.get.eval.result(
                booster = bad_input
                , data_name = "test"
                , eval_name = "l2"
            )
        }, regexp = "Can only use", fixed = TRUE)
    }
})

test_that("lgb.get.eval.result() should throw an informative error for incorrect data_name", {
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    dtrain <- lgb.Dataset(
        agaricus.train$data
        , label = agaricus.train$label
    )
    model <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
        )
        , data = dtrain
        , nrounds = 5L
        , valids = list(
            "test" = lgb.Dataset.create.valid(
                dtrain
                , agaricus.test$data
                , label = agaricus.test$label
            )
        )
        , min_data = 1L
        , learning_rate = 1.0
    )
    expect_error({
        eval_results <- lgb.get.eval.result(
            booster = model
            , data_name = "testing"
            , eval_name = "l2"
        )
    }, regexp = "Only the following datasets exist in record evals: [test]", fixed = TRUE)
})

test_that("lgb.get.eval.result() should throw an informative error for incorrect eval_name", {
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    dtrain <- lgb.Dataset(
        agaricus.train$data
        , label = agaricus.train$label
    )
    model <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
        )
        , data = dtrain
        , nrounds = 5L
        , valids = list(
            "test" = lgb.Dataset.create.valid(
                dtrain
                , agaricus.test$data
                , label = agaricus.test$label
            )
        )
        , min_data = 1L
        , learning_rate = 1.0
    )
    expect_error({
        eval_results <- lgb.get.eval.result(
            booster = model
            , data_name = "test"
            , eval_name = "l1"
        )
    }, regexp = "Only the following eval_names exist for dataset.*\\: \\[l2\\]", fixed = FALSE)
})

context("lgb.load()")

test_that("lgb.load() gives the expected error messages given different incorrect inputs", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    train <- agaricus.train
    test <- agaricus.test
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , num_leaves = 4L
        , learning_rate = 1.0
        , nrounds = 2L
        , objective = "binary"
        , save_name = tempfile(fileext = ".model")
    )

    # you have to give model_str or filename
    expect_error({
        lgb.load()
    }, regexp = "either filename or model_str must be given")
    expect_error({
        lgb.load(filename = NULL, model_str = NULL)
    }, regexp = "either filename or model_str must be given")

    # if given, filename should be a string that points to an existing file
    model_file <- tempfile(fileext = ".model")
    expect_error({
        lgb.load(filename = list(model_file))
    }, regexp = "filename should be character")
    file_to_check <- paste0("a.model")
    while (file.exists(file_to_check)) {
        file_to_check <- paste0("a", file_to_check)
    }
    expect_error({
        lgb.load(filename = file_to_check)
    }, regexp = "passed to filename does not exist")

    # if given, model_str should be a string
    expect_error({
        lgb.load(model_str = c(4.0, 5.0, 6.0))
    }, regexp = "model_str should be character")

})

test_that("Loading a Booster from a file works", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    train <- agaricus.train
    test <- agaricus.test
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , num_leaves = 4L
        , learning_rate = 1.0
        , nrounds = 2L
        , objective = "binary"
        , save_name = tempfile(fileext = ".model")
    )
    expect_true(lgb.is.Booster(bst))

    pred <- predict(bst, test$data)
    model_file <- tempfile(fileext = ".model")
    lgb.save(bst, model_file)

    # finalize the booster and destroy it so you know we aren't cheating
    bst$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)

    bst2 <- lgb.load(
        filename = model_file
    )
    pred2 <- predict(bst2, test$data)
    expect_identical(pred, pred2)
})

test_that("Loading a Booster from a string works", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    train <- agaricus.train
    test <- agaricus.test
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , num_leaves = 4L
        , learning_rate = 1.0
        , nrounds = 2L
        , objective = "binary"
        , save_name = tempfile(fileext = ".model")
    )
    expect_true(lgb.is.Booster(bst))

    pred <- predict(bst, test$data)
    model_string <- bst$save_model_to_string()

    # finalize the booster and destroy it so you know we aren't cheating
    bst$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)

    bst2 <- lgb.load(
        model_str = model_string
    )
    pred2 <- predict(bst2, test$data)
    expect_identical(pred, pred2)
})

test_that("If a string and a file are both passed to lgb.load() the file is used model_str is totally ignored", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    train <- agaricus.train
    test <- agaricus.test
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , num_leaves = 4L
        , learning_rate = 1.0
        , nrounds = 2L
        , objective = "binary"
        , save_name = tempfile(fileext = ".model")
    )
    expect_true(lgb.is.Booster(bst))

    pred <- predict(bst, test$data)
    model_file <- tempfile(fileext = ".model")
    lgb.save(bst, model_file)

    # finalize the booster and destroy it so you know we aren't cheating
    bst$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)

    bst2 <- lgb.load(
        filename = model_file
        , model_str = 4.0
    )
    pred2 <- predict(bst2, test$data)
    expect_identical(pred, pred2)
})

context("Booster")

test_that("Creating a Booster from a Dataset should work", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    dtrain <- lgb.Dataset(
        agaricus.train$data
        , label = agaricus.train$label
    )
    bst <- Booster$new(
        params = list(
            objective = "binary"
        ),
        train_set = dtrain
    )
    expect_true(lgb.is.Booster(bst))
    expect_equal(bst$current_iter(), 0L)
    expect_true(is.na(bst$best_score))
    expect_true(all(bst$predict(agaricus.train$data) == 0.5))
})

test_that("Creating a Booster from a Dataset with an existing predictor should work", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    nrounds <- 2L
    bst <- lightgbm(
        data = as.matrix(agaricus.train$data)
        , label = agaricus.train$label
        , num_leaves = 4L
        , learning_rate = 1.0
        , nrounds = nrounds
        , objective = "binary"
        , save_name = tempfile(fileext = ".model")
    )
    data(agaricus.test, package = "lightgbm")
    dtest <- Dataset$new(
        data = agaricus.test$data
        , label = agaricus.test$label
        , predictor = bst$to_predictor()
    )
    bst_from_ds <- Booster$new(
        train_set = dtest
    )
    expect_true(lgb.is.Booster(bst))
    expect_equal(bst$current_iter(), nrounds)
    expect_equal(bst$eval_train()[[1L]][["value"]], 0.1115352)
    expect_equal(bst_from_ds$current_iter(), nrounds)
    dumped_model <- jsonlite::fromJSON(bst$dump_model())
    expect_identical(bst_from_ds$eval_train(), list())
    expect_equal(bst_from_ds$current_iter(), nrounds)
})

test_that("Booster$rollback_one_iter() should work as expected", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    train <- agaricus.train
    test <- agaricus.test
    nrounds <- 5L
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , num_leaves = 4L
        , learning_rate = 1.0
        , nrounds = nrounds
        , objective = "binary"
        , save_name = tempfile(fileext = ".model")
    )
    expect_equal(bst$current_iter(), nrounds)
    expect_true(lgb.is.Booster(bst))
    logloss <- bst$eval_train()[[1L]][["value"]]
    expect_equal(logloss, 0.01904786)

    x <- bst$rollback_one_iter()

    # rollback_one_iter() should return a booster and modify the original
    # booster in place
    expect_true(lgb.is.Booster(x))
    expect_equal(bst$current_iter(), nrounds - 1L)

    # score should now come from the model as of 4 iterations
    logloss <- bst$eval_train()[[1L]][["value"]]
    expect_equal(logloss, 0.027915146)
})

test_that("Booster$update() passing a train_set works as expected", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    nrounds <- 2L

    # train with 2 rounds and then update
    bst <- lightgbm(
        data = as.matrix(agaricus.train$data)
        , label = agaricus.train$label
        , num_leaves = 4L
        , learning_rate = 1.0
        , nrounds = nrounds
        , objective = "binary"
        , save_name = tempfile(fileext = ".model")
    )
    expect_true(lgb.is.Booster(bst))
    expect_equal(bst$current_iter(), nrounds)
    bst$update(
        train_set = Dataset$new(
            data = agaricus.train$data
            , label = agaricus.train$label
        )
    )
    expect_true(lgb.is.Booster(bst))
    expect_equal(bst$current_iter(), nrounds + 1L)

    # train with 3 rounds directly
    bst2 <- lightgbm(
        data = as.matrix(agaricus.train$data)
        , label = agaricus.train$label
        , num_leaves = 4L
        , learning_rate = 1.0
        , nrounds = nrounds +  1L
        , objective = "binary"
        , save_name = tempfile(fileext = ".model")
    )
    expect_true(lgb.is.Booster(bst2))
    expect_equal(bst2$current_iter(), nrounds +  1L)

    # model with 2 rounds + 1 update should be identical to 3 rounds
    expect_equal(bst2$eval_train()[[1L]][["value"]], 0.04806585)
    expect_equal(bst$eval_train()[[1L]][["value"]], bst2$eval_train()[[1L]][["value"]])
})

test_that("Booster$update() throws an informative error if you provide a non-Dataset to update()", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    nrounds <- 2L

    # train with 2 rounds and then update
    bst <- lightgbm(
        data = as.matrix(agaricus.train$data)
        , label = agaricus.train$label
        , num_leaves = 4L
        , learning_rate = 1.0
        , nrounds = nrounds
        , objective = "binary"
        , save_name = tempfile(fileext = ".model")
    )
    expect_error({
        bst$update(
            train_set = data.frame(x = rnorm(10L))
        )
    }, regexp = "lgb.Booster.update: Only can use lgb.Dataset", fixed = TRUE)
})

context("save_model")

test_that("Saving a model with different feature importance types works", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , num_leaves = 4L
        , learning_rate = 1.0
        , nrounds = 2L
        , objective = "binary"
        , save_name = tempfile(fileext = ".model")
    )
    expect_true(lgb.is.Booster(bst))

    .feat_importance_from_string <- function(model_string) {
        file_lines <- strsplit(model_string, "\n")[[1L]]
        start_indx <- which(grepl("^feature_importances\\:$", file_lines)) + 1L
        blank_line_indices <- which(file_lines == "")
        end_indx <- blank_line_indices[blank_line_indices > start_indx][1L] - 1L
        importances <- file_lines[start_indx: end_indx]
        return(importances)
    }

    GAIN_IMPORTANCE <- 1L
    model_string <- bst$save_model_to_string(feature_importance_type = GAIN_IMPORTANCE)
    expect_equal(
        .feat_importance_from_string(model_string)
        , c(
            "odor=none=4010"
            , "stalk-root=club=1163"
            , "stalk-root=rooted=573"
            , "stalk-surface-above-ring=silky=450"
            , "spore-print-color=green=397"
            , "gill-color=buff=281"
        )
    )

    SPLIT_IMPORTANCE <- 0L
    model_string <- bst$save_model_to_string(feature_importance_type = SPLIT_IMPORTANCE)
    expect_equal(
        .feat_importance_from_string(model_string)
        , c(
            "odor=none=1"
            , "gill-color=buff=1"
            , "stalk-root=club=1"
            , "stalk-root=rooted=1"
            , "stalk-surface-above-ring=silky=1"
            , "spore-print-color=green=1"
        )
    )

    UNSUPPORTED_IMPORTANCE <- 2L
    expect_error({
        model_string <- bst$save_model_to_string(feature_importance_type = UNSUPPORTED_IMPORTANCE)
    }, "Unknown importance type")
})
