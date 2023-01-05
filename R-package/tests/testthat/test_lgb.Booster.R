VERBOSITY <- as.integer(
  Sys.getenv("LIGHTGBM_TEST_VERBOSITY", "-1")
)

ON_WINDOWS <- .Platform$OS.type == "windows"
TOLERANCE <- 1e-6

test_that("Booster$finalize() should not fail", {
    X <- as.matrix(as.integer(iris[, "Species"]), ncol = 1L)
    y <- iris[["Sepal.Length"]]
    dtrain <- lgb.Dataset(X, label = y)
    bst <- lgb.train(
        data = dtrain
        , params = list(
            objective = "regression"
        )
        , verbose = VERBOSITY
        , nrounds = 3L
    )
    expect_true(lgb.is.Booster(bst))

    expect_false(lgb.is.null.handle(bst$.__enclos_env__$private$handle))

    bst$finalize()
    expect_true(lgb.is.null.handle(bst$.__enclos_env__$private$handle))

    # calling finalize() a second time shouldn't cause any issues
    bst$finalize()
    expect_true(lgb.is.null.handle(bst$.__enclos_env__$private$handle))
})

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
            , min_data = 1L
            , learning_rate = 1.0
            , verbose = VERBOSITY
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
            , min_data = 1L
            , learning_rate = 1.0
            , verbose = VERBOSITY
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
    )
    expect_error({
        eval_results <- lgb.get.eval.result(
            booster = model
            , data_name = "test"
            , eval_name = "l1"
        )
    }, regexp = "Only the following eval_names exist for dataset.*\\: \\[l2\\]", fixed = FALSE)
})

test_that("lgb.load() gives the expected error messages given different incorrect inputs", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    train <- agaricus.train
    test <- agaricus.test
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , params = list(
            objective = "binary"
            , num_leaves = 4L
            , learning_rate = 1.0
            , verbose = VERBOSITY
        )
        , nrounds = 2L
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
    }, regexp = "lgb.load: model_str should be a character/raw vector")

})

test_that("Loading a Booster from a text file works", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    train <- agaricus.train
    test <- agaricus.test
    params <- list(
        num_leaves = 4L
        , boosting = "rf"
        , bagging_fraction = 0.8
        , bagging_freq = 1L
        , boost_from_average = FALSE
        , categorical_feature = c(1L, 2L)
        , interaction_constraints = list(c(1L, 2L), 1L)
        , feature_contri = rep(0.5, ncol(train$data))
        , metric = c("mape", "average_precision")
        , learning_rate = 1.0
        , objective = "binary"
        , verbosity = VERBOSITY
    )
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , params = params
        , nrounds = 2L
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

    # check that the parameters are loaded correctly
    expect_equal(bst2$params[names(params)], params)
})

test_that("boosters with linear models at leaves can be written to text file and re-loaded successfully", {
    X <- matrix(rnorm(100L), ncol = 1L)
    labels <- 2L * X + runif(nrow(X), 0L, 0.1)
    dtrain <- lgb.Dataset(
        data = X
        , label = labels
    )

    params <- list(
        objective = "regression"
        , verbose = -1L
        , metric = "mse"
        , seed = 0L
        , num_leaves = 2L
    )

    bst <- lgb.train(
        data = dtrain
        , nrounds = 10L
        , params = params
        , verbose = VERBOSITY
    )
    expect_true(lgb.is.Booster(bst))

    # save predictions, then write the model to a file and destroy it in R
    preds <- predict(bst, X)
    model_file <- tempfile(fileext = ".model")
    lgb.save(bst, model_file)
    bst$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)

    # load the booster and make predictions...should be the same
    bst2 <- lgb.load(
        filename = model_file
    )
    preds2 <- predict(bst2, X)
    expect_identical(preds, preds2)
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
        , params = list(
            num_leaves = 4L
            , learning_rate = 1.0
            , objective = "binary"
            , verbose = VERBOSITY
        )
        , nrounds = 2L
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

test_that("Saving a large model to string should work", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , params = list(
            num_leaves = 100L
            , learning_rate = 0.01
            , objective = "binary"
        )
        , nrounds = 500L
        , verbose = VERBOSITY
    )

    pred <- predict(bst, train$data)
    pred_leaf_indx <- predict(bst, train$data, type = "leaf")
    pred_raw_score <- predict(bst, train$data, type = "raw")
    model_string <- bst$save_model_to_string()

    # make sure this test is still producing a model bigger than the default
    # buffer size used in LGBM_BoosterSaveModelToString_R
    expect_gt(nchar(model_string), 1024L * 1024L)

    # finalize the booster and destroy it so you know we aren't cheating
    bst$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)

    # make sure a new model can be created from this string, and that it
    # produces expected results
    bst2 <- lgb.load(
        model_str = model_string
    )
    pred2 <- predict(bst2, train$data)
    pred2_leaf_indx <- predict(bst2, train$data, type = "leaf")
    pred2_raw_score <- predict(bst2, train$data, type = "raw")
    expect_identical(pred, pred2)
    expect_identical(pred_leaf_indx, pred2_leaf_indx)
    expect_identical(pred_raw_score, pred2_raw_score)
})

test_that("Saving a large model to JSON should work", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , params = list(
            num_leaves = 100L
            , learning_rate = 0.01
            , objective = "binary"
        )
        , nrounds = 200L
        , verbose = VERBOSITY
    )

    model_json <- bst$dump_model()

    # make sure this test is still producing a model bigger than the default
    # buffer size used in LGBM_BoosterDumpModel_R
    expect_gt(nchar(model_json), 1024L * 1024L)

    # check that it is valid JSON that looks like a LightGBM model
    model_list <- jsonlite::fromJSON(model_json)
    expect_equal(model_list[["objective"]], "binary sigmoid:1")
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
        , params = list(
            num_leaves = 4L
            , learning_rate = 1.0
            , objective = "binary"
            , verbose = VERBOSITY
        )
        , nrounds = 2L
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
            , verbose = VERBOSITY
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
        , params = list(
            num_leaves = 4L
            , learning_rate = 1.0
            , objective = "binary"
            , verbose = VERBOSITY
        )
        , nrounds = nrounds
    )
    data(agaricus.test, package = "lightgbm")
    dtest <- Dataset$new(
        data = agaricus.test$data
        , label = agaricus.test$label
        , predictor = bst$to_predictor()
    )
    bst_from_ds <- Booster$new(
        train_set = dtest
        , params = list(
            verbose = VERBOSITY
        )
    )
    expect_true(lgb.is.Booster(bst))
    expect_equal(bst$current_iter(), nrounds)
    expect_equal(bst$eval_train()[[1L]][["value"]], 0.1115352)
    expect_true(lgb.is.Booster(bst_from_ds))
    expect_equal(bst_from_ds$current_iter(), nrounds)
    expect_equal(bst_from_ds$eval_train()[[1L]][["value"]], 5.65704892)
    dumped_model <- jsonlite::fromJSON(bst$dump_model())
})

test_that("Booster$eval() should work on a Dataset stored in a binary file", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(train$data, label = train$label)

    bst <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
            , num_leaves = 4L
            , verbose = VERBOSITY
        )
        , data = dtrain
        , nrounds = 2L
    )

    data(agaricus.test, package = "lightgbm")
    test <- agaricus.test
    dtest <- lgb.Dataset.create.valid(
        dataset = dtrain
        , data = test$data
        , label = test$label
    )
    dtest$construct()

    eval_in_mem <- bst$eval(
        data = dtest
        , name = "test"
    )

    test_file <- tempfile(pattern = "lgb.Dataset_")
    lgb.Dataset.save(
        dataset = dtest
        , fname = test_file
    )
    rm(dtest)

    eval_from_file <- bst$eval(
        data = lgb.Dataset(
            data = test_file
            , params = list(verbose = VERBOSITY)
        )$construct()
        , name = "test"
    )

    expect_true(abs(eval_in_mem[[1L]][["value"]] - 0.1744423) < TOLERANCE)
    # refer to https://github.com/microsoft/LightGBM/issues/4680
    if (isTRUE(ON_WINDOWS)) {
      expect_equal(eval_in_mem, eval_from_file)
    } else {
      expect_identical(eval_in_mem, eval_from_file)
    }
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
        , params = list(
            num_leaves = 4L
            , learning_rate = 1.0
            , objective = "binary"
            , verbose = VERBOSITY
        )
        , nrounds = nrounds
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
        , params = list(
            num_leaves = 4L
            , learning_rate = 1.0
            , objective = "binary"
            , verbose = VERBOSITY
        )
        , nrounds = nrounds
    )
    expect_true(lgb.is.Booster(bst))
    expect_equal(bst$current_iter(), nrounds)
    bst$update(
        train_set = Dataset$new(
            data = agaricus.train$data
            , label = agaricus.train$label
            , params = list(verbose = VERBOSITY)
        )
    )
    expect_true(lgb.is.Booster(bst))
    expect_equal(bst$current_iter(), nrounds + 1L)

    # train with 3 rounds directly
    bst2 <- lightgbm(
        data = as.matrix(agaricus.train$data)
        , label = agaricus.train$label
        , params = list(
            num_leaves = 4L
            , learning_rate = 1.0
            , objective = "binary"
            , verbose = VERBOSITY
        )
        , nrounds = nrounds +  1L
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
        , params = list(
            num_leaves = 4L
            , learning_rate = 1.0
            , objective = "binary"
            , verbose = VERBOSITY
        )
        , nrounds = nrounds
    )
    expect_error({
        bst$update(
            train_set = data.frame(x = rnorm(10L))
        )
    }, regexp = "lgb.Booster.update: Only can use lgb.Dataset", fixed = TRUE)
})

test_that("Booster should store parameters and Booster$reset_parameter() should update them", {
    data(agaricus.train, package = "lightgbm")
    dtrain <- lgb.Dataset(
        agaricus.train$data
        , label = agaricus.train$label
    )
    # testing that this works for some cases that could break it:
    #    - multiple metrics
    #    - using "metric", "boosting", "num_class" in params
    params <- list(
        objective = "multiclass"
        , max_depth = 4L
        , bagging_fraction = 0.8
        , metric = c("multi_logloss", "multi_error")
        , boosting = "gbdt"
        , num_class = 5L
        , verbose = VERBOSITY
    )
    bst <- Booster$new(
        params = params
        , train_set = dtrain
    )
    expect_identical(bst$params, params)

    params[["bagging_fraction"]] <- 0.9
    ret_bst <- bst$reset_parameter(params = params)
    expect_identical(ret_bst$params, params)
    expect_identical(bst$params, params)
})

test_that("Booster$params should include dataset params, before and after Booster$reset_parameter()", {
    data(agaricus.train, package = "lightgbm")
    dtrain <- lgb.Dataset(
        agaricus.train$data
        , label = agaricus.train$label
        , params = list(
            max_bin = 17L
        )
    )
    params <- list(
        objective = "binary"
        , max_depth = 4L
        , bagging_fraction = 0.8
        , verbose = VERBOSITY
    )
    bst <- Booster$new(
        params = params
        , train_set = dtrain
    )
    expect_identical(
        bst$params
        , list(
            objective = "binary"
            , max_depth = 4L
            , bagging_fraction = 0.8
            , verbose = VERBOSITY
            , max_bin = 17L
        )
    )

    params[["bagging_fraction"]] <- 0.9
    ret_bst <- bst$reset_parameter(params = params)
    expected_params <- list(
        objective = "binary"
        , max_depth = 4L
        , bagging_fraction = 0.9
        , verbose = VERBOSITY
        , max_bin = 17L
    )
    expect_identical(ret_bst$params, expected_params)
    expect_identical(bst$params, expected_params)
})

test_that("Saving a model with different feature importance types works", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , params = list(
            num_leaves = 4L
            , learning_rate = 1.0
            , objective = "binary"
            , verbose = VERBOSITY
        )
        , nrounds = 2L
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
})

test_that("Saving a model with unknown importance type fails", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , params = list(
            num_leaves = 4L
            , learning_rate = 1.0
            , objective = "binary"
            , verbose = VERBOSITY
        )
        , nrounds = 2L
    )
    expect_true(lgb.is.Booster(bst))

    UNSUPPORTED_IMPORTANCE <- 2L
    expect_error({
        capture.output({
          model_string <- bst$save_model_to_string(
            feature_importance_type = UNSUPPORTED_IMPORTANCE
          )
        }, type = "message")
    }, "Unknown importance type")
})


.params_from_model_string <- function(model_str) {
    file_lines <- strsplit(model_str, "\n")[[1L]]
    start_indx <- which(grepl("^parameters\\:$", file_lines)) + 1L
    blank_line_indices <- which(file_lines == "")
    end_indx <- blank_line_indices[blank_line_indices > start_indx][1L] - 1L
    params <- file_lines[start_indx: end_indx]
    return(params)
}

test_that("all parameters are stored correctly with save_model_to_string()", {
    dtrain <- lgb.Dataset(
        data = matrix(rnorm(500L), nrow = 100L)
        , label = rnorm(100L)
    )
    nrounds <- 4L
    bst <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
        )
        , data = dtrain
        , nrounds = nrounds
        , verbose = VERBOSITY
    )

    model_str <- bst$save_model_to_string()
    params_in_file <- .params_from_model_string(model_str = model_str)

    # parameters should match what was passed from the R package
    expect_equal(sum(startsWith(params_in_file, "[metric:")), 1L)
    expect_equal(sum(params_in_file == "[metric: l2]"), 1L)

    expect_equal(sum(startsWith(params_in_file, "[num_iterations:")), 1L)
    expect_equal(sum(params_in_file == "[num_iterations: 4]"), 1L)

    expect_equal(sum(startsWith(params_in_file, "[objective:")), 1L)
    expect_equal(sum(params_in_file == "[objective: regression]"), 1L)

    expect_equal(sum(startsWith(params_in_file, "[verbosity:")), 1L)
    expect_equal(sum(params_in_file == sprintf("[verbosity: %i]", VERBOSITY)), 1L)

    # early stopping should be off by default
    expect_equal(sum(startsWith(params_in_file, "[early_stopping_round:")), 1L)
    expect_equal(sum(params_in_file == "[early_stopping_round: 0]"), 1L)
})

test_that("early_stopping, num_iterations are stored correctly in model string even with aliases", {
    dtrain <- lgb.Dataset(
        data = matrix(rnorm(500L), nrow = 100L)
        , label = rnorm(100L)
    )
    dvalid <- lgb.Dataset(
        data = matrix(rnorm(500L), nrow = 100L)
        , label = rnorm(100L)
    )

    # num_iterations values (all different)
    num_iterations <- 4L
    num_boost_round <- 2L
    n_iter <- 3L
    nrounds_kwarg <- 6L

    # early_stopping_round values (all different)
    early_stopping_round <- 2L
    early_stopping_round_kwarg <- 3L
    n_iter_no_change <- 4L

    params <- list(
        objective = "regression"
        , metric = "l2"
        , num_boost_round = num_boost_round
        , num_iterations = num_iterations
        , n_iter = n_iter
        , early_stopping_round = early_stopping_round
        , n_iter_no_change = n_iter_no_change
    )

    bst <- lgb.train(
        params = params
        , data = dtrain
        , nrounds = nrounds_kwarg
        , early_stopping_rounds = early_stopping_round_kwarg
        , valids = list(
            "random_valid" = dvalid
        )
        , verbose = VERBOSITY
    )

    model_str <- bst$save_model_to_string()
    params_in_file <- .params_from_model_string(model_str = model_str)

    # parameters should match what was passed from the R package, and the "main" (non-alias)
    # params values in `params` should be preferred to keyword argumentts or aliases
    expect_equal(sum(startsWith(params_in_file, "[num_iterations:")), 1L)
    expect_equal(sum(params_in_file == sprintf("[num_iterations: %s]", num_iterations)), 1L)
    expect_equal(sum(startsWith(params_in_file, "[early_stopping_round:")), 1L)
    expect_equal(sum(params_in_file == sprintf("[early_stopping_round: %s]", early_stopping_round)), 1L)

    # none of the aliases shouold have been written to the model file
    expect_equal(sum(startsWith(params_in_file, "[num_boost_round:")), 0L)
    expect_equal(sum(startsWith(params_in_file, "[n_iter:")), 0L)
    expect_equal(sum(startsWith(params_in_file, "[n_iter_no_change:")), 0L)

})

test_that("Booster: method calls Booster with a null handle should raise an informative error and not segfault", {
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(train$data, label = train$label)
    bst <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
            , num_leaves = 8L
        )
        , data = dtrain
        , verbose = VERBOSITY
        , nrounds = 5L
        , valids = list(
            train = dtrain
        )
        , serializable = FALSE
    )
    tmp_file <- tempfile(fileext = ".rds")
    saveRDS(bst, tmp_file)
    rm(bst)
    bst <- readRDS(tmp_file)
    .expect_booster_error <- function(object) {
        error_regexp <- "Attempting to use a Booster which no longer exists"
        expect_error(object, regexp = error_regexp)
    }
    .expect_booster_error({
        bst$current_iter()
    })
    .expect_booster_error({
        bst$dump_model()
    })
    .expect_booster_error({
        bst$eval(data = dtrain, name = "valid")
    })
    .expect_booster_error({
        bst$eval_train()
    })
    .expect_booster_error({
        bst$lower_bound()
    })
    .expect_booster_error({
        bst$predict(data = train$data[seq_len(5L), ])
    })
    .expect_booster_error({
        bst$reset_parameter(params = list(learning_rate = 0.123))
    })
    .expect_booster_error({
        bst$rollback_one_iter()
    })
    .expect_booster_error({
        bst$save_raw()
    })
    .expect_booster_error({
        bst$save_model(filename = tempfile(fileext = ".model"))
    })
    .expect_booster_error({
        bst$save_model_to_string()
    })
    .expect_booster_error({
        bst$update()
    })
    .expect_booster_error({
        bst$upper_bound()
    })
    predictor <- bst$to_predictor()
    .expect_booster_error({
        predictor$current_iter()
    })
    .expect_booster_error({
        predictor$predict(data = train$data[seq_len(5L), ])
    })
})

test_that("Booster$new() using a Dataset with a null handle should raise an informative error and not segfault", {
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(train$data, label = train$label)
    dtrain$construct()
    tmp_file <- tempfile(fileext = ".bin")
    saveRDS(dtrain, tmp_file)
    rm(dtrain)
    dtrain <- readRDS(tmp_file)
    expect_error({
        bst <- Booster$new(
            train_set = dtrain
            , params = list(
                verbose = VERBOSITY
            )
        )
    }, regexp = "Attempting to create a Dataset without any raw data")
})

test_that("Booster$new() raises informative errors for malformed inputs", {
  data(agaricus.train, package = "lightgbm")
  train <- agaricus.train
  dtrain <- lgb.Dataset(train$data, label = train$label)

  # no inputs
  expect_error({
    Booster$new()
  }, regexp = "lgb.Booster: Need at least either training dataset, model file, or model_str")

  # unrecognized objective
  expect_error({
    capture.output({
      Booster$new(
        params = list(objective = "not_a_real_objective")
        , train_set = dtrain
      )
    }, type = "message")
  }, regexp = "Unknown objective type name: not_a_real_objective")

  # train_set is not a Dataset
  expect_error({
    Booster$new(
      train_set = data.table::data.table(rnorm(1L:10L))
    )
  }, regexp = "lgb.Booster: Can only use lgb.Dataset as training data")

  # model file isn't a string
  expect_error({
    Booster$new(
      modelfile = list()
    )
  }, regexp = "lgb.Booster: Can only use a string as model file path")

  # model file doesn't exist
  expect_error({
    capture.output({
      Booster$new(
        params = list()
        , modelfile = "file-that-does-not-exist.model"
      )
    }, type = "message")
  }, regexp = "Could not open file-that-does-not-exist.model")

  # model file doesn't contain a valid LightGBM model
  model_file <- tempfile(fileext = ".model")
  writeLines(
    text = c("make", "good", "predictions")
    , con = model_file
  )
  expect_error({
    capture.output({
      Booster$new(
        params = list()
        , modelfile = model_file
      )
    }, type = "message")
  }, regexp = "Unknown model format or submodel type in model file")

  # malformed model string
  expect_error({
    capture.output({
      Booster$new(
        params = list()
        , model_str = "a\nb\n"
      )
    }, type = "message")
  }, regexp = "Model file doesn't specify the number of classes")

  # model string isn't character or raw
  expect_error({
    Booster$new(
      model_str = numeric()
    )
  }, regexp = "lgb.Booster: Can only use a character/raw vector as model_str")
})

# this is almost identical to the test above it, but for lgb.cv(). A lot of code
# is duplicated between lgb.train() and lgb.cv(), and this will catch cases where
# one is updated and the other isn't
test_that("lgb.cv() correctly handles passing through params to the model file", {
    dtrain <- lgb.Dataset(
        data = matrix(rnorm(500L), nrow = 100L)
        , label = rnorm(100L)
    )

    # num_iterations values (all different)
    num_iterations <- 4L
    num_boost_round <- 2L
    n_iter <- 3L
    nrounds_kwarg <- 6L

    # early_stopping_round values (all different)
    early_stopping_round <- 2L
    early_stopping_round_kwarg <- 3L
    n_iter_no_change <- 4L

    params <- list(
        objective = "regression"
        , metric = "l2"
        , num_boost_round = num_boost_round
        , num_iterations = num_iterations
        , n_iter = n_iter
        , early_stopping_round = early_stopping_round
        , n_iter_no_change = n_iter_no_change
        , verbose = VERBOSITY
    )

    cv_bst <- lgb.cv(
        params = params
        , data = dtrain
        , nrounds = nrounds_kwarg
        , early_stopping_rounds = early_stopping_round_kwarg
        , nfold = 3L
        , verbose = VERBOSITY
    )

    for (bst in cv_bst$boosters) {
        model_str <- bst[["booster"]]$save_model_to_string()
        params_in_file <- .params_from_model_string(model_str = model_str)

        # parameters should match what was passed from the R package, and the "main" (non-alias)
        # params values in `params` should be preferred to keyword argumentts or aliases
        expect_equal(sum(startsWith(params_in_file, "[num_iterations:")), 1L)
        expect_equal(sum(params_in_file == sprintf("[num_iterations: %s]", num_iterations)), 1L)
        expect_equal(sum(startsWith(params_in_file, "[early_stopping_round:")), 1L)
        expect_equal(sum(params_in_file == sprintf("[early_stopping_round: %s]", early_stopping_round)), 1L)

        # none of the aliases shouold have been written to the model file
        expect_equal(sum(startsWith(params_in_file, "[num_boost_round:")), 0L)
        expect_equal(sum(startsWith(params_in_file, "[n_iter:")), 0L)
        expect_equal(sum(startsWith(params_in_file, "[n_iter_no_change:")), 0L)
    }

})

test_that("params (including dataset params) should be stored in .rds file for Booster", {
    data(agaricus.train, package = "lightgbm")
    dtrain <- lgb.Dataset(
        agaricus.train$data
        , label = agaricus.train$label
        , params = list(
            max_bin = 17L
        )
    )
    params <- list(
        objective = "binary"
        , max_depth = 4L
        , bagging_fraction = 0.8
        , verbose = VERBOSITY
    )
    bst <- Booster$new(
        params = params
        , train_set = dtrain
    )
    bst_file <- tempfile(fileext = ".rds")
    expect_warning(saveRDS.lgb.Booster(bst, file = bst_file))

    expect_warning(bst_from_file <- readRDS.lgb.Booster(file = bst_file))
    expect_identical(
        bst_from_file$params
        , list(
            objective = "binary"
            , max_depth = 4L
            , bagging_fraction = 0.8
            , verbose = VERBOSITY
            , max_bin = 17L
        )
    )
})

test_that("params (including dataset params) should be stored in .rds file for Booster", {
    data(agaricus.train, package = "lightgbm")
    dtrain <- lgb.Dataset(
        agaricus.train$data
        , label = agaricus.train$label
        , params = list(
            max_bin = 17L
        )
    )
    params <- list(
        objective = "binary"
        , max_depth = 4L
        , bagging_fraction = 0.8
        , verbose = VERBOSITY
    )
    bst <- Booster$new(
        params = params
        , train_set = dtrain
    )
    bst_file <- tempfile(fileext = ".rds")
    saveRDS(bst, file = bst_file)

    bst_from_file <- readRDS(file = bst_file)
    expect_identical(
        bst_from_file$params
        , list(
            objective = "binary"
            , max_depth = 4L
            , bagging_fraction = 0.8
            , verbose = VERBOSITY
            , max_bin = 17L
        )
    )
})

test_that("Handle is automatically restored when calling predict", {
    data(agaricus.train, package = "lightgbm")
    bst <- lightgbm(
        agaricus.train$data
        , agaricus.train$label
        , nrounds = 5L
        , obj = "binary"
        , params = list(
            verbose = VERBOSITY
        )
    )
    bst_file <- tempfile(fileext = ".rds")
    saveRDS(bst, file = bst_file)

    bst_from_file <- readRDS(file = bst_file)

    pred_before <- predict(bst, agaricus.train$data)
    pred_after <- predict(bst_from_file, agaricus.train$data)
    expect_equal(pred_before, pred_after)
})

test_that("boosters with linear models at leaves work with saveRDS.lgb.Booster and readRDS.lgb.Booster", {
    X <- matrix(rnorm(100L), ncol = 1L)
    labels <- 2L * X + runif(nrow(X), 0L, 0.1)
    dtrain <- lgb.Dataset(
        data = X
        , label = labels
    )

    params <- list(
        objective = "regression"
        , verbose = VERBOSITY
        , metric = "mse"
        , seed = 0L
        , num_leaves = 2L
    )

    bst <- lgb.train(
        data = dtrain
        , nrounds = 10L
        , params = params
    )
    expect_true(lgb.is.Booster(bst))

    # save predictions, then write the model to a file and destroy it in R
    preds <- predict(bst, X)
    model_file <- tempfile(fileext = ".rds")
    expect_warning(saveRDS.lgb.Booster(bst, file = model_file))
    bst$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)

    # load the booster and make predictions...should be the same
    expect_warning({
        bst2 <- readRDS.lgb.Booster(file = model_file)
    })
    preds2 <- predict(bst2, X)
    expect_identical(preds, preds2)
})

test_that("boosters with linear models at leaves can be written to RDS and re-loaded successfully", {
    X <- matrix(rnorm(100L), ncol = 1L)
    labels <- 2L * X + runif(nrow(X), 0L, 0.1)
    dtrain <- lgb.Dataset(
        data = X
        , label = labels
    )

    params <- list(
        objective = "regression"
        , verbose = VERBOSITY
        , metric = "mse"
        , seed = 0L
        , num_leaves = 2L
    )

    bst <- lgb.train(
        data = dtrain
        , nrounds = 10L
        , params = params
    )
    expect_true(lgb.is.Booster(bst))

    # save predictions, then write the model to a file and destroy it in R
    preds <- predict(bst, X)
    model_file <- tempfile(fileext = ".rds")
    saveRDS(bst, file = model_file)
    bst$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)

    # load the booster and make predictions...should be the same
    bst2 <- readRDS(file = model_file)
    preds2 <- predict(bst2, X)
    expect_identical(preds, preds2)
})

test_that("Booster's print, show, and summary work correctly", {
    .have_same_handle <- function(model, other_model) {
       expect_equal(
         model$.__enclos_env__$private$handle
         , other_model$.__enclos_env__$private$handle
       )
    }

    .has_expected_content_for_fitted_model <- function(printed_txt) {
      expect_true(any(startsWith(printed_txt, "LightGBM Model")))
      expect_true(any(startsWith(printed_txt, "Fitted to dataset")))
    }

    .has_expected_content_for_finalized_model <- function(printed_txt) {
      expect_true(any(grepl("^LightGBM Model$", printed_txt)))
      expect_true(any(grepl("Booster handle is invalid", printed_txt)))
    }

    .check_methods_work <- function(model) {

        #--- should work for fitted models --- #

        # print()
        log_txt <- capture.output({
          ret <- print(model)
        })
        .have_same_handle(ret, model)
        .has_expected_content_for_fitted_model(log_txt)

        # show()
        log_txt <- capture.output({
          ret <- show(model)
        })
        expect_null(ret)
        .has_expected_content_for_fitted_model(log_txt)

        # summary()
        log_text <- capture.output({
          ret <- summary(model)
        })
        .have_same_handle(ret, model)
        .has_expected_content_for_fitted_model(log_txt)

        #--- should not fail for finalized models ---#
        model$finalize()

        # print()
        log_txt <- capture.output({
          ret <- print(model)
        })
        .has_expected_content_for_finalized_model(log_txt)

        # show()
        .have_same_handle(ret, model)
        log_txt <- capture.output({
          ret <- show(model)
        })
        expect_null(ret)
        .has_expected_content_for_finalized_model(log_txt)

        # summary()
        log_txt <- capture.output({
          ret <- summary(model)
        })
        .have_same_handle(ret, model)
        .has_expected_content_for_finalized_model(log_txt)
    }

    data("mtcars")
    model <- lgb.train(
        params = list(
          objective = "regression"
          , min_data_in_leaf = 1L
        )
        , data = lgb.Dataset(
            as.matrix(mtcars[, -1L])
            , label = mtcars$mpg
            , params = list(
              min_data_in_bin = 1L
            )
        )
        , verbose = VERBOSITY
        , nrounds = 5L
    )
    .check_methods_work(model)

    data("iris")
    model <- lgb.train(
        params = list(objective = "multiclass", num_class = 3L)
        , data = lgb.Dataset(
            as.matrix(iris[, -5L])
            , label = as.numeric(factor(iris$Species)) - 1.0
        )
        , verbose = VERBOSITY
        , nrounds = 5L
    )
    .check_methods_work(model)


    # with custom objective
    .logregobj <- function(preds, dtrain) {
        labels <- get_field(dtrain, "label")
        preds <- 1.0 / (1.0 + exp(-preds))
        grad <- preds - labels
        hess <- preds * (1.0 - preds)
        return(list(grad = grad, hess = hess))
    }

    .evalerror <- function(preds, dtrain) {
        labels <- get_field(dtrain, "label")
        preds <- 1.0 / (1.0 + exp(-preds))
        err <- as.numeric(sum(labels != (preds > 0.5))) / length(labels)
        return(list(
            name = "error"
            , value = err
            , higher_better = FALSE
        ))
    }

    model <- lgb.train(
        data = lgb.Dataset(
            as.matrix(iris[, -5L])
            , label = as.numeric(iris$Species == "virginica")
        )
        , obj = .logregobj
        , eval = .evalerror
        , verbose = VERBOSITY
        , nrounds = 5L
    )

    .check_methods_work(model)
})

test_that("LGBM_BoosterGetNumFeature_R returns correct outputs", {
    data("mtcars")
    model <- lgb.train(
        params = list(
          objective = "regression"
          , min_data_in_leaf = 1L
        )
        , data = lgb.Dataset(
            as.matrix(mtcars[, -1L])
            , label = mtcars$mpg
            , params = list(
              min_data_in_bin = 1L
            )
        )
        , verbose = VERBOSITY
        , nrounds = 5L
    )
    ncols <- .Call(LGBM_BoosterGetNumFeature_R, model$.__enclos_env__$private$handle)
    expect_equal(ncols, ncol(mtcars) - 1L)

    data("iris")
    model <- lgb.train(
        params = list(objective = "multiclass", num_class = 3L)
        , data = lgb.Dataset(
            as.matrix(iris[, -5L])
            , label = as.numeric(factor(iris$Species)) - 1.0
        )
        , verbose = VERBOSITY
        , nrounds = 5L
    )
    ncols <- .Call(LGBM_BoosterGetNumFeature_R, model$.__enclos_env__$private$handle)
    expect_equal(ncols, ncol(iris) - 1L)
})
