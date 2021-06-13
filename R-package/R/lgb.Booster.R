#' @importFrom R6 R6Class
Booster <- R6::R6Class(
  classname = "lgb.Booster",
  cloneable = FALSE,
  public = list(

    best_iter = -1L,
    best_score = NA_real_,
    params = list(),
    record_evals = list(),

    # Finalize will free up the handles
    finalize = function() {
      .Call(
        LGBM_BoosterFree_R
        , private$handle
      )
      private$handle <- NULL
      return(invisible(NULL))
    },

    # Initialize will create a starter booster
    initialize = function(params = list(),
                          train_set = NULL,
                          modelfile = NULL,
                          model_str = NULL,
                          ...) {

      # Create parameters and handle
      params <- append(params, list(...))
      handle <- NULL

      # Attempts to create a handle for the dataset
      try({

        # Check if training dataset is not null
        if (!is.null(train_set)) {
          # Check if training dataset is lgb.Dataset or not
          if (!lgb.is.Dataset(train_set)) {
            stop("lgb.Booster: Can only use lgb.Dataset as training data")
          }
          train_set_handle <- train_set$.__enclos_env__$private$get_handle()
          params <- modifyList(params, train_set$get_params())
          params_str <- lgb.params2str(params = params)
          # Store booster handle
          handle <- .Call(
            LGBM_BoosterCreate_R
            , train_set_handle
            , params_str
          )

          # Create private booster information
          private$train_set <- train_set
          private$train_set_version <- train_set$.__enclos_env__$private$version
          private$num_dataset <- 1L
          private$init_predictor <- train_set$.__enclos_env__$private$predictor

          # Check if predictor is existing
          if (!is.null(private$init_predictor)) {

            # Merge booster
            .Call(
              LGBM_BoosterMerge_R
              , handle
              , private$init_predictor$.__enclos_env__$private$handle
            )

          }

          # Check current iteration
          private$is_predicted_cur_iter <- c(private$is_predicted_cur_iter, FALSE)

        } else if (!is.null(modelfile)) {

          # Do we have a model file as character?
          if (!is.character(modelfile)) {
            stop("lgb.Booster: Can only use a string as model file path")
          }

          # Create booster from model
          handle <- .Call(
            LGBM_BoosterCreateFromModelfile_R
            , modelfile
          )

        } else if (!is.null(model_str)) {

          # Do we have a model_str as character?
          if (!is.character(model_str)) {
            stop("lgb.Booster: Can only use a string as model_str")
          }

          # Create booster from model
          handle <- .Call(
            LGBM_BoosterLoadModelFromString_R
            , model_str
          )

        } else {

          # Booster non existent
          stop(
            "lgb.Booster: Need at least either training dataset, "
            , "model file, or model_str to create booster instance"
          )

        }

      })

      # Check whether the handle was created properly if it was not stopped earlier by a stop call
      if (isTRUE(lgb.is.null.handle(x = handle))) {

        stop("lgb.Booster: cannot create Booster handle")

      } else {

        # Create class
        class(handle) <- "lgb.Booster.handle"
        private$handle <- handle
        private$num_class <- 1L
        .Call(
          LGBM_BoosterGetNumClasses_R
          , private$handle
          , private$num_class
        )

      }

      self$params <- params

      return(invisible(NULL))

    },

    # Set training data name
    set_train_data_name = function(name) {

      # Set name
      private$name_train_set <- name
      return(invisible(self))

    },

    # Add validation data
    add_valid = function(data, name) {

      # Check if data is lgb.Dataset
      if (!lgb.is.Dataset(data)) {
        stop("lgb.Booster.add_valid: Can only use lgb.Dataset as validation data")
      }

      # Check if predictors are identical
      if (!identical(data$.__enclos_env__$private$predictor, private$init_predictor)) {
        stop(
          "lgb.Booster.add_valid: Failed to add validation data; "
          , "you should use the same predictor for these data"
        )
      }

      # Check if names are character
      if (!is.character(name)) {
        stop("lgb.Booster.add_valid: Can only use characters as data name")
      }

      # Add validation data to booster
      .Call(
        LGBM_BoosterAddValidData_R
        , private$handle
        , data$.__enclos_env__$private$get_handle()
      )

      # Store private information
      private$valid_sets <- c(private$valid_sets, data)
      private$name_valid_sets <- c(private$name_valid_sets, name)
      private$num_dataset <- private$num_dataset + 1L
      private$is_predicted_cur_iter <- c(private$is_predicted_cur_iter, FALSE)

      return(invisible(self))

    },

    # Reset parameters of booster
    reset_parameter = function(params, ...) {

      if (methods::is(self$params, "list")) {
        params <- modifyList(self$params, params)
      }

      params <- modifyList(params, list(...))
      params_str <- lgb.params2str(params = params)

      .Call(
        LGBM_BoosterResetParameter_R
        , private$handle
        , params_str
      )
      self$params <- params

      return(invisible(self))

    },

    # Perform boosting update iteration
    update = function(train_set = NULL, fobj = NULL) {

      if (is.null(train_set)) {
        if (private$train_set$.__enclos_env__$private$version != private$train_set_version) {
          train_set <- private$train_set
        }
      }

      # Check if training set is not null
      if (!is.null(train_set)) {

        # Check if training set is lgb.Dataset
        if (!lgb.is.Dataset(train_set)) {
          stop("lgb.Booster.update: Only can use lgb.Dataset as training data")
        }

        # Check if predictors are identical
        if (!identical(train_set$predictor, private$init_predictor)) {
          stop("lgb.Booster.update: Change train_set failed, you should use the same predictor for these data")
        }

        # Reset training data on booster
        .Call(
          LGBM_BoosterResetTrainingData_R
          , private$handle
          , train_set$.__enclos_env__$private$get_handle()
        )

        # Store private train set
        private$train_set <- train_set
        private$train_set_version <- train_set$.__enclos_env__$private$version

      }

      # Check if objective is empty
      if (is.null(fobj)) {
        if (private$set_objective_to_none) {
          stop("lgb.Booster.update: cannot update due to null objective function")
        }
        # Boost iteration from known objective
        .Call(
          LGBM_BoosterUpdateOneIter_R
          , private$handle
        )

      } else {

        # Check if objective is function
        if (!is.function(fobj)) {
          stop("lgb.Booster.update: fobj should be a function")
        }
        if (!private$set_objective_to_none) {
          self$reset_parameter(params = list(objective = "none"))
          private$set_objective_to_none <- TRUE
        }
        # Perform objective calculation
        gpair <- fobj(private$inner_predict(1L), private$train_set)

        # Check for gradient and hessian as list
        if (is.null(gpair$grad) || is.null(gpair$hess)) {
          stop("lgb.Booster.update: custom objective should
            return a list with attributes (hess, grad)")
        }

        # Return custom boosting gradient/hessian
        .Call(
          LGBM_BoosterUpdateOneIterCustom_R
          , private$handle
          , gpair$grad
          , gpair$hess
          , length(gpair$grad)
        )

      }

      # Loop through each iteration
      for (i in seq_along(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }

      return(invisible(self))

    },

    # Return one iteration behind
    rollback_one_iter = function() {

      # Return one iteration behind
      .Call(
        LGBM_BoosterRollbackOneIter_R
        , private$handle
      )

      # Loop through each iteration
      for (i in seq_along(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }

      return(invisible(self))

    },

    # Get current iteration
    current_iter = function() {

      cur_iter <- 0L
      .Call(
        LGBM_BoosterGetCurrentIteration_R
        , private$handle
        , cur_iter
      )
      return(cur_iter)

    },

    # Get upper bound
    upper_bound = function() {

      upper_bound <- 0.0
      .Call(
        LGBM_BoosterGetUpperBoundValue_R
        , private$handle
        , upper_bound
      )
      return(upper_bound)

    },

    # Get lower bound
    lower_bound = function() {

      lower_bound <- 0.0
      .Call(
        LGBM_BoosterGetLowerBoundValue_R
        , private$handle
        , lower_bound
      )
      return(lower_bound)

    },

    # Evaluate data on metrics
    eval = function(data, name, feval = NULL) {

      # Check if dataset is lgb.Dataset
      if (!lgb.is.Dataset(data)) {
        stop("lgb.Booster.eval: Can only use lgb.Dataset to eval")
      }

      # Check for identical data
      data_idx <- 0L
      if (identical(data, private$train_set)) {
        data_idx <- 1L
      } else {

        # Check for validation data
        if (length(private$valid_sets) > 0L) {

          # Loop through each validation set
          for (i in seq_along(private$valid_sets)) {

            # Check for identical validation data with training data
            if (identical(data, private$valid_sets[[i]])) {

              # Found identical data, skip
              data_idx <- i + 1L
              break

            }

          }

        }

      }

      # Check if evaluation was not done
      if (data_idx == 0L) {

        # Add validation data by name
        self$add_valid(data, name)
        data_idx <- private$num_dataset

      }

      # Evaluate data
      return(
        private$inner_eval(
          data_name = name
          , data_idx = data_idx
          , feval = feval
        )
      )

    },

    # Evaluation training data
    eval_train = function(feval = NULL) {
      return(private$inner_eval(private$name_train_set, 1L, feval))
    },

    # Evaluation validation data
    eval_valid = function(feval = NULL) {

      # Create ret list
      ret <- list()

      # Check if validation is empty
      if (length(private$valid_sets) <= 0L) {
        return(ret)
      }

      # Loop through each validation set
      for (i in seq_along(private$valid_sets)) {
        ret <- append(
          x = ret
          , values = private$inner_eval(private$name_valid_sets[[i]], i + 1L, feval)
        )
      }

      return(ret)

    },

    # Save model
    save_model = function(filename, num_iteration = NULL, feature_importance_type = 0L) {

      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      # Save booster model
      .Call(
        LGBM_BoosterSaveModel_R
        , private$handle
        , as.integer(num_iteration)
        , as.integer(feature_importance_type)
        , filename
      )

      return(invisible(self))
    },

    # Save model to string
    save_model_to_string = function(num_iteration = NULL, feature_importance_type = 0L) {

      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      model_str <- .Call(
          LGBM_BoosterSaveModelToString_R
          , private$handle
          , as.integer(num_iteration)
          , as.integer(feature_importance_type)
      )

      return(model_str)

    },

    # Dump model in memory
    dump_model = function(num_iteration = NULL, feature_importance_type = 0L) {

      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      model_str <- .Call(
        LGBM_BoosterDumpModel_R
        , private$handle
        , as.integer(num_iteration)
        , as.integer(feature_importance_type)
      )

      return(model_str)

    },

    # Predict on new data
    predict = function(data,
                       start_iteration = NULL,
                       num_iteration = NULL,
                       rawscore = FALSE,
                       predleaf = FALSE,
                       predcontrib = FALSE,
                       header = FALSE,
                       reshape = FALSE,
                       ...) {

      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }
      # Check if start iteration is non existent
      if (is.null(start_iteration)) {
        start_iteration <- 0L
      }

      # Predict on new data
      params <- list(...)
      predictor <- Predictor$new(
        modelfile = private$handle
        , params = params
      )
      return(
        predictor$predict(
          data = data
          , start_iteration = start_iteration
          , num_iteration = num_iteration
          , rawscore = rawscore
          , predleaf = predleaf
          , predcontrib = predcontrib
          , header = header
          , reshape = reshape
        )
      )

    },

    # Transform into predictor
    to_predictor = function() {
      return(Predictor$new(modelfile = private$handle))
    },

    # Used for save
    raw = NA,

    # Save model to temporary file for in-memory saving
    save = function() {

      # Overwrite model in object
      self$raw <- self$save_model_to_string(NULL)

      return(invisible(NULL))

    }

  ),
  private = list(
    handle = NULL,
    train_set = NULL,
    name_train_set = "training",
    valid_sets = list(),
    name_valid_sets = list(),
    predict_buffer = list(),
    is_predicted_cur_iter = list(),
    num_class = 1L,
    num_dataset = 0L,
    init_predictor = NULL,
    eval_names = NULL,
    higher_better_inner_eval = NULL,
    set_objective_to_none = FALSE,
    train_set_version = 0L,
    # Predict data
    inner_predict = function(idx) {

      # Store data name
      data_name <- private$name_train_set

      # Check for id bigger than 1
      if (idx > 1L) {
        data_name <- private$name_valid_sets[[idx - 1L]]
      }

      # Check for unknown dataset (over the maximum provided range)
      if (idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }

      # Check for prediction buffer
      if (is.null(private$predict_buffer[[data_name]])) {

        # Store predictions
        npred <- 0L
        .Call(
          LGBM_BoosterGetNumPredict_R
          , private$handle
          , as.integer(idx - 1L)
          , npred
        )
        private$predict_buffer[[data_name]] <- numeric(npred)

      }

      # Check if current iteration was already predicted
      if (!private$is_predicted_cur_iter[[idx]]) {

        # Use buffer
        .Call(
          LGBM_BoosterGetPredict_R
          , private$handle
          , as.integer(idx - 1L)
          , private$predict_buffer[[data_name]]
        )
        private$is_predicted_cur_iter[[idx]] <- TRUE
      }

      return(private$predict_buffer[[data_name]])
    },

    # Get evaluation information
    get_eval_info = function() {

      # Check for evaluation names emptiness
      if (is.null(private$eval_names)) {
        eval_names <- .Call(
          LGBM_BoosterGetEvalNames_R
          , private$handle
        )

        # Check names' length
        if (length(eval_names) > 0L) {

          # Parse and store privately names
          private$eval_names <- eval_names

          # some metrics don't map cleanly to metric names, for example "ndcg@1" is just the
          # ndcg metric evaluated at the first "query result" in learning-to-rank
          metric_names <- gsub("@.*", "", eval_names)
          private$higher_better_inner_eval <- .METRICS_HIGHER_BETTER()[metric_names]

        }

      }

      return(private$eval_names)

    },

    # Perform inner evaluation
    inner_eval = function(data_name, data_idx, feval = NULL) {

      # Check for unknown dataset (over the maximum provided range)
      if (data_idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }

      # Get evaluation information
      private$get_eval_info()

      # Prepare return
      ret <- list()

      # Check evaluation names existence
      if (length(private$eval_names) > 0L) {

        # Create evaluation values
        tmp_vals <- numeric(length(private$eval_names))
        .Call(
          LGBM_BoosterGetEval_R
          , private$handle
          , as.integer(data_idx - 1L)
          , tmp_vals
        )

        # Loop through all evaluation names
        for (i in seq_along(private$eval_names)) {

          # Store evaluation and append to return
          res <- list()
          res$data_name <- data_name
          res$name <- private$eval_names[i]
          res$value <- tmp_vals[i]
          res$higher_better <- private$higher_better_inner_eval[i]
          ret <- append(ret, list(res))

        }

      }

      # Check if there are evaluation metrics
      if (!is.null(feval)) {

        # Check if evaluation metric is a function
        if (!is.function(feval)) {
          stop("lgb.Booster.eval: feval should be a function")
        }

        # Prepare data
        data <- private$train_set

        # Check if data to assess is existing differently
        if (data_idx > 1L) {
          data <- private$valid_sets[[data_idx - 1L]]
        }

        # Perform function evaluation
        res <- feval(private$inner_predict(data_idx), data)

        # Check for name correctness
        if (is.null(res$name) || is.null(res$value) ||  is.null(res$higher_better)) {
          stop("lgb.Booster.eval: custom eval function should return a
            list with attribute (name, value, higher_better)");
        }

        # Append names and evaluation
        res$data_name <- data_name
        ret <- append(ret, list(res))
      }

      return(ret)

    }

  )
)

#' @name predict.lgb.Booster
#' @title Predict method for LightGBM model
#' @description Predicted values based on class \code{lgb.Booster}
#' @param object Object of class \code{lgb.Booster}
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param start_iteration int or None, optional (default=None)
#'                        Start index of the iteration to predict.
#'                        If None or <= 0, starts from the first iteration.
#' @param num_iteration int or None, optional (default=None)
#'                      Limit number of iterations in the prediction.
#'                      If None, if the best iteration exists and start_iteration is None or <= 0, the
#'                      best iteration is used; otherwise, all iterations from start_iteration are used.
#'                      If <= 0, all iterations from start_iteration are used (no limits).
#' @param rawscore whether the prediction should be returned in the for of original untransformed
#'                 sum of predictions from boosting iterations' results. E.g., setting \code{rawscore=TRUE}
#'                 for logistic regression would result in predictions for log-odds instead of probabilities.
#' @param predleaf whether predict leaf index instead.
#' @param predcontrib return per-feature contributions for each record.
#' @param header only used for prediction for text file. True if text file has header
#' @param reshape whether to reshape the vector of predictions to a matrix form when there are several
#'                prediction outputs per case.
#' @param ... Additional named arguments passed to the \code{predict()} method of
#'            the \code{lgb.Booster} object passed to \code{object}.
#' @return For regression or binary classification, it returns a vector of length \code{nrows(data)}.
#'         For multiclass classification, either a \code{num_class * nrows(data)} vector or
#'         a \code{(nrows(data), num_class)} dimension matrix is returned, depending on
#'         the \code{reshape} value.
#'
#'         When \code{predleaf = TRUE}, the output is a matrix object with the
#'         number of columns corresponding to the number of trees.
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#'   , valids = valids
#'   , min_data = 1L
#'   , learning_rate = 1.0
#' )
#' preds <- predict(model, test$data)
#' }
#' @export
predict.lgb.Booster <- function(object,
                                data,
                                start_iteration = NULL,
                                num_iteration = NULL,
                                rawscore = FALSE,
                                predleaf = FALSE,
                                predcontrib = FALSE,
                                header = FALSE,
                                reshape = FALSE,
                                ...) {

  if (!lgb.is.Booster(x = object)) {
    stop("predict.lgb.Booster: object should be an ", sQuote("lgb.Booster"))
  }

  # Return booster predictions
  return(
    object$predict(
      data = data
      , start_iteration = start_iteration
      , num_iteration = num_iteration
      , rawscore = rawscore
      , predleaf =  predleaf
      , predcontrib =  predcontrib
      , header = header
      , reshape = reshape
      , ...
    )
  )
}

#' @name lgb.load
#' @title Load LightGBM model
#' @description Load LightGBM takes in either a file path or model string.
#'              If both are provided, Load will default to loading from file
#' @param filename path of model file
#' @param model_str a str containing the model
#'
#' @return lgb.Booster
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#'   , valids = valids
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , early_stopping_rounds = 3L
#' )
#' model_file <- tempfile(fileext = ".txt")
#' lgb.save(model, model_file)
#' load_booster <- lgb.load(filename = model_file)
#' model_string <- model$save_model_to_string(NULL) # saves best iteration
#' load_booster_from_str <- lgb.load(model_str = model_string)
#' }
#' @export
lgb.load <- function(filename = NULL, model_str = NULL) {

  filename_provided <- !is.null(filename)
  model_str_provided <- !is.null(model_str)

  if (filename_provided) {
    if (!is.character(filename)) {
      stop("lgb.load: filename should be character")
    }
    if (!file.exists(filename)) {
      stop(sprintf("lgb.load: file '%s' passed to filename does not exist", filename))
    }
    return(invisible(Booster$new(modelfile = filename)))
  }

  if (model_str_provided) {
    if (!is.character(model_str)) {
      stop("lgb.load: model_str should be character")
    }
    return(invisible(Booster$new(model_str = model_str)))
  }

  stop("lgb.load: either filename or model_str must be given")
}

#' @name lgb.save
#' @title Save LightGBM model
#' @description Save LightGBM model
#' @param booster Object of class \code{lgb.Booster}
#' @param filename saved filename
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#'
#' @return lgb.Booster
#'
#' @examples
#' \donttest{
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 10L
#'   , valids = valids
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , early_stopping_rounds = 5L
#' )
#' lgb.save(model, tempfile(fileext = ".txt"))
#' }
#' @export
lgb.save <- function(booster, filename, num_iteration = NULL) {

  if (!lgb.is.Booster(x = booster)) {
    stop("lgb.save: booster should be an ", sQuote("lgb.Booster"))
  }

  if (!(is.character(filename) && length(filename) == 1L)) {
    stop("lgb.save: filename should be a string")
  }

  # Store booster
  return(
    invisible(booster$save_model(
      filename = filename
      , num_iteration = num_iteration
    ))
  )

}

#' @name lgb.dump
#' @title Dump LightGBM model to json
#' @description Dump LightGBM model to json
#' @param booster Object of class \code{lgb.Booster}
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#'
#' @return json format of model
#'
#' @examples
#' \donttest{
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 10L
#'   , valids = valids
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , early_stopping_rounds = 5L
#' )
#' json_model <- lgb.dump(model)
#' }
#' @export
lgb.dump <- function(booster, num_iteration = NULL) {

  if (!lgb.is.Booster(x = booster)) {
    stop("lgb.save: booster should be an ", sQuote("lgb.Booster"))
  }

  # Return booster at requested iteration
  return(booster$dump_model(num_iteration =  num_iteration))

}

#' @name lgb.get.eval.result
#' @title Get record evaluation result from booster
#' @description Given a \code{lgb.Booster}, return evaluation results for a
#'              particular metric on a particular dataset.
#' @param booster Object of class \code{lgb.Booster}
#' @param data_name Name of the dataset to return evaluation results for.
#' @param eval_name Name of the evaluation metric to return results for.
#' @param iters An integer vector of iterations you want to get evaluation results for. If NULL
#'              (the default), evaluation results for all iterations will be returned.
#' @param is_err TRUE will return evaluation error instead
#'
#' @return numeric vector of evaluation result
#'
#' @examples
#' \donttest{
#' # train a regression model
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#'   , valids = valids
#'   , min_data = 1L
#'   , learning_rate = 1.0
#' )
#'
#' # Examine valid data_name values
#' print(setdiff(names(model$record_evals), "start_iter"))
#'
#' # Examine valid eval_name values for dataset "test"
#' print(names(model$record_evals[["test"]]))
#'
#' # Get L2 values for "test" dataset
#' lgb.get.eval.result(model, "test", "l2")
#' }
#' @export
lgb.get.eval.result <- function(booster, data_name, eval_name, iters = NULL, is_err = FALSE) {

  # Check if booster is booster
  if (!lgb.is.Booster(x = booster)) {
    stop("lgb.get.eval.result: Can only use ", sQuote("lgb.Booster"), " to get eval result")
  }

  # Check if data and evaluation name are characters or not
  if (!is.character(data_name) || !is.character(eval_name)) {
    stop("lgb.get.eval.result: data_name and eval_name should be characters")
  }

  # NOTE: "start_iter" exists in booster$record_evals but is not a valid data_name
  data_names <- setdiff(names(booster$record_evals), "start_iter")
  if (!(data_name %in% data_names)) {
    stop(paste0(
      "lgb.get.eval.result: data_name "
      , shQuote(data_name)
      , " not found. Only the following datasets exist in record evals: ["
      , paste(data_names, collapse = ", ")
      , "]"
    ))
  }

  # Check if evaluation result is existing
  eval_names <- names(booster$record_evals[[data_name]])
  if (!(eval_name %in% eval_names)) {
    stop(paste0(
      "lgb.get.eval.result: eval_name "
      , shQuote(eval_name)
      , " not found. Only the following eval_names exist for dataset "
      , shQuote(data_name)
      , ": ["
      , paste(eval_names, collapse = ", ")
      , "]"
    ))
    stop("lgb.get.eval.result: wrong eval name")
  }

  result <- booster$record_evals[[data_name]][[eval_name]][[.EVAL_KEY()]]

  # Check if error is requested
  if (is_err) {
    result <- booster$record_evals[[data_name]][[eval_name]][[.EVAL_ERR_KEY()]]
  }

  # Check if iteration is non existant
  if (is.null(iters)) {
    return(as.numeric(result))
  }

  # Parse iteration and booster delta
  iters <- as.integer(iters)
  delta <- booster$record_evals$start_iter - 1.0
  iters <- iters - delta

  # Return requested result
  return(as.numeric(result[iters]))
}
