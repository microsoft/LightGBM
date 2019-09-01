#' @importFrom R6 R6Class
Booster <- R6::R6Class(
  classname = "lgb.Booster",
  cloneable = FALSE,
  public = list(

    best_iter = -1,
    best_score = NA,
    record_evals = list(),

    # Finalize will free up the handles
    finalize = function() {

      # Check the need for freeing handle
      if (!lgb.is.null.handle(private$handle)) {

        # Freeing up handle
        lgb.call("LGBM_BoosterFree_R", ret = NULL, private$handle)
        private$handle <- NULL

      }

    },

    # Initialize will create a starter booster
    initialize = function(params = list(),
                          train_set = NULL,
                          modelfile = NULL,
                          model_str = NULL,
                          ...) {

      # Create parameters and handle
      params <- append(params, list(...))
      params_str <- lgb.params2str(params)
      handle <- 0.0

      # Attempts to create a handle for the dataset
      try({

        # Check if training dataset is not null
        if (!is.null(train_set)) {

          # Check if training dataset is lgb.Dataset or not
          if (!lgb.check.r6.class(train_set, "lgb.Dataset")) {
            stop("lgb.Booster: Can only use lgb.Dataset as training data")
          }

          # Store booster handle
          handle <- lgb.call("LGBM_BoosterCreate_R", ret = handle, train_set$.__enclos_env__$private$get_handle(), params_str)

          # Create private booster information
          private$train_set <- train_set
          private$num_dataset <- 1
          private$init_predictor <- train_set$.__enclos_env__$private$predictor

          # Check if predictor is existing
          if (!is.null(private$init_predictor)) {

            # Merge booster
            lgb.call("LGBM_BoosterMerge_R",
                     ret = NULL,
                     handle,
                     private$init_predictor$.__enclos_env__$private$handle)

          }

          # Check current iteration
          private$is_predicted_cur_iter <- c(private$is_predicted_cur_iter, FALSE)

        } else if (!is.null(modelfile)) {

          # Do we have a model file as character?
          if (!is.character(modelfile)) {
            stop("lgb.Booster: Can only use a string as model file path")
          }

          # Create booster from model
          handle <- lgb.call("LGBM_BoosterCreateFromModelfile_R",
                             ret = handle,
                             lgb.c_str(modelfile))

        } else if (!is.null(model_str)) {

          # Do we have a model_str as character?
          if (!is.character(model_str)) {
            stop("lgb.Booster: Can only use a string as model_str")
          }

          # Create booster from model
          handle <- lgb.call("LGBM_BoosterLoadModelFromString_R",
                             ret = handle,
                             lgb.c_str(model_str))

        } else {

          # Booster non existent
          stop("lgb.Booster: Need at least either training dataset, model file, or model_str to create booster instance")

        }

      })

      # Check whether the handle was created properly if it was not stopped earlier by a stop call
      if (lgb.is.null.handle(handle)) {

        stop("lgb.Booster: cannot create Booster handle")

      } else {

        # Create class
        class(handle) <- "lgb.Booster.handle"
        private$handle <- handle
        private$num_class <- 1L
        private$num_class <- lgb.call("LGBM_BoosterGetNumClasses_R",
                                      ret = private$num_class,
                                      private$handle)

      }

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
      if (!lgb.check.r6.class(data, "lgb.Dataset")) {
        stop("lgb.Booster.add_valid: Can only use lgb.Dataset as validation data")
      }

      # Check if predictors are identical
      if (!identical(data$.__enclos_env__$private$predictor, private$init_predictor)) {
        stop("lgb.Booster.add_valid: Failed to add validation data; you should use the same predictor for these data")
      }

      # Check if names are character
      if (!is.character(name)) {
        stop("lgb.Booster.add_valid: Can only use characters as data name")
      }

      # Add validation data to booster
      lgb.call("LGBM_BoosterAddValidData_R",
               ret = NULL,
               private$handle,
               data$.__enclos_env__$private$get_handle())

      # Store private information
      private$valid_sets <- c(private$valid_sets, data)
      private$name_valid_sets <- c(private$name_valid_sets, name)
      private$num_dataset <- private$num_dataset + 1
      private$is_predicted_cur_iter <- c(private$is_predicted_cur_iter, FALSE)

      # Return self
      return(invisible(self))

    },

    # Reset parameters of booster
    reset_parameter = function(params, ...) {

      # Append parameters
      params <- append(params, list(...))
      params_str <- lgb.params2str(params)

      # Reset parameters
      lgb.call("LGBM_BoosterResetParameter_R",
               ret = NULL,
               private$handle,
               params_str)

      # Return self
      return(invisible(self))

    },

    # Perform boosting update iteration
    update = function(train_set = NULL, fobj = NULL) {

      # Check if training set is not null
      if (!is.null(train_set)) {

        # Check if training set is lgb.Dataset
        if (!lgb.check.r6.class(train_set, "lgb.Dataset")) {
          stop("lgb.Booster.update: Only can use lgb.Dataset as training data")
        }

        # Check if predictors are identical
        if (!identical(train_set$predictor, private$init_predictor)) {
          stop("lgb.Booster.update: Change train_set failed, you should use the same predictor for these data")
        }

        # Reset training data on booster
        lgb.call("LGBM_BoosterResetTrainingData_R",
                 ret = NULL,
                 private$handle,
                 train_set$.__enclos_env__$private$get_handle())

        # Store private train set
        private$train_set = train_set

      }

      # Check if objective is empty
      if (is.null(fobj)) {
        if (private$set_objective_to_none) {
          stop("lgb.Booster.update: cannot update due to null objective function")
        }
        # Boost iteration from known objective
        ret <- lgb.call("LGBM_BoosterUpdateOneIter_R", ret = NULL, private$handle)

      } else {

        # Check if objective is function
        if (!is.function(fobj)) {
          stop("lgb.Booster.update: fobj should be a function")
        }
        if (!private$set_objective_to_none) {
          self$reset_parameter(params = list(objective = "none"))
          private$set_objective_to_none = TRUE
        }
        # Perform objective calculation
        gpair <- fobj(private$inner_predict(1), private$train_set)

        # Check for gradient and hessian as list
        if(is.null(gpair$grad) || is.null(gpair$hess)){
          stop("lgb.Booster.update: custom objective should
            return a list with attributes (hess, grad)")
        }

        # Return custom boosting gradient/hessian
        ret <- lgb.call("LGBM_BoosterUpdateOneIterCustom_R",
                        ret = NULL,
                        private$handle,
                        gpair$grad,
                        gpair$hess,
                        length(gpair$grad))

      }

      # Loop through each iteration
      for (i in seq_along(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }

      return(ret)

    },

    # Return one iteration behind
    rollback_one_iter = function() {

      # Return one iteration behind
      lgb.call("LGBM_BoosterRollbackOneIter_R",
               ret = NULL,
               private$handle)

      # Loop through each iteration
      for (i in seq_along(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }

      # Return self
      return(invisible(self))

    },

    # Get current iteration
    current_iter = function() {

      cur_iter <- 0L
      lgb.call("LGBM_BoosterGetCurrentIteration_R",
               ret = cur_iter,
               private$handle)

    },

    # Evaluate data on metrics
    eval = function(data, name, feval = NULL) {

      # Check if dataset is lgb.Dataset
      if (!lgb.check.r6.class(data, "lgb.Dataset")) {
        stop("lgb.Booster.eval: Can only use lgb.Dataset to eval")
      }

      # Check for identical data
      data_idx <- 0
      if (identical(data, private$train_set)) {
        data_idx <- 1
      } else {

        # Check for validation data
        if (length(private$valid_sets) > 0) {

          # Loop through each validation set
          for (i in seq_along(private$valid_sets)) {

            # Check for identical validation data with training data
            if (identical(data, private$valid_sets[[i]])) {

              # Found identical data, skip
              data_idx <- i + 1
              break

            }

          }

        }

      }

      # Check if evaluation was not done
      if (data_idx == 0) {

        # Add validation data by name
        self$add_valid(data, name)
        data_idx <- private$num_dataset

      }

      # Evaluate data
      private$inner_eval(name, data_idx, feval)

    },

    # Evaluation training data
    eval_train = function(feval = NULL) {
      private$inner_eval(private$name_train_set, 1, feval)
    },

    # Evaluation validation data
    eval_valid = function(feval = NULL) {

      # Create ret list
      ret = list()

      # Check if validation is empty
      if (length(private$valid_sets) <= 0) {
        return(ret)
      }

      # Loop through each validation set
      for (i in seq_along(private$valid_sets)) {
        ret <- append(ret, private$inner_eval(private$name_valid_sets[[i]], i + 1, feval))
      }

      # Return ret
      return(ret)

    },

    # Save model
    save_model = function(filename, num_iteration = NULL) {

      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      # Save booster model
      lgb.call("LGBM_BoosterSaveModel_R",
               ret = NULL,
               private$handle,
               as.integer(num_iteration),
               lgb.c_str(filename))

      # Return self
      return(invisible(self))
    },

    # Save model to string
    save_model_to_string = function(num_iteration = NULL) {

      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      # Return model string
      return(lgb.call.return.str("LGBM_BoosterSaveModelToString_R",
                                 private$handle,
                                 as.integer(num_iteration)))

    },

    # Dump model in memory
    dump_model = function(num_iteration = NULL) {

      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      # Return dumped model
      lgb.call.return.str("LGBM_BoosterDumpModel_R",
                          private$handle,
                          as.integer(num_iteration))

    },

    # Predict on new data
    predict = function(data,
                       num_iteration = NULL,
                       rawscore = FALSE,
                       predleaf = FALSE,
                       predcontrib = FALSE,
                       header = FALSE,
                       reshape = FALSE, ...) {

      # Check if number of iteration is  non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      # Predict on new data
      predictor <- Predictor$new(private$handle, ...)
      predictor$predict(data, num_iteration, rawscore, predleaf, predcontrib, header, reshape)

    },

    # Transform into predictor
    to_predictor = function() {
      Predictor$new(private$handle)
    },

    # Used for save
    raw = NA,

    # Save model to temporary file for in-memory saving
    save = function() {

      # Overwrite model in object
      self$raw <- self$save_model_to_string(NULL)

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
    num_class = 1,
    num_dataset = 0,
    init_predictor = NULL,
    eval_names = NULL,
    higher_better_inner_eval = NULL,
    set_objective_to_none = FALSE,
    # Predict data
    inner_predict = function(idx) {

      # Store data name
      data_name <- private$name_train_set

      # Check for id bigger than 1
      if (idx > 1) {
        data_name <- private$name_valid_sets[[idx - 1]]
      }

      # Check for unknown dataset (over the maximum provided range)
      if (idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }

      # Check for prediction buffer
      if (is.null(private$predict_buffer[[data_name]])) {

        # Store predictions
        npred <- 0L
        npred <- lgb.call("LGBM_BoosterGetNumPredict_R",
                          ret = npred,
                          private$handle,
                          as.integer(idx - 1))
        private$predict_buffer[[data_name]] <- numeric(npred)

      }

      # Check if current iteration was already predicted
      if (!private$is_predicted_cur_iter[[idx]]) {

        # Use buffer
        private$predict_buffer[[data_name]] <- lgb.call("LGBM_BoosterGetPredict_R",
                                                        ret = private$predict_buffer[[data_name]],
                                                        private$handle,
                                                        as.integer(idx - 1))
        private$is_predicted_cur_iter[[idx]] <- TRUE
      }

      # Return prediction buffer
      return(private$predict_buffer[[data_name]])
    },

    # Get evaluation information
    get_eval_info = function() {

      # Check for evaluation names emptiness
      if (is.null(private$eval_names)) {

        # Get evaluation names
        names <- lgb.call.return.str("LGBM_BoosterGetEvalNames_R",
                                     private$handle)

        # Check names' length
        if (nchar(names) > 0) {

          # Parse and store privately names
          names <- strsplit(names, "\t")[[1]]
          private$eval_names <- names
          private$higher_better_inner_eval <- grepl("^ndcg|^map|^auc$", names)

        }

      }

      # Return evaluation names
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
      if (length(private$eval_names) > 0) {

        # Create evaluation values
        tmp_vals <- numeric(length(private$eval_names))
        tmp_vals <- lgb.call("LGBM_BoosterGetEval_R",
                             ret = tmp_vals,
                             private$handle,
                             as.integer(data_idx - 1))

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
        if (data_idx > 1) {
          data <- private$valid_sets[[data_idx - 1]]
        }

        # Perform function evaluation
        res <- feval(private$inner_predict(data_idx), data)

        # Check for name correctness
        if(is.null(res$name) || is.null(res$value) ||  is.null(res$higher_better)) {
          stop("lgb.Booster.eval: custom eval function should return a
            list with attribute (name, value, higher_better)");
        }

        # Append names and evaluation
        res$data_name <- data_name
        ret <- append(ret, list(res))
      }

      # Return ret
      return(ret)

    }

  )
)


#' Predict method for LightGBM model
#'
#' Predicted values based on class \code{lgb.Booster}
#'
#' @param object Object of class \code{lgb.Booster}
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#' @param rawscore whether the prediction should be returned in the for of original untransformed
#'        sum of predictions from boosting iterations' results. E.g., setting \code{rawscore=TRUE} for
#'        logistic regression would result in predictions for log-odds instead of probabilities.
#' @param predleaf whether predict leaf index instead.
#' @param predcontrib return per-feature contributions for each record.
#' @param header only used for prediction for text file. True if text file has header
#' @param reshape whether to reshape the vector of predictions to a matrix form when there are several
#'        prediction outputs per case.
#' @param ... Additional named arguments passed to the \code{predict()} method of
#'            the \code{lgb.Booster} object passed to \code{object}.
#' @return
#' For regression or binary classification, it returns a vector of length \code{nrows(data)}.
#' For multiclass classification, either a \code{num_class * nrows(data)} vector or
#' a \code{(nrows(data), num_class)} dimension matrix is returned, depending on
#' the \code{reshape} value.
#'
#' When \code{predleaf = TRUE}, the output is a matrix object with the
#' number of columns corresponding to the number of trees.
#'
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(params,
#'                    dtrain,
#'                    10,
#'                    valids,
#'                    min_data = 1,
#'                    learning_rate = 1,
#'                    early_stopping_rounds = 5)
#' preds <- predict(model, test$data)
#'
#' @rdname predict.lgb.Booster
#' @export
predict.lgb.Booster <- function(object,
                                data,
                                num_iteration = NULL,
                                rawscore = FALSE,
                                predleaf = FALSE,
                                predcontrib = FALSE,
                                header = FALSE,
                                reshape = FALSE,
                                ...) {

  # Check booster existence
  if (!lgb.is.Booster(object)) {
    stop("predict.lgb.Booster: object should be an ", sQuote("lgb.Booster"))
  }

  # Return booster predictions
  object$predict(data,
                 num_iteration,
                 rawscore,
                 predleaf,
                 predcontrib,
                 header,
                 reshape, ...)
}

#' Load LightGBM model
#'
#' Load LightGBM model from saved model file or string
#' Load LightGBM takes in either a file path or model string
#' If both are provided, Load will default to loading from file
#'
#' @param filename path of model file
#' @param model_str a str containing the model
#'
#' @return lgb.Booster
#'
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(params,
#'                    dtrain,
#'                    10,
#'                    valids,
#'                    min_data = 1,
#'                    learning_rate = 1,
#'                    early_stopping_rounds = 5)
#' lgb.save(model, "model.txt")
#' load_booster <- lgb.load(filename = "model.txt")
#' model_string <- model$save_model_to_string(NULL) # saves best iteration
#' load_booster_from_str <- lgb.load(model_str = model_string)
#'
#' @rdname lgb.load
#' @export
lgb.load <- function(filename = NULL, model_str = NULL){

  if (is.null(filename) && is.null(model_str)) {
    stop("lgb.load: either filename or model_str must be given")
  }

  # Load from filename
  if (!is.null(filename) && !is.character(filename)) {
    stop("lgb.load: filename should be character")
  }

  # Return new booster
  if (!is.null(filename) && !file.exists(filename)) stop("lgb.load: file does not exist for supplied filename")
  if (!is.null(filename)) return(invisible(Booster$new(modelfile = filename)))

  # Load from model_str
  if (!is.null(model_str) && !is.character(model_str)) {
    stop("lgb.load: model_str should be character")
  }
  # Return new booster
  if (!is.null(model_str)) return(invisible(Booster$new(model_str = model_str)))

}

#' Save LightGBM model
#'
#' Save LightGBM model
#'
#' @param booster Object of class \code{lgb.Booster}
#' @param filename saved filename
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#'
#' @return lgb.Booster
#'
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(params,
#'                    dtrain,
#'                    10,
#'                    valids,
#'                    min_data = 1,
#'                    learning_rate = 1,
#'                    early_stopping_rounds = 5)
#' lgb.save(model, "model.txt")
#'
#' @rdname lgb.save
#' @export
lgb.save <- function(booster, filename, num_iteration = NULL){

  # Check if booster is booster
  if (!lgb.is.Booster(booster)) {
    stop("lgb.save: booster should be an ", sQuote("lgb.Booster"))
  }

  # Check if file name is character
  if (!is.character(filename)) {
    stop("lgb.save: filename should be a character")
  }

  # Store booster
  invisible(booster$save_model(filename, num_iteration))

}

#' Dump LightGBM model to json
#'
#' Dump LightGBM model to json
#'
#' @param booster Object of class \code{lgb.Booster}
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#'
#' @return json format of model
#'
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(params,
#'                   dtrain,
#'                    10,
#'                    valids,
#'                    min_data = 1,
#'                    learning_rate = 1,
#'                    early_stopping_rounds = 5)
#' json_model <- lgb.dump(model)
#'
#' @rdname lgb.dump
#' @export
lgb.dump <- function(booster, num_iteration = NULL){

  # Check if booster is booster
  if (!lgb.is.Booster(booster)) {
    stop("lgb.save: booster should be an ", sQuote("lgb.Booster"))
  }

  # Return booster at requested iteration
  booster$dump_model(num_iteration)

}

#' Get record evaluation result from booster
#'
#' Get record evaluation result from booster
#' @param booster Object of class \code{lgb.Booster}
#' @param data_name name of dataset
#' @param eval_name name of evaluation
#' @param iters iterations, NULL will return all
#' @param is_err TRUE will return evaluation error instead
#'
#' @return vector of evaluation result
#'
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(params,
#'                    dtrain,
#'                    10,
#'                    valids,
#'                    min_data = 1,
#'                    learning_rate = 1,
#'                    early_stopping_rounds = 5)
#' lgb.get.eval.result(model, "test", "l2")
#' @rdname lgb.get.eval.result
#' @export
lgb.get.eval.result <- function(booster, data_name, eval_name, iters = NULL, is_err = FALSE) {

  # Check if booster is booster
  if (!lgb.is.Booster(booster)) {
    stop("lgb.get.eval.result: Can only use ", sQuote("lgb.Booster"), " to get eval result")
  }

  # Check if data and evaluation name are characters or not
  if (!is.character(data_name) || !is.character(eval_name)) {
    stop("lgb.get.eval.result: data_name and eval_name should be characters")
  }

  # Check if recorded evaluation is existing
  if (is.null(booster$record_evals[[data_name]])) {
    stop("lgb.get.eval.result: wrong data name")
  }

  # Check if evaluation result is existing
  if (is.null(booster$record_evals[[data_name]][[eval_name]])) {
    stop("lgb.get.eval.result: wrong eval name")
  }

  # Create result
  result <- booster$record_evals[[data_name]][[eval_name]]$eval

  # Check if error is requested
  if (is_err) {
    result <- booster$record_evals[[data_name]][[eval_name]]$eval_err
  }

  # Check if iteration is non existant
  if (is.null(iters)) {
    return(as.numeric(result))
  }

  # Parse iteration and booster delta
  iters <- as.integer(iters)
  delta <- booster$record_evals$start_iter - 1
  iters <- iters - delta

  # Return requested result
  as.numeric(result[iters])
}
