#' @title Main training logic for LightGBM
#' @name lgb.train
#' @description Logic to train with LightGBM
#' @inheritParams lgb_shared_params
#' @param valids a list of \code{lgb.Dataset} objects, used for validation
#' @param obj objective function, can be character or custom objective function. Examples include
#'        \code{regression}, \code{regression_l1}, \code{huber},
#'        \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}
#' @param eval evaluation function, can be (a list of) character or custom eval function
#' @param record Boolean, TRUE will record iteration message to \code{booster$record_evals}
#' @param colnames feature names, if not null, will use this to overwrite the names in dataset
#' @param categorical_feature list of str or int
#'        type int represents index,
#'        type str represents feature names
#' @param callbacks list of callback functions
#'        List of callback functions that are applied at each iteration.
#' @param reset_data Boolean, setting it to TRUE (not the default value) will transform the booster model into a predictor model which frees up memory and the original datasets
#' @param ... other parameters, see Parameters.rst for more information. A few key parameters:
#'            \itemize{
#'                \item{boosting}{Boosting type. \code{"gbdt"} or \code{"dart"}}
#'                \item{num_leaves}{number of leaves in one tree. defaults to 127}
#'                \item{max_depth}{Limit the max depth for tree model. This is used to deal with
#'                                 overfit when #data is small. Tree still grow by leaf-wise.}
#'                \item{num_threads}{Number of threads for LightGBM. For the best speed, set this to
#'                                   the number of real CPU cores, not the number of threads (most
#'                                   CPU using hyper-threading to generate 2 threads per CPU core).}
#'            }
#' @return a trained booster model \code{lgb.Booster}.
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
#' @export
lgb.train <- function(params = list(),
                      data,
                      nrounds = 10,
                      valids = list(),
                      obj = NULL,
                      eval = NULL,
                      verbose = 1,
                      record = TRUE,
                      eval_freq = 1L,
                      init_model = NULL,
                      colnames = NULL,
                      categorical_feature = NULL,
                      early_stopping_rounds = NULL,
                      callbacks = list(),
                      reset_data = FALSE,
                      ...) {

  # Setup temporary variables
  additional_params <- list(...)
  params <- append(params, additional_params)
  params$verbose <- verbose
  params <- lgb.check.obj(params, obj)
  params <- lgb.check.eval(params, eval)
  fobj <- NULL
  feval <- NULL

  if (nrounds <= 0) {
    stop("nrounds should be greater than zero")
  }

  # Check for objective (function or not)
  if (is.function(params$objective)) {
    fobj <- params$objective
    params$objective <- "NONE"
  }

  # Check for loss (function or not)
  if (is.function(eval)) {
    feval <- eval
  }

  # Check for parameters
  lgb.check.params(params)

  # Init predictor to empty
  predictor <- NULL

  # Check for boosting from a trained model
  if (is.character(init_model)) {
    predictor <- Predictor$new(init_model)
  } else if (lgb.is.Booster(init_model)) {
    predictor <- init_model$to_predictor()
  }

  # Set the iteration to start from / end to (and check for boosting from a trained model, again)
  begin_iteration <- 1
  if (!is.null(predictor)) {
    begin_iteration <- predictor$current_iter() + 1
  }
  # Check for number of rounds passed as parameter - in case there are multiple ones, take only the first one
  n_rounds <- c("num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees", "num_round", "num_rounds", "num_boost_round", "n_estimators")
  if (any(names(params) %in% n_rounds)) {
    end_iteration <- begin_iteration + params[[which(names(params) %in% n_rounds)[1]]] - 1
  } else {
    end_iteration <- begin_iteration + nrounds - 1
  }


  # Check for training dataset type correctness
  if (!lgb.is.Dataset(data)) {
    stop("lgb.train: data only accepts lgb.Dataset object")
  }

  # Check for validation dataset type correctness
  if (length(valids) > 0) {

    # One or more validation dataset

    # Check for list as input and type correctness by object
    if (!is.list(valids) || !all(vapply(valids, lgb.is.Dataset, logical(1)))) {
      stop("lgb.train: valids must be a list of lgb.Dataset elements")
    }

    # Attempt to get names
    evnames <- names(valids)

    # Check for names existance
    if (is.null(evnames) || !all(nzchar(evnames))) {
      stop("lgb.train: each element of the valids must have a name tag")
    }
  }

  # Update parameters with parsed parameters
  data$update_params(params)

  # Create the predictor set
  data$.__enclos_env__$private$set_predictor(predictor)

  # Write column names
  if (!is.null(colnames)) {
    data$set_colnames(colnames)
  }

  # Write categorical features
  if (!is.null(categorical_feature)) {
    data$set_categorical_feature(categorical_feature)
  }

  # Construct datasets, if needed
  data$construct()
  vaild_contain_train <- FALSE
  train_data_name <- "train"
  reduced_valid_sets <- list()

  # Parse validation datasets
  if (length(valids) > 0) {

    # Loop through all validation datasets using name
    for (key in names(valids)) {

      # Use names to get validation datasets
      valid_data <- valids[[key]]

      # Check for duplicate train/validation dataset
      if (identical(data, valid_data)) {
        vaild_contain_train <- TRUE
        train_data_name <- key
        next
      }

      # Update parameters, data
      valid_data$update_params(params)
      valid_data$set_reference(data)
      reduced_valid_sets[[key]] <- valid_data

    }

  }

  # Add printing log callback
  if (verbose > 0 && eval_freq > 0) {
    callbacks <- add.cb(callbacks, cb.print.evaluation(eval_freq))
  }

  # Add evaluation log callback
  if (record && length(valids) > 0) {
    callbacks <- add.cb(callbacks, cb.record.evaluation())
  }

  # Check for early stopping passed as parameter when adding early stopping callback
  early_stop <- c("early_stopping_round", "early_stopping_rounds", "early_stopping", "n_iter_no_change")
  if (any(names(params) %in% early_stop)) {
    if (params[[which(names(params) %in% early_stop)[1]]] > 0) {
      callbacks <- add.cb(callbacks, cb.early.stop(params[[which(names(params) %in% early_stop)[1]]], verbose = verbose))
    }
  } else {
    if (!is.null(early_stopping_rounds)) {
      if (early_stopping_rounds > 0) {
        callbacks <- add.cb(callbacks, cb.early.stop(early_stopping_rounds, verbose = verbose))
      }
    }
  }

  # "Categorize" callbacks
  cb <- categorize.callbacks(callbacks)

  # Construct booster with datasets
  booster <- Booster$new(params = params, train_set = data)
  if (vaild_contain_train) { booster$set_train_data_name(train_data_name) }
  for (key in names(reduced_valid_sets)) {
    booster$add_valid(reduced_valid_sets[[key]], key)
  }

  # Callback env
  env <- CB_ENV$new()
  env$model <- booster
  env$begin_iteration <- begin_iteration
  env$end_iteration <- end_iteration

  # Start training model using number of iterations to start and end with
  for (i in seq.int(from = begin_iteration, to = end_iteration)) {

    # Overwrite iteration in environment
    env$iteration <- i
    env$eval_list <- list()

    # Loop through "pre_iter" element
    for (f in cb$pre_iter) {
      f(env)
    }

    # Update one boosting iteration
    booster$update(fobj = fobj)

    # Prepare collection of evaluation results
    eval_list <- list()

    # Collection: Has validation dataset?
    if (length(valids) > 0) {

      # Validation has training dataset?
      if (vaild_contain_train) {
        eval_list <- append(eval_list, booster$eval_train(feval = feval))
      }

      # Has no validation dataset
      eval_list <- append(eval_list, booster$eval_valid(feval = feval))
    }

    # Write evaluation result in environment
    env$eval_list <- eval_list

    # Loop through env
    for (f in cb$post_iter) {
      f(env)
    }

    # Check for early stopping and break if needed
    if (env$met_early_stop) break

  }

  # When early stopping is not activated, we compute the best iteration / score ourselves by selecting the first metric and the first dataset
  if (record && length(valids) > 0 && is.na(env$best_score)) {
    if (env$eval_list[[1]]$higher_better[1] == TRUE) {
      booster$best_iter <- unname(which.max(unlist(booster$record_evals[[2]][[1]][[1]])))
      booster$best_score <- booster$record_evals[[2]][[1]][[1]][[booster$best_iter]]
    } else {
      booster$best_iter <- unname(which.min(unlist(booster$record_evals[[2]][[1]][[1]])))
      booster$best_score <- booster$record_evals[[2]][[1]][[1]][[booster$best_iter]]
    }
  }

  # Check for booster model conversion to predictor model
  if (reset_data) {

    # Store temporarily model data elsewhere
    booster_old <- list(best_iter = booster$best_iter,
                        best_score = booster$best_score,
                        record_evals = booster$record_evals)

    # Reload model
    booster <- lgb.load(model_str = booster$save_model_to_string())
    booster$best_iter <- booster_old$best_iter
    booster$best_score <- booster_old$best_score
    booster$record_evals <- booster_old$record_evals

  }

  # Return booster
  return(booster)

}
