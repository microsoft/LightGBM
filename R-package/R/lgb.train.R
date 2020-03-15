#' @name lgb.train
#' @title Main training logic for LightGBM
#' @description Logic to train with LightGBM
#' @inheritParams lgb_shared_params
#' @param valids a list of \code{lgb.Dataset} objects, used for validation
#' @param obj objective function, can be character or custom objective function. Examples include
#'            \code{regression}, \code{regression_l1}, \code{huber},
#'            \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}
#' @param eval evaluation function(s). This can be a character vector, function, or list with a mixture of
#'             strings and functions.
#'
#'             \itemize{
#'                 \item{\bold{a. character vector}:
#'                     If you provide a character vector to this argument, it should contain strings with valid
#'                     evaluation metrics. See \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric}{The "metric" section of the documentation}
#'                     for a list of valid metrics.
#'                 }
#'                 \item{\bold{b. function}:
#'                      You can provide a custom evaluation function. This
#'                      should accept the keyword arguments \code{preds} and \code{dtrain} and should return a named
#'                      list with three elements:
#'                      \itemize{
#'                          \item{\code{name}: A string with the name of the metric, used for printing and storing results.}
#'                          \item{\code{value}: A single number indicating the value of the metric for the given predictions and true values}
#'                          \item{
#'                              \code{higher_better}: A boolean indicating whether higher values indicate a better fit.
#'                              For example, this would be \code{FALSE} for metrics like MAE or RMSE.
#'                          }
#'                      }
#'                 }
#'                 \item{\bold{c. list}:
#'                     If a list is given, it should only contain character vectors and functions. These should follow the
#'                     requirements from the descriptions above.
#'                 }
#'             }
#' @param record Boolean, TRUE will record iteration message to \code{booster$record_evals}
#' @param colnames feature names, if not null, will use this to overwrite the names in dataset
#' @param categorical_feature list of str or int
#'                            type int represents index,
#'                            type str represents feature names
#' @param callbacks List of callback functions that are applied at each iteration.
#' @param reset_data Boolean, setting it to TRUE (not the default value) will transform the
#'                   booster model into a predictor model which frees up memory and the
#'                   original datasets
#' @param ... other parameters, see Parameters.rst for more information. A few key parameters:
#'            \itemize{
#'                \item{\code{boosting}: Boosting type. \code{"gbdt"}, \code{"rf"}, \code{"dart"} or \code{"goss"}.}
#'                \item{\code{num_leaves}: Maximum number of leaves in one tree.}
#'                \item{\code{max_depth}: Limit the max depth for tree model. This is used to deal with
#'                                 overfit when #data is small. Tree still grow by leaf-wise.}
#'                \item{\code{num_threads}: Number of threads for LightGBM. For the best speed, set this to
#'                                   the number of real CPU cores, not the number of threads (most
#'                                   CPU using hyper-threading to generate 2 threads per CPU core).}
#'            }
#' @section Early Stopping:
#'
#'          "early stopping" refers to stopping the training process if the model's performance on a given
#'          validation set does not improve for several consecutive iterations.
#'
#'          If multiple arguments are given to \code{eval}, their order will be preserved. If you enable
#'          early stopping by setting \code{early_stopping_rounds} in \code{params}, by default all
#'          metrics will be considered for early stopping.
#'
#'          If you want to only consider the first metric for early stopping, pass
#'          \code{first_metric_only = TRUE} in \code{params}. Note that if you also specify \code{metric}
#'          in \code{params}, that metric will be considered the "first" one. If you omit \code{metric},
#'          a default metric will be used based on your choice for the parameter \code{obj} (keyword argument)
#'          or \code{objective} (passed into \code{params}).
#'
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
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 10L
#'   , valids = valids
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , early_stopping_rounds = 5L
#' )
#' @export
lgb.train <- function(params = list(),
                      data,
                      nrounds = 10L,
                      valids = list(),
                      obj = NULL,
                      eval = NULL,
                      verbose = 1L,
                      record = TRUE,
                      eval_freq = 1L,
                      init_model = NULL,
                      colnames = NULL,
                      categorical_feature = NULL,
                      early_stopping_rounds = NULL,
                      callbacks = list(),
                      reset_data = FALSE,
                      ...) {

  # validate inputs early to avoid unnecessary computation
  if (nrounds <= 0L) {
    stop("nrounds should be greater than zero")
  }
  if (!lgb.is.Dataset(data)) {
    stop("lgb.train: data must be an lgb.Dataset instance")
  }
  if (length(valids) > 0L) {
    if (!identical(class(valids), "list") || !all(vapply(valids, lgb.is.Dataset, logical(1L)))) {
      stop("lgb.train: valids must be a list of lgb.Dataset elements")
    }
    evnames <- names(valids)
    if (is.null(evnames) || !all(nzchar(evnames))) {
      stop("lgb.train: each element of valids must have a name")
    }
  }

  # Setup temporary variables
  additional_params <- list(...)
  params <- append(params, additional_params)
  params$verbose <- verbose
  params <- lgb.check.obj(params, obj)
  params <- lgb.check.eval(params, eval)
  fobj <- NULL
  eval_functions <- list(NULL)

  # Check for objective (function or not)
  if (is.function(params$objective)) {
    fobj <- params$objective
    params$objective <- "NONE"
  }

  # If loss is a single function, store it as a 1-element list
  # (for backwards compatibility). If it is a list of functions, store
  # all of them
  if (is.function(eval)) {
    eval_functions <- list(eval)
  }
  if (methods::is(eval, "list")) {
    eval_functions <- Filter(
      f = function(eval_element){
        is.function(eval_element)
      }
      , x = eval
    )
  }

  # Init predictor to empty
  predictor <- NULL

  # Check for boosting from a trained model
  if (is.character(init_model)) {
    predictor <- Predictor$new(init_model)
  } else if (lgb.is.Booster(init_model)) {
    predictor <- init_model$to_predictor()
  }

  # Set the iteration to start from / end to (and check for boosting from a trained model, again)
  begin_iteration <- 1L
  if (!is.null(predictor)) {
    begin_iteration <- predictor$current_iter() + 1L
  }

  # Check for number of rounds passed as parameter - in case there are multiple ones, take only the first one
  n_trees <- .PARAMETER_ALIASES()[["num_iterations"]]
  if (any(names(params) %in% n_trees)) {
    end_iteration <- begin_iteration + params[[which(names(params) %in% n_trees)[1L]]] - 1L
  } else {
    end_iteration <- begin_iteration + nrounds - 1L
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
  valid_contain_train <- FALSE
  train_data_name <- "train"
  reduced_valid_sets <- list()

  # Parse validation datasets
  if (length(valids) > 0L) {

    # Loop through all validation datasets using name
    for (key in names(valids)) {

      # Use names to get validation datasets
      valid_data <- valids[[key]]

      # Check for duplicate train/validation dataset
      if (identical(data, valid_data)) {
        valid_contain_train <- TRUE
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
  if (verbose > 0L && eval_freq > 0L) {
    callbacks <- add.cb(callbacks, cb.print.evaluation(eval_freq))
  }

  # Add evaluation log callback
  if (record && length(valids) > 0L) {
    callbacks <- add.cb(callbacks, cb.record.evaluation())
  }

  # If early stopping was passed as a parameter in params(), prefer that to keyword argument
  # early_stopping_rounds by overwriting the value in 'early_stopping_rounds'
  early_stop <- .PARAMETER_ALIASES()[["early_stopping_round"]]
  early_stop_param_indx <- names(params) %in% early_stop
  if (any(early_stop_param_indx)) {
    first_early_stop_param <- which(early_stop_param_indx)[[1L]]
    first_early_stop_param_name <- names(params)[[first_early_stop_param]]
    early_stopping_rounds <- params[[first_early_stop_param_name]]
  }

  # Did user pass parameters that indicate they want to use early stopping?
  using_early_stopping_via_args <- !is.null(early_stopping_rounds)

  boosting_param_names <- .PARAMETER_ALIASES()[["boosting"]]
  using_dart <- any(
    sapply(
      X = boosting_param_names
      , FUN = function(param) {
        identical(params[[param]], "dart")
      }
    )
  )

  # Cannot use early stopping with 'dart' boosting
  if (using_dart) {
    warning("Early stopping is not available in 'dart' mode.")
    using_early_stopping_via_args <- FALSE

    # Remove the cb.early.stop() function if it was passed in to callbacks
    callbacks <- Filter(
      f = function(cb_func) {
        !identical(attr(cb_func, "name"), "cb.early.stop")
      }
      , x = callbacks
    )
  }

  # If user supplied early_stopping_rounds, add the early stopping callback
  if (using_early_stopping_via_args) {
    callbacks <- add.cb(
      callbacks
      , cb.early.stop(
        stopping_rounds = early_stopping_rounds
        , first_metric_only = isTRUE(params[["first_metric_only"]])
        , verbose = verbose
      )
    )
  }

  # "Categorize" callbacks
  cb <- categorize.callbacks(callbacks)

  # Construct booster with datasets
  booster <- Booster$new(params = params, train_set = data)
  if (valid_contain_train) {
    booster$set_train_data_name(train_data_name)
  }
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
    if (length(valids) > 0L) {

      for (eval_function in eval_functions){

        # Validation has training dataset?
        if (valid_contain_train) {
          eval_list <- append(eval_list, booster$eval_train(feval = eval_function))
        }

        # Has no validation dataset
        eval_list <- append(eval_list, booster$eval_valid(feval = eval_function))
      }

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

  # When early stopping is not activated, we compute the best iteration / score ourselves by
  # selecting the first metric and the first dataset
  if (record && length(valids) > 0L && is.na(env$best_score)) {
    if (env$eval_list[[1L]]$higher_better[1L] == TRUE) {
      booster$best_iter <- unname(which.max(unlist(booster$record_evals[[2L]][[1L]][[1L]])))
      booster$best_score <- booster$record_evals[[2L]][[1L]][[1L]][[booster$best_iter]]
    } else {
      booster$best_iter <- unname(which.min(unlist(booster$record_evals[[2L]][[1L]][[1L]])))
      booster$best_score <- booster$record_evals[[2L]][[1L]][[1L]][[booster$best_iter]]
    }
  }

  # Check for booster model conversion to predictor model
  if (reset_data) {

    # Store temporarily model data elsewhere
    booster_old <- list(
      best_iter = booster$best_iter
      , best_score = booster$best_score
      , record_evals = booster$record_evals
    )

    # Reload model
    booster <- lgb.load(model_str = booster$save_model_to_string())
    booster$best_iter <- booster_old$best_iter
    booster$best_score <- booster_old$best_score
    booster$record_evals <- booster_old$record_evals

  }

  # Return booster
  return(booster)

}
