#' @importFrom R6 R6Class
CVBooster <- R6::R6Class(
  classname = "lgb.CVBooster",
  cloneable = FALSE,
  public = list(
    best_iter = -1,
    best_score = NA,
    record_evals = list(),
    boosters = list(),
    initialize = function(x) {
      self$boosters <- x
    },
    reset_parameter = function(new_params) {
      for (x in boosters) { x$reset_parameter(new_params) }
      self
    }
  )
)

#' @title Main CV logic for LightGBM
#' @description Cross validation logic used by LightGBM
#' @name lgb.cv
#' @inheritParams lgb_shared_params
#' @param nfold the original dataset is randomly partitioned into \code{nfold} equal size subsamples.
#' @param label vector of response values. Should be provided only when data is an R-matrix.
#' @param weight vector of response values. If not NULL, will set to dataset
#' @param obj objective function, can be character or custom objective function. Examples include
#'        \code{regression}, \code{regression_l1}, \code{huber},
#'        \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}
#' @param eval evaluation function, can be (list of) character or custom eval function
#' @param record Boolean, TRUE will record iteration message to \code{booster$record_evals}
#' @param showsd \code{boolean}, whether to show standard deviation of cross validation
#' @param stratified a \code{boolean} indicating whether sampling of folds should be stratified
#'        by the values of outcome labels.
#' @param folds \code{list} provides a possibility to use a list of pre-defined CV folds
#'        (each element must be a vector of test fold's indices). When folds are supplied,
#'        the \code{nfold} and \code{stratified} parameters are ignored.
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
#'
#' @return a trained model \code{lgb.CVBooster}.
#'
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' params <- list(objective = "regression", metric = "l2")
#' model <- lgb.cv(params,
#'                 dtrain,
#'                 10,
#'                 nfold = 3,
#'                 min_data = 1,
#'                 learning_rate = 1,
#'                 early_stopping_rounds = 5)
#' @export
lgb.cv <- function(params = list(),
                   data,
                   nrounds = 10,
                   nfold = 3,
                   label = NULL,
                   weight = NULL,
                   obj = NULL,
                   eval = NULL,
                   verbose = 1,
                   record = TRUE,
                   eval_freq = 1L,
                   showsd = TRUE,
                   stratified = TRUE,
                   folds = NULL,
                   init_model = NULL,
                   colnames = NULL,
                   categorical_feature = NULL,
                   early_stopping_rounds = NULL,
                   callbacks = list(),
                   reset_data = FALSE,
                   ...) {

  # Setup temporary variables
  addiction_params <- list(...)
  params <- append(params, addiction_params)
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
  n_trees <- c("num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees", "num_round", "num_rounds", "num_boost_round", "n_estimators")
  if (any(names(params) %in% n_trees)) {
    end_iteration <- begin_iteration + params[[which(names(params) %in% n_trees)[1]]] - 1
  } else {
    end_iteration <- begin_iteration + nrounds - 1
  }

  # Check for training dataset type correctness
  if (!lgb.is.Dataset(data)) {
    if (is.null(label)) {
      stop("Labels must be provided for lgb.cv")
    }
    data <- lgb.Dataset(data, label = label)
  }

  # Check for weights
  if (!is.null(weight)) {
    data$setinfo("weight", weight)
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

  # Check for folds
  if (!is.null(folds)) {

    # Check for list of folds or for single value
    if (!is.list(folds) || length(folds) < 2) {
      stop(sQuote("folds"), " must be a list with 2 or more elements that are vectors of indices for each CV-fold")
    }

    # Set number of folds
    nfold <- length(folds)

  } else {

    # Check fold value
    if (nfold <= 1) {
      stop(sQuote("nfold"), " must be > 1")
    }

    # Create folds
    folds <- generate.cv.folds(nfold,
                               nrow(data),
                               stratified,
                               getinfo(data, "label"),
                               getinfo(data, "group"),
                               params)

  }

  # Add printing log callback
  if (verbose > 0 && eval_freq > 0) {
    callbacks <- add.cb(callbacks, cb.print.evaluation(eval_freq))
  }

  # Add evaluation log callback
  if (record) {
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

  # Categorize callbacks
  cb <- categorize.callbacks(callbacks)

  # Construct booster using a list apply, check if requires group or not
  if (!is.list(folds[[1]])) {
    bst_folds <- lapply(seq_along(folds), function(k) {
      dtest <- slice(data, folds[[k]])
      dtrain <- slice(data, seq_len(nrow(data))[-folds[[k]]])
      setinfo(dtrain, "weight", getinfo(data, "weight")[-folds[[k]]])
      setinfo(dtrain, "init_score", getinfo(data, "init_score")[-folds[[k]]])
      setinfo(dtest, "weight", getinfo(data, "weight")[folds[[k]]])
      setinfo(dtest, "init_score", getinfo(data, "init_score")[folds[[k]]])
      booster <- Booster$new(params, dtrain)
      booster$add_valid(dtest, "valid")
      list(booster = booster)
    })
  } else {
    bst_folds <- lapply(seq_along(folds), function(k) {
      dtest <- slice(data, folds[[k]]$fold)
      dtrain <- slice(data, (seq_len(nrow(data)))[-folds[[k]]$fold])
      setinfo(dtrain, "weight", getinfo(data, "weight")[-folds[[k]]$fold])
      setinfo(dtrain, "init_score", getinfo(data, "init_score")[-folds[[k]]$fold])
      setinfo(dtrain, "group", getinfo(data, "group")[-folds[[k]]$group])
      setinfo(dtest, "weight", getinfo(data, "weight")[folds[[k]]$fold])
      setinfo(dtest, "init_score", getinfo(data, "init_score")[folds[[k]]$fold])
      setinfo(dtest, "group", getinfo(data, "group")[folds[[k]]$group])
      booster <- Booster$new(params, dtrain)
      booster$add_valid(dtest, "valid")
      list(booster = booster)
    })
  }


  # Create new booster
  cv_booster <- CVBooster$new(bst_folds)

  # Callback env
  env <- CB_ENV$new()
  env$model <- cv_booster
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
    msg <- lapply(cv_booster$boosters, function(fd) {
      fd$booster$update(fobj = fobj)
      fd$booster$eval_valid(feval = feval)
    })

    # Prepare collection of evaluation results
    merged_msg <- lgb.merge.cv.result(msg)

    # Write evaluation result in environment
    env$eval_list <- merged_msg$eval_list

    # Check for standard deviation requirement
    if(showsd) {
      env$eval_err_list <- merged_msg$eval_err_list
    }

    # Loop through env
    for (f in cb$post_iter) {
      f(env)
    }

    # Check for early stopping and break if needed
    if (env$met_early_stop) break

  }

  if (record && is.na(env$best_score)) {
    if (env$eval_list[[1]]$higher_better[1] == TRUE) {
      cv_booster$best_iter <- unname(which.max(unlist(cv_booster$record_evals[[2]][[1]][[1]])))
      cv_booster$best_score <- cv_booster$record_evals[[2]][[1]][[1]][[cv_booster$best_iter]]
    } else {
      cv_booster$best_iter <- unname(which.min(unlist(cv_booster$record_evals[[2]][[1]][[1]])))
      cv_booster$best_score <- cv_booster$record_evals[[2]][[1]][[1]][[cv_booster$best_iter]]
    }
  }

  if (reset_data) {
    lapply(cv_booster$boosters, function(fd) {
      # Store temporarily model data elsewhere
      booster_old <- list(best_iter = fd$booster$best_iter,
                          best_score = fd$booster$best_score,
                          record_evals = fd$booster$record_evals)
      # Reload model
      fd$booster <- lgb.load(model_str = fd$booster$save_model_to_string())
      fd$booster$best_iter <- booster_old$best_iter
      fd$booster$best_score <- booster_old$best_score
      fd$booster$record_evals <- booster_old$record_evals
    })
  }

  # Return booster
  return(cv_booster)

}

# Generates random (stratified if needed) CV folds
generate.cv.folds <- function(nfold, nrows, stratified, label, group, params) {

  # Check for group existence
  if (is.null(group)) {

    # Shuffle
    rnd_idx <- sample.int(nrows)

    # Request stratified folds
    if (isTRUE(stratified) && params$objective %in% c("binary", "multiclass") && length(label) == length(rnd_idx)) {

      y <- label[rnd_idx]
      y <- factor(y)
      folds <- lgb.stratified.folds(y, nfold)

    } else {

      # Make simple non-stratified folds
      folds <- list()

      # Loop through each fold
      for (i in seq_len(nfold)) {
        kstep <- length(rnd_idx) %/% (nfold - i + 1)
        folds[[i]] <- rnd_idx[seq_len(kstep)]
        rnd_idx <- rnd_idx[-seq_len(kstep)]
      }

    }

  } else {

    # When doing group, stratified is not possible (only random selection)
    if (nfold > length(group)) {
      stop("\n\tYou requested too many folds for the number of available groups.\n")
    }

    # Degroup the groups
    ungrouped <- inverse.rle(list(lengths = group, values = seq_along(group)))

    # Can't stratify, shuffle
    rnd_idx <- sample.int(length(group))

    # Make simple non-stratified folds
    folds <- list()

    # Loop through each fold
    for (i in seq_len(nfold)) {
      kstep <- length(rnd_idx) %/% (nfold - i + 1)
      folds[[i]] <- list(fold = which(ungrouped %in% rnd_idx[seq_len(kstep)]),
                         group = rnd_idx[seq_len(kstep)])
      rnd_idx <- rnd_idx[-seq_len(kstep)]
    }

  }

  # Return folds
  return(folds)

}

# Creates CV folds stratified by the values of y.
# It was borrowed from caret::lgb.stratified.folds and simplified
# by always returning an unnamed list of fold indices.
#' @importFrom stats quantile
lgb.stratified.folds <- function(y, k = 10) {

  ## Group the numeric data based on their magnitudes
  ## and sample within those groups.
  ## When the number of samples is low, we may have
  ## issues further slicing the numeric data into
  ## groups. The number of groups will depend on the
  ## ratio of the number of folds to the sample size.
  ## At most, we will use quantiles. If the sample
  ## is too small, we just do regular unstratified CV
  if (is.numeric(y)) {

    cuts <- length(y) %/% k
    if (cuts < 2) { cuts <- 2 }
    if (cuts > 5) { cuts <- 5 }
    y <- cut(y,
             unique(stats::quantile(y, probs = seq.int(0, 1, length.out = cuts))),
             include.lowest = TRUE)

  }

  if (k < length(y)) {

    ## Reset levels so that the possible levels and
    ## the levels in the vector are the same
    y <- factor(as.character(y))
    numInClass <- table(y)
    foldVector <- vector(mode = "integer", length(y))

    ## For each class, balance the fold allocation as far
    ## as possible, then resample the remainder.
    ## The final assignment of folds is also randomized.

    for (i in seq_along(numInClass)) {

      ## Create a vector of integers from 1:k as many times as possible without
      ## going over the number of samples in the class. Note that if the number
      ## of samples in a class is less than k, nothing is producd here.
      seqVector <- rep(seq_len(k), numInClass[i] %/% k)

      ## Add enough random integers to get  length(seqVector) == numInClass[i]
      if (numInClass[i] %% k > 0) {
        seqVector <- c(seqVector, sample.int(k, numInClass[i] %% k))
      }

      ## Shuffle the integers for fold assignment and assign to this classes's data
      foldVector[y == dimnames(numInClass)$y[i]] <- sample(seqVector)

    }

  } else {

    foldVector <- seq(along = y)

  }

  # Return data
  out <- split(seq(along = y), foldVector)
  names(out) <- NULL
  out
}

lgb.merge.cv.result <- function(msg, showsd = TRUE) {

  # Get CV message length
  if (length(msg) == 0) {
    stop("lgb.cv: size of cv result error")
  }

  # Get evaluation message length
  eval_len <- length(msg[[1]])

  # Is evaluation message empty?
  if (eval_len == 0) {
    stop("lgb.cv: should provide at least one metric for CV")
  }

  # Get evaluation results using a list apply
  eval_result <- lapply(seq_len(eval_len), function(j) {
    as.numeric(lapply(seq_along(msg), function(i) {
      msg[[i]][[j]]$value }))
  })

  # Get evaluation
  ret_eval <- msg[[1]]

  # Go through evaluation length items
  for (j in seq_len(eval_len)) {
    ret_eval[[j]]$value <- mean(eval_result[[j]])
  }

  # Preinit evaluation error
  ret_eval_err <- NULL

  # Check for standard deviation
  if (showsd) {

    # Parse standard deviation
    for (j in seq_len(eval_len)) {
      ret_eval_err <- c(ret_eval_err,
                        sqrt(mean(eval_result[[j]] ^ 2) - mean(eval_result[[j]]) ^ 2))
    }

    # Convert to list
    ret_eval_err <- as.list(ret_eval_err)

  }

  # Return errors
  list(eval_list = ret_eval,
       eval_err_list = ret_eval_err)

}
