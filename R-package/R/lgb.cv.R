CVBooster <- R6Class(
  "lgb.CVBooster",
  cloneable=FALSE,
  public = list(
    best_iter = -1,
    record_evals = list(),
    boosters=list(),
    initialize=function(x){
      self$boosters <- x
    },
    reset_parameter=function(new_paramas){
      for(x in boosters){
        x$reset_parameter(new_paramas)
      }
      return(self)
    }
  )
)


#' Main CV logic for LightGBM
#' 
#' Main CV logic for LightGBM
#' 
#' @param params List of parameters
#' @param data a \code{lgb.Dataset} object, used for CV
#' @param nrounds number of CV rounds
#' @param nfold the original dataset is randomly partitioned into \code{nfold} equal size subsamples. 
#' @param label vector of response values. Should be provided only when data is an R-matrix.
#' @param weight vector of response values. If not NULL, will set to dataset 
#' @param obj objective function, can be character or custom objective function
#' @param eval evaluation function, can be (list of) character or custom eval function
#' @param verbose verbosity for output
#'        if verbose > 0 , also will record iteration message to booster$record_evals
#' @param eval_freq evalutaion output frequence
#' @param showsd \code{boolean}, whether to show standard deviation of cross validation
#' @param stratified a \code{boolean} indicating whether sampling of folds should be stratified 
#'        by the values of outcome labels.
#' @param folds \code{list} provides a possibility to use a list of pre-defined CV folds
#'        (each element must be a vector of test fold's indices). When folds are supplied, 
#'        the \code{nfold} and \code{stratified} parameters are ignored.
#' @param init_model path of model file of \code{lgb.Booster} object, will continue train from this model
#' @param colnames feature names, if not null, will use this to overwrite the names in dataset
#' @param categorical_feature list of str or int
#'        type int represents index,
#'        type str represents feature names
#' @param early_stopping_rounds int
#'        Activates early stopping.
#'        Requires at least one validation data and one metric
#'        If there's more than one, will check all of them
#'        Returns the model with (best_iter + early_stopping_rounds)
#'        If early stopping occurs, the model will have 'best_iter' field
#' @param callbacks list of callback functions
#'        List of callback functions that are applied at each iteration.
#' @param ... other parameters, see parameters.md for more informations
#' @return a trained booster model \code{lgb.Booster}. 
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' params <- list(objective="regression", metric="l2")
#' model <- lgb.cv(params, dtrain, 10, nfold=5, min_data=1, learning_rate=1, early_stopping_rounds=10)
#'
#' @rdname lgb.train
#' @export
lgb.cv <- function(params=list(), data, nrounds=10, nfold=3,
                      label = NULL, weight = NULL,
                      obj=NULL, eval=NULL,
                      verbose=1, eval_freq=1L, showsd = TRUE,
                      stratified = TRUE, folds = NULL, 
                      init_model=NULL, 
                      colnames=NULL,
                      categorical_feature=NULL,
                      early_stopping_rounds=NULL,
                      callbacks=list(), ...) {
  addiction_params <- list(...)
  params <- append(params, addiction_params)
  params$verbose <- verbose
  params <- lgb.check.obj(params, obj)
  params <- lgb.check.eval(params, eval)
  fobj <- NULL
  feval <- NULL
  if(typeof(params$objective) == "closure"){
    fobj <- params$objective
    params$objective <- "NONE"
  } 
  if (typeof(eval) == "closure"){
    feval <- eval
  }
  lgb.check.params(params)
  predictor <- NULL
  if(is.character(init_model)){
    predictor <- Predictor$new(init_model)
  } else if(lgb.is.Booster(init_model)) {
    predictor <- init_model$to_predictor()
  }
  begin_iteration <- 1
  if(!is.null(predictor)){
    begin_iteration <- predictor$current_iter() + 1
  }
  end_iteration <- begin_iteration + nrounds - 1

  # check dataset
  if(!lgb.is.Dataset(data)){
    if(is.null(label)){
       stop("Labels must be provided for lgb.cv")
    }
    data <- lgb.Dataset(data, label=label)
  }

  if(!is.null(weight)) data$set_info("weight", weight)

  data$update_params(params)
  data$.__enclos_env__$private$set_predictor(predictor)
  if(!is.null(colnames)){
    data$set_colnames(colnames)
  }
  data$set_categorical_feature(categorical_feature)
  data$construct()

  # CV folds
  if(!is.null(folds)) {
    if(class(folds) != "list" || length(folds) < 2)
      stop("'folds' must be a list with 2 or more elements that are vectors of indices for each CV-fold")
    nfold <- length(folds)
  } else {
    if (nfold <= 1)
      stop("'nfold' must be > 1")
    folds <- generate.cv.folds(nfold, nrow(data), stratified, getinfo(data, 'label'), params)
  }

  # process callbacks
  if(eval_freq > 0){
    callbacks <- add.cb(callbacks, cb.print.evaluation(eval_freq))
  }

  if (verbose > 0) {
    callbacks <- add.cb(callbacks, cb.record.evaluation())
  }

  # Early stopping callback
  if (!is.null(early_stopping_rounds)) {
    if(early_stopping_rounds > 0){
      callbacks <- add.cb(callbacks, cb.early.stop(early_stopping_rounds, verbose=verbose))
    }
  }

  cb <- categorize.callbacks(callbacks)

  # construct booster

  bst_folds <- lapply(1:length(folds), function(k) {
    dtest  <- slice(data, folds[[k]])
    dtrain <- slice(data, unlist(folds[-k]))
    booster <- Booster$new(params, dtrain)
    booster$add_valid(dtest, "valid")
    list(booster=booster)
  })

  cv_booster <- CVBooster$new(bst_folds)

  # callback env

  env <- CB_ENV$new()
  env$model <- cv_booster
  env$begin_iteration <- begin_iteration
  env$end_iteration <- end_iteration

  #start training
  for(i in begin_iteration:end_iteration){
    env$iteration <- i
    env$eval_list <- list()
    for (f in cb$pre_iter) f(env)
    # update one iter
    msg <- lapply(cv_booster$boosters, function(fd) {
      fd$booster$update(fobj=fobj)
      fd$booster$eval_valid(feval=feval)
    })

    merged_msg <- lgb.merge.cv.result(msg)

    env$eval_list <- merged_msg$eval_list
    if(showsd) env$eval_err_list <- merged_msg$eval_err_list
    
    for (f in cb$post_iter) f(env)

    # met early stopping
    if(env$met_early_stop) break
  }

  return(cv_booster)
}

# Generates random (stratified if needed) CV folds
generate.cv.folds <- function(nfold, nrows, stratified, label, params) {
  
  # cannot do it for rank
  if (exists('objective', where=params) &&
      is.character(params$objective) &&
      params$objective == 'lambdarank') {
    stop("\n\tAutomatic generation of CV-folds is not implemented for lambdarank!\n",
         "\tConsider providing pre-computed CV-folds through the 'folds=' parameter.\n")
  }
  # shuffle
  rnd_idx <- sample(1:nrows)
  if (stratified &&
      length(label) == length(rnd_idx)) {
    y <- label[rnd_idx]
    y <- factor(y)
    folds <- lgb.stratified.folds(y, nfold)
  } else {
    # make simple non-stratified folds
    kstep <- length(rnd_idx) %/% nfold
    folds <- list()
    for (i in 1:(nfold - 1)) {
      folds[[i]] <- rnd_idx[1:kstep]
      rnd_idx <- rnd_idx[-(1:kstep)]
    }
    folds[[nfold]] <- rnd_idx
  }
  return(folds)
}

# Creates CV folds stratified by the values of y.
# It was borrowed from caret::lgb.stratified.folds and simplified
# by always returning an unnamed list of fold indices.
lgb.stratified.folds <- function(y, k = 10)
{
  if (is.numeric(y)) {
    ## Group the numeric data based on their magnitudes
    ## and sample within those groups.

    ## When the number of samples is low, we may have
    ## issues further slicing the numeric data into
    ## groups. The number of groups will depend on the
    ## ratio of the number of folds to the sample size.
    ## At most, we will use quantiles. If the sample
    ## is too small, we just do regular unstratified
    ## CV
    cuts <- floor(length(y) / k)
    if (cuts < 2) cuts <- 2
    if (cuts > 5) cuts <- 5
    y <- cut(y,
             unique(stats::quantile(y, probs = seq(0, 1, length = cuts))),
             include.lowest = TRUE)
  }

  if (k < length(y)) {
    ## reset levels so that the possible levels and
    ## the levels in the vector are the same
    y <- factor(as.character(y))
    numInClass <- table(y)
    foldVector <- vector(mode = "integer", length(y))

    ## For each class, balance the fold allocation as far
    ## as possible, then resample the remainder.
    ## The final assignment of folds is also randomized.
    for (i in 1:length(numInClass)) {
      ## create a vector of integers from 1:k as many times as possible without
      ## going over the number of samples in the class. Note that if the number
      ## of samples in a class is less than k, nothing is producd here.
      seqVector <- rep(1:k, numInClass[i] %/% k)
      ## add enough random integers to get  length(seqVector) == numInClass[i]
      if (numInClass[i] %% k > 0) seqVector <- c(seqVector, sample(1:k, numInClass[i] %% k))
      ## shuffle the integers for fold assignment and assign to this classes's data
      foldVector[which(y == dimnames(numInClass)$y[i])] <- sample(seqVector)
    }
  } else {
    foldVector <- seq(along = y)
  }

  out <- split(seq(along = y), foldVector)
  names(out) <- NULL
  out
}

lgb.merge.cv.result <- function(msg, showsd=TRUE){
  if(length(msg) == 0){
    stop("lgb.cv: size of cv result error")
  }
  eval_len <- length(msg[[1]])
  if(eval_len == 0){
    stop("lgb.cv: should provide at least metric for CV")
  }
  eval_result <- lapply(1:eval_len, function(j) {
    as.numeric(lapply(1:length(msg), function(i){
      msg[[i]][[j]]$value
    }))
  })
  ret_eval <- msg[[1]]
  for(j in 1:eval_len){
    ret_eval[[j]]$value <- mean(eval_result[[j]])
  }
  ret_eval_err <- NULL
  if(showsd){
    for(j in 1:eval_len){
      ret_eval_err <- c( ret_eval_err, sqrt( mean(eval_result[[j]]^2) - mean(eval_result[[j]])^2 ))
    }
    ret_eval_err <- as.list(ret_eval_err)
  }
  return(list(eval_list=ret_eval, eval_err_list=ret_eval_err))
}