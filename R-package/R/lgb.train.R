#' Main training logic for LightGBM
#' 
#' Main training logic for LightGBM
#' 
#' @param params List of parameters
#' @param data a \code{lgb.Dataset} object, used for training
#' @param nrounds number of training rounds
#' @param valids a list of \code{lgb.Dataset} object, used for validation
#' @param obj objective function, can be character or custom objective function
#' @param eval evaluation function, can be (list of) character or custom eval function
#' @param verbose verbosity for output
#'        if verbose > 0 , also will record iteration message to booster$record_evals
#' @param eval_freq evalutaion output frequence
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
#' data(agaricus.test, package='lightgbm')
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label=test$label)
#' params <- list(objective="regression", metric="l2")
#' valids <- list(test=dtest)
#' model <- lgb.train(params, dtrain, 100, valids, min_data=1, learning_rate=1, early_stopping_rounds=10)
#'
#' @rdname lgb.train
#' @export
lgb.train <- function(params=list(), data, nrounds=10, 
                      valids=list(), 
                      obj=NULL, eval=NULL,
                      verbose=1, eval_freq=1L,
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
    stop("lgb.train: data only accepts lgb.Dataset object")
  }
  if (length(valids) > 0) {
    if (typeof(valids) != "list" ||
        !all(sapply(valids, lgb.is.Dataset)))
      stop("valids must be a list of lgb.Dataset elements")
    evnames <- names(valids)
    if (is.null(evnames) || any(evnames == ""))
      stop("each element of the valids must have a name tag")
  }

  data$update_params(params)
  data$.__enclos_env__$private$set_predictor(predictor)
  if(!is.null(colnames)){
    data$set_colnames(colnames)
  }
  data$set_categorical_feature(categorical_feature)
  data$construct()
  vaild_contain_train <- FALSE
  train_data_name <- "train"
  reduced_valid_sets <- list()
  if(length(valids) > 0){
    for (key in names(valids)) {
      valid_data <- valids[[key]]
      if(identical(data, valid_data)){
        vaild_contain_train <- TRUE
        train_data_name <- key
        next
      }
      valid_data$update_params(params)
      valid_data$set_reference(data)
      reduced_valid_sets[[key]] <- valid_data
    }
  }
  # process callbacks
  if(eval_freq > 0){
    callbacks <- add.cb(callbacks, cb.print.evaluation(eval_freq))
  }

  if (verbose > 0 && length(valids) > 0) {
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
  booster <- Booster$new(params=params, train_set=data)
  if(vaild_contain_train){
    booster$set_train_data_name(train_data_name)
  }
  for (key in names(reduced_valid_sets)) {
    booster$add_valid(reduced_valid_sets[[key]], key)
  }

  # callback env

  env <- CB_ENV$new()
  env$model <- booster
  env$begin_iteration <- begin_iteration
  env$end_iteration <- end_iteration

  #start training
  for(i in begin_iteration:end_iteration){
    env$iteration <- i
    env$eval_list <- list()
    for (f in cb$pre_iter) f(env)
    # update one iter
    booster$update(fobj=fobj)

    # collect eval result
    eval_list <- list()
    if(length(valids) > 0){
      if(vaild_contain_train){
        eval_list <- append(eval_list, booster$eval_train(feval=feval))
      }
      eval_list <- append(eval_list, booster$eval_valid(feval=feval))
    }
    env$eval_list <- eval_list
    
    for (f in cb$post_iter) f(env)

    # met early stopping
    if(env$met_early_stop) break
  }

  return(booster)
}



