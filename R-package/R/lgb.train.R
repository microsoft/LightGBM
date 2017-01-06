#' Main training logic
#' 
#' Main training logic
#' 
#' @rdname lgb.train
#' @export
lgb.train <- function(params=list(), data, nrounds, 
                      valids=list(), 
                      obj=NULL, eval=NULL,
                      verbose=1, eval_freq=1,
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
    feavl <- eval
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

  vaild_contain_train <- FALSE
  train_data_name <- "training"
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
        eval_list <- append(eval_list, booster$eval_train())
      }
      eval_list <- append(eval_list, booster$eval_valid())
    }

    env$eval_list <- eval_list
    
    for (f in cb$post_iter) f(env)

    # met early stopping
    if(env$met_early_stop) break
  }

  return(booster)
}



