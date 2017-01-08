CB_ENV <- R6Class(
  "lgb.cb_env",
  cloneable=FALSE,
  public = list(
    model=NULL,
    iteration=NULL,
    begin_iteration=NULL,
    end_iteration=NULL,
    eval_list=list(),
    eval_err_list=list(),
    best_iter=-1,
    met_early_stop=FALSE
  )
)

cb.reset.parameters <- function(new_params) {
  if (typeof(new_params) != "list") 
    stop("'new_params' must be a list")
  pnames <- gsub("\\.", "_", names(new_params))
  nrounds <- NULL
  
  # run some checks in the begining
  init <- function(env) {
    nrounds <<- env$end_iteration - env$begin_iteration + 1
    
    if (is.null(env$model))
      stop("Env should has 'model'")
    
    # Some parameters are not allowed to be changed,
    # since changing them would simply wreck some chaos
    not_allowed <- pnames %in% 
      c('num_class', 'metric', 'boosting_type')
    if (any(not_allowed))
      stop('Parameters ', paste(pnames[not_allowed]), " cannot be changed during boosting.")
    
    for (n in pnames) {
      p <- new_params[[n]]
      if (is.function(p)) {
        if (length(formals(p)) != 2)
          stop("Parameter '", n, "' is a function but not of two arguments")
      } else if (is.numeric(p) || is.character(p)) {
        if (length(p) != nrounds)
          stop("Length of '", n, "' has to be equal to 'nrounds'")
      } else {
        stop("Parameter '", n, "' is not a function or a vector")
      }
    }
  }
  
  callback <- function(env) {
    if (is.null(nrounds))
      init(env)
    
    i <- env$iteration - env$begin_iteration
    pars <- lapply(new_params, function(p) {
      if (is.function(p))
        return(p(i, nrounds))
      p[i]
    })
    # to-do check pars
    if (!is.null(env$model)) {
      env$model$reset_parameter(pars)
    } 
  }
  attr(callback, 'call') <- match.call()
  attr(callback, 'is_pre_iteration') <- TRUE
  attr(callback, 'name') <- 'cb.reset.parameters'
  return(callback)
}

# Format the evaluation metric string
format.eval.string <- function(eval_res, eval_err=NULL) {
  if (is.null(eval_res))
    stop('no evaluation results')
  if (length(eval_res) == 0)
    stop('no evaluation results')
  if (!is.null(eval_err)) {
    res <- sprintf('%s\'s %s:%g+%g', eval_res$data_name, eval_res$name, eval_res$value, eval_err)
  } else {
    res <- sprintf('%s\'s %s:%g', eval_res$data_name, eval_res$name, eval_res$value)
  }
  return(res)
}

merge.eval.string <- function(env){
  if(length(env$eval_list) <= 0){
    return("")
  }
  msg <- list(sprintf('[%d]:',env$iteration))
  is_eval_err <- FALSE
  if(length(env$eval_err_list) > 0){
    is_eval_err <- TRUE
  }
  for(j in 1:length(env$eval_list)) {
    eval_err <- NULL
    if(is_eval_err){
      eval_err <- env$eval_err_list[[j]]
    }
    msg <- c(msg, format.eval.string(env$eval_list[[j]],eval_err))
  }
  return(paste0(msg, collapse='\t'))
}

cb.print.evaluation <- function(period=1){
  callback <- function(env){
    if(period > 0){
      i <- env$iteration
      if( (i - 1) %% period == 0
         | i == env$begin_iteration
         | i == env$end_iteration ){
        cat(merge.eval.string(env), "\n")
      }
    }
  }
  attr(callback, 'call') <- match.call()
  attr(callback, 'name') <- 'cb.print.evaluation'
  return(callback)
}

cb.record.evaluation <- function() {
  callback <- function(env){
    if(length(env$eval_list) <= 0) return()
    is_eval_err <- FALSE
    if(length(env$eval_err_list) > 0){
      is_eval_err <- TRUE
    }
    if(length(env$model$record_evals) == 0){
      for(j in 1:length(env$eval_list)) {
        data_name <- env$eval_list[[j]]$data_name
        name <- env$eval_list[[j]]$name
        env$model$record_evals$start_iter <- env$begin_iteration
        if(is.null(env$model$record_evals[[data_name]])){
          env$model$record_evals[[data_name]] <- list()
        }
        env$model$record_evals[[data_name]][[name]] <- list()
        env$model$record_evals[[data_name]][[name]]$eval <- list()
        env$model$record_evals[[data_name]][[name]]$eval_err <- list()
      }
    }
    for(j in 1:length(env$eval_list)) {
      eval_res <- env$eval_list[[j]]
      eval_err <- NULL
      if(is_eval_err){
        eval_err <- env$eval_err_list[[j]]
      }
      data_name <- eval_res$data_name
      name <- eval_res$name
      env$model$record_evals[[data_name]][[name]]$eval <- c(env$model$record_evals[[data_name]][[name]]$eval, eval_res$value)
      env$model$record_evals[[data_name]][[name]]$eval_err <- c(env$model$record_evals[[data_name]][[name]]$eval_err, eval_err)
    }
    
  }
  attr(callback, 'call') <- match.call()
  attr(callback, 'name') <- 'cb.record.evaluation'
  return(callback)
}

cb.early.stop <- function(stopping_rounds, verbose=TRUE) {
  # state variables
  factor_to_bigger_better <- NULL
  best_iter <- NULL
  best_score <- NULL
  best_msg <- NULL
  eval_len <- NULL
  init <- function(env) {
    eval_len <<-  length(env$eval_list)
    if (eval_len == 0)
      stop("For early stopping, valids must have at least one element")
    
    if (verbose)
      cat("Will train until hasn't improved in ", 
          stopping_rounds, " rounds.\n\n", sep = '')

    factor_to_bigger_better <<- rep(1.0, eval_len)
    best_iter <<- rep(-1, eval_len)
    best_score <<- rep(-Inf, eval_len)
    best_msg <<- list()
    for(i in 1:eval_len){
      best_msg <<- c(best_msg, "")
      if(!env$eval_list[[i]]$higher_better){
        factor_to_bigger_better[i] <<- -1.0
      }
    }
  }
  
  callback <- function(env, finalize = FALSE) {
    if (is.null(eval_len))
      init(env)
    cur_iter <- env$iteration
    for(i in 1:eval_len){
      score <- env$eval_list[[i]]$value * factor_to_bigger_better[i]
      if(score > best_score[i]){
        best_score[i] <<- score
        best_iter[i] <<- cur_iter
        if(verbose){
          best_msg[[i]] <<- as.character(merge.eval.string(env))
        }
      } else {
        if(cur_iter - best_iter[i] >= stopping_rounds){
          if(!is.null(env$model)){
            env$model$best_iter <- best_iter[i]
          }
          if(verbose){
            cat('Early stopping, best iteration is:',"\n")
            cat(best_msg[[i]],"\n")
          }
          env$best_iter <- best_iter[i]
          env$met_early_stop <- TRUE
        }
      }
    }
  }
  attr(callback, 'call') <- match.call()
  attr(callback, 'name') <- 'cb.early.stop'
  return(callback)
}

# Extract callback names from the list of callbacks
callback.names <- function(cb_list) {
  unlist(lapply(cb_list, function(x) attr(x, 'name')))
}

add.cb <- function(cb_list, cb) {
  cb_list <- c(cb_list, cb)
  names(cb_list) <- callback.names(cb_list)
  if ('cb.early.stop' %in% names(cb_list)) {
    cb_list <- c(cb_list, cb_list['cb.early.stop'])
    # this removes only the first one
    cb_list['cb.early.stop'] <- NULL 
  }
  cb_list
}

categorize.callbacks <- function(cb_list) {
  list(
    pre_iter = Filter(function(x) {
        pre <- attr(x, 'is_pre_iteration')
        !is.null(pre) && pre 
      }, cb_list),
    post_iter = Filter(function(x) {
        pre <- attr(x, 'is_pre_iteration')
        is.null(pre) || !pre
      }, cb_list)
  )
}
