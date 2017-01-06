CB_ENV <- R6Class(
  "lgb.cb_env",
  cloneable=FALSE,
  public = list(
    model=NULL,
    cvfolds=NULL,
    iteration=NULL,
    begin_iteration=NULL,
    end_iteration=NULL,
    eval_list=list(),
    eval_err_list=list(),
    record_evals=list(),
    best_iter=-1,
    met_early_stop=FALSE
  )
)

#' 
#' @export
cb.reset.parameters <- function(new_params) {
  if (typeof(new_params) != "list") 
    stop("'new_params' must be a list")
  pnames <- gsub("\\.", "_", names(new_params))
  nrounds <- NULL
  
  # run some checks in the begining
  init <- function(env) {
    nrounds <<- env$end_iteration - env$begin_iteration + 1
    
    if (is.null(env$model) & is.null(env$cvfolds))
      stop("Env should has neither 'model' nor 'cvfolds'")
    
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
  
  callback <- function(env = parent.frame()) {
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
    } else {
      for (fd in env$cvfolds)
        fd$reset_parameter(pars)
    }
  }
  attr(callback, 'is_pre_iteration') <- TRUE
  attr(callback, 'name') <- 'cb.reset.parameters'
  attr(callback, 'order') <- 10
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
  msg <- list(sprintf('[%d]:',i))
  is_eval_err <- FALSE
  if(length(env$eval_err_list) > 0){
    is_eval_err <- TRUE
  }
  for( j in 1:length(env$eval_list)) {
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
        print(merge.eval.string(env))
      }
    }
  }
  attr(callback, 'name') <- 'cb.print.evaluation'
  attr(callback, 'order') <- 10
  return(callback)
}

cb.record.evaluation <- function() {
  init <- function(env){

  }
  callback <- function(env){
    if(length(env$record_evals) == 0){
      init(env)
    }
    is_eval_err <- FALSE
    if(length(eval_err_list) > 0){
      is_eval_err <- TRUE
    }
    for( j in 1:length(env$eval_list)) {
      eval_res <- env$eval_list[[j]]
      eval_err <- NULL
      if(is_err){
        eval_err <- env$eval_err_list[[j]]
      }
      env$record_evals <- c(env$record_evals, list(c(iter=env$iteration, c(eval_res, eval_err))))
    }
    
  }
  attr(callback, 'name') <- 'cb.record.evaluation'
  attr(callback, 'order') <- 20
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
    eval_len <-  length(env$eval_list) 
    if (eval_len == 0)
      stop("For early stopping, valids must have at least one element")
    
    if (verbose)
      cat("Will train until ", metric_name, " hasn't improved in ", 
          stopping_rounds, " rounds.\n\n", sep = '')

    factor_to_bigger_better <- rep(1.0, eval_len)
    best_iter <- rep(-1, eval_len)
    best_score <- rep(-Inf, eval_len)
    best_msg <- list()
    for(i in 1:eval_len){
      best_msg <- c(best_msg, "")
      if(!env$eval_list[[i]]$higher_better){
        factor_to_bigger_better[i] <- -1.0
      }
    }
  }
  
  callback <- function(env = parent.frame(), finalize = FALSE) {
    if (is.null(eval_len))
      init(env)
    
    cur_iter <- env$iteration

    for(i in 1:eval_len){
      score <- env$eval_list[[i]]$value * factor_to_bigger_better[i]
      if(score > best_score[i]){
        best_score[i] <- score
        best_iter[i] <- cur_iter
        if(verbose){
          best_msg[i] <- merge.eval.string(env)
        }
      } else {
        if(cur_iter - best_iter[i] >= stopping_rounds){
          if(!is.null(env$model)){
            env$model$best_iter <- best_iter[i]
          }
          if(verbose){
            print('Early stopping, best iteration is:')
            print(best_msg[i])
          }
          env$best_iter <- best_iter[i]
          env$met_early_stop <- TRUE
        }
      }
    }
  }
  attr(callback, 'name') <- 'cb.early.stop'
  attr(callback, 'order') <- 30
  return(callback)
}

