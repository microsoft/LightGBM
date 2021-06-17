# constants that control naming in lists
.EVAL_KEY <- function() {
  return("eval")
}
.EVAL_ERR_KEY <- function() {
  return("eval_err")
}

#' @importFrom R6 R6Class
CB_ENV <- R6::R6Class(
  "lgb.cb_env",
  cloneable = FALSE,
  public = list(
    model = NULL,
    iteration = NULL,
    begin_iteration = NULL,
    end_iteration = NULL,
    eval_list = list(),
    eval_err_list = list(),
    best_iter = -1L,
    best_score = NA,
    met_early_stop = FALSE
  )
)

cb.reset.parameters <- function(new_params) {

  if (!identical(class(new_params), "list")) {
    stop(sQuote("new_params"), " must be a list")
  }

  # Deparse parameter list
  pnames  <- gsub("\\.", "_", names(new_params))
  nrounds <- NULL

  # Run some checks in the beginning
  init <- function(env) {

    # Check for model environment
    if (is.null(env$model)) {
      stop("Env should have a ", sQuote("model"))
    }

    # Store boosting rounds
    nrounds <<- env$end_iteration - env$begin_iteration + 1L

    # Check parameter names
    for (n in pnames) {

      # Set name
      p <- new_params[[n]]

      # Check if function for parameter
      if (is.function(p)) {

        # Check if requires at least two arguments
        if (length(formals(p)) != 2L) {
          stop("Parameter ", sQuote(n), " is a function but not of two arguments")
        }

        # Check if numeric or character
      } else if (is.numeric(p) || is.character(p)) {

        # Check if length is matching
        if (length(p) != nrounds) {
          stop("Length of ", sQuote(n), " has to be equal to length of ", sQuote("nrounds"))
        }

      } else {

        stop("Parameter ", sQuote(n), " is not a function or a vector")

      }

    }

    return(invisible(NULL))

  }

  callback <- function(env) {

    # Check if rounds is null
    if (is.null(nrounds)) {
      init(env = env)
    }

    # Store iteration
    i <- env$iteration - env$begin_iteration

    # Apply list on parameters
    pars <- lapply(new_params, function(p) {
      if (is.function(p)) {
        return(p(i, nrounds))
      }
      p[i]
    })

    if (!is.null(env$model)) {
      return(env$model$reset_parameter(params = pars))
    }

    return(invisible(NULL))

  }

  attr(callback, "call") <- match.call()
  attr(callback, "is_pre_iteration") <- TRUE
  attr(callback, "name") <- "cb.reset.parameters"
  return(callback)
}

# Format the evaluation metric string
format.eval.string <- function(eval_res, eval_err) {

  # Check for empty evaluation string
  if (is.null(eval_res) || length(eval_res) == 0L) {
    stop("no evaluation results")
  }

  # Check for empty evaluation error
  if (!is.null(eval_err)) {
    return(sprintf("%s\'s %s:%g+%g", eval_res$data_name, eval_res$name, eval_res$value, eval_err))
  } else {
    return(sprintf("%s\'s %s:%g", eval_res$data_name, eval_res$name, eval_res$value))
  }

}

merge.eval.string <- function(env) {

  # Check length of evaluation list
  if (length(env$eval_list) <= 0L) {
    return("")
  }

  # Get evaluation
  msg <- list(sprintf("[%d]:", env$iteration))

  # Set if evaluation error
  is_eval_err <- length(env$eval_err_list) > 0L

  # Loop through evaluation list
  for (j in seq_along(env$eval_list)) {

    # Store evaluation error
    eval_err <- NULL
    if (isTRUE(is_eval_err)) {
      eval_err <- env$eval_err_list[[j]]
    }

    # Set error message
    msg <- c(msg, format.eval.string(eval_res = env$eval_list[[j]], eval_err = eval_err))

  }

  return(paste0(msg, collapse = "  "))

}

cb.print.evaluation <- function(period) {

  # Create callback
  callback <- function(env) {

    # Check if period is at least 1 or more
    if (period > 0L) {

      # Store iteration
      i <- env$iteration

      # Check if iteration matches moduo
      if ((i - 1L) %% period == 0L || is.element(i, c(env$begin_iteration, env$end_iteration))) {

        # Merge evaluation string
        msg <- merge.eval.string(env = env)

        # Check if message is existing
        if (nchar(msg) > 0L) {
          print(merge.eval.string(env = env))
        }

      }

    }

    return(invisible(NULL))

  }

  # Store attributes
  attr(callback, "call") <- match.call()
  attr(callback, "name") <- "cb.print.evaluation"

  return(callback)

}

cb.record.evaluation <- function() {

  # Create callback
  callback <- function(env) {

    if (length(env$eval_list) <= 0L) {
      return()
    }

    # Set if evaluation error
    is_eval_err <- length(env$eval_err_list) > 0L

    # Check length of recorded evaluation
    if (length(env$model$record_evals) == 0L) {

      # Loop through each evaluation list element
      for (j in seq_along(env$eval_list)) {

        # Store names
        data_name <- env$eval_list[[j]]$data_name
        name <- env$eval_list[[j]]$name
        env$model$record_evals$start_iter <- env$begin_iteration

        # Check if evaluation record exists
        if (is.null(env$model$record_evals[[data_name]])) {
          env$model$record_evals[[data_name]] <- list()
        }

        # Create dummy lists
        env$model$record_evals[[data_name]][[name]] <- list()
        env$model$record_evals[[data_name]][[name]][[.EVAL_KEY()]] <- list()
        env$model$record_evals[[data_name]][[name]][[.EVAL_ERR_KEY()]] <- list()

      }

    }

    # Loop through each evaluation list element
    for (j in seq_along(env$eval_list)) {

      # Get evaluation data
      eval_res <- env$eval_list[[j]]
      eval_err <- NULL
      if (isTRUE(is_eval_err)) {
        eval_err <- env$eval_err_list[[j]]
      }

      # Store names
      data_name <- eval_res$data_name
      name <- eval_res$name

      # Store evaluation data
      env$model$record_evals[[data_name]][[name]][[.EVAL_KEY()]] <- c(
        env$model$record_evals[[data_name]][[name]][[.EVAL_KEY()]]
        , eval_res$value
      )
      env$model$record_evals[[data_name]][[name]][[.EVAL_ERR_KEY()]] <- c(
        env$model$record_evals[[data_name]][[name]][[.EVAL_ERR_KEY()]]
        , eval_err
      )

    }

    return(invisible(NULL))

  }

  # Store attributes
  attr(callback, "call") <- match.call()
  attr(callback, "name") <- "cb.record.evaluation"

  return(callback)

}

cb.early.stop <- function(stopping_rounds, first_metric_only, verbose) {

  factor_to_bigger_better <- NULL
  best_iter <- NULL
  best_score <- NULL
  best_msg <- NULL
  eval_len <- NULL

  # Initialization function
  init <- function(env) {

    # Early stopping cannot work without metrics
    if (length(env$eval_list) == 0L) {
      stop("For early stopping, valids must have at least one element")
    }

    # Store evaluation length
    eval_len <<- length(env$eval_list)

    # Check if verbose or not
    if (isTRUE(verbose)) {
      msg <- paste0(
        "Will train until there is no improvement in "
        , stopping_rounds
        , " rounds."
      )
      print(msg)
    }

    # Internally treat everything as a maximization task
    factor_to_bigger_better <<- rep.int(1.0, eval_len)
    best_iter <<- rep.int(-1L, eval_len)
    best_score <<- rep.int(-Inf, eval_len)
    best_msg <<- list()

    # Loop through evaluation elements
    for (i in seq_len(eval_len)) {

      # Prepend message
      best_msg <<- c(best_msg, "")

      # Internally treat everything as a maximization task
      if (!isTRUE(env$eval_list[[i]]$higher_better)) {
        factor_to_bigger_better[i] <<- -1.0
      }

    }

    return(invisible(NULL))

  }

  # Create callback
  callback <- function(env) {

    # Check for empty evaluation
    if (is.null(eval_len)) {
      init(env = env)
    }

    # Store iteration
    cur_iter <- env$iteration

    # By default, any metric can trigger early stopping. This can be disabled
    # with 'first_metric_only = TRUE'
    if (isTRUE(first_metric_only)) {
      evals_to_check <- 1L
    } else {
      evals_to_check <- seq_len(eval_len)
    }

    # Loop through evaluation
    for (i in evals_to_check) {

      # Store score
      score <- env$eval_list[[i]]$value * factor_to_bigger_better[i]

        # Check if score is better
        if (score > best_score[i]) {

          # Store new scores
          best_score[i] <<- score
          best_iter[i] <<- cur_iter

          # Prepare to print if verbose
          if (verbose) {
            best_msg[[i]] <<- as.character(merge.eval.string(env = env))
          }

        } else {

          # Check if early stopping is required
          if (cur_iter - best_iter[i] >= stopping_rounds) {

            # Check if model is not null
            if (!is.null(env$model)) {
              env$model$best_score <- best_score[i]
              env$model$best_iter <- best_iter[i]
            }

            # Print message if verbose
            if (isTRUE(verbose)) {

              print(paste0("Early stopping, best iteration is: ", best_msg[[i]]))

            }

            # Store best iteration and stop
            env$best_iter <- best_iter[i]
            env$met_early_stop <- TRUE
          }

        }

      if (!isTRUE(env$met_early_stop) && cur_iter == env$end_iteration) {
        # Check if model is not null
        if (!is.null(env$model)) {
          env$model$best_score <- best_score[i]
          env$model$best_iter <- best_iter[i]
        }

        # Print message if verbose
        if (isTRUE(verbose)) {
          print(paste0("Did not meet early stopping, best iteration is: ", best_msg[[i]]))
        }

        # Store best iteration and stop
        env$best_iter <- best_iter[i]
        env$met_early_stop <- TRUE
      }
    }

    return(invisible(NULL))

  }

  attr(callback, "call") <- match.call()
  attr(callback, "name") <- "cb.early.stop"

  return(callback)

}

# Extract callback names from the list of callbacks
callback.names <- function(cb_list) {
  return(unlist(lapply(cb_list, attr, "name")))
}

add.cb <- function(cb_list, cb) {

  # Combine two elements
  cb_list <- c(cb_list, cb)

  # Set names of elements
  names(cb_list) <- callback.names(cb_list = cb_list)

  # Check for existence
  if ("cb.early.stop" %in% names(cb_list)) {

    # Concatenate existing elements
    cb_list <- c(cb_list, cb_list["cb.early.stop"])

    # Remove only the first one
    cb_list["cb.early.stop"] <- NULL

  }

  # Return element
  return(cb_list)

}

categorize.callbacks <- function(cb_list) {

  # Check for pre-iteration or post-iteration
  return(
    list(
      pre_iter = Filter(function(x) {
        pre <- attr(x, "is_pre_iteration")
        !is.null(pre) && pre
      }, cb_list),
      post_iter = Filter(function(x) {
        pre <- attr(x, "is_pre_iteration")
        is.null(pre) || !pre
      }, cb_list)
    )
  )

}
