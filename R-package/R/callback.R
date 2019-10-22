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
    best_iter = -1,
    best_score = NA,
    met_early_stop = FALSE
  )
)

cb.reset.parameters <- function(new_params) {

  # Check for parameter list
  if (!is.list(new_params)) {
    stop(sQuote("new_params"), " must be a list")
  }

  # Deparse parameter list
  pnames  <- gsub("\\.", "_", names(new_params))
  nrounds <- NULL

  # Run some checks in the beginning
  init <- function(env) {

    # Store boosting rounds
    nrounds <<- env$end_iteration - env$begin_iteration + 1

    # Check for model environment
    if (is.null(env$model)) { stop("Env should have a ", sQuote("model")) }

    # Some parameters are not allowed to be changed,
    # since changing them would simply wreck some chaos
    not_allowed <- c("num_class", "metric", "boosting_type")
    if (any(pnames %in% not_allowed)) {
      stop("Parameters ", paste0(pnames[pnames %in% not_allowed], collapse = ", "), " cannot be changed during boosting")
    }

    # Check parameter names
    for (n in pnames) {

      # Set name
      p <- new_params[[n]]

      # Check if function for parameter
      if (is.function(p)) {

        # Check if requires at least two arguments
        if (length(formals(p)) != 2) {
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

  }

  callback <- function(env) {

    # Check if rounds is null
    if (is.null(nrounds)) {
      init(env)
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

    # To-do check pars
    if (!is.null(env$model)) {
      env$model$reset_parameter(pars)
    }

  }

  attr(callback, "call") <- match.call()
  attr(callback, "is_pre_iteration") <- TRUE
  attr(callback, "name") <- "cb.reset.parameters"
  callback
}

# Format the evaluation metric string
format.eval.string <- function(eval_res, eval_err = NULL) {

  # Check for empty evaluation string
  if (is.null(eval_res) || length(eval_res) == 0) {
    stop("no evaluation results")
  }

  # Check for empty evaluation error
  if (!is.null(eval_err)) {
    sprintf("%s\'s %s:%g+%g", eval_res$data_name, eval_res$name, eval_res$value, eval_err)
  } else {
    sprintf("%s\'s %s:%g", eval_res$data_name, eval_res$name, eval_res$value)
  }

}

merge.eval.string <- function(env) {

  # Check length of evaluation list
  if (length(env$eval_list) <= 0) {
    return("")
  }

  # Get evaluation
  msg <- list(sprintf("[%d]:", env$iteration))

  # Set if evaluation error
  is_eval_err <- length(env$eval_err_list) > 0

  # Loop through evaluation list
  for (j in seq_along(env$eval_list)) {

    # Store evaluation error
    eval_err <- NULL
    if (is_eval_err) {
      eval_err <- env$eval_err_list[[j]]
    }

    # Set error message
    msg <- c(msg, format.eval.string(env$eval_list[[j]], eval_err))

  }

  # Return tabulated separated message
  paste0(msg, collapse = "\t")

}

cb.print.evaluation <- function(period = 1) {

  # Create callback
  callback <- function(env) {

    # Check if period is at least 1 or more
    if (period > 0) {

      # Store iteration
      i <- env$iteration

      # Check if iteration matches moduo
      if ((i - 1) %% period == 0 || is.element(i, c(env$begin_iteration, env$end_iteration ))) {

        # Merge evaluation string
        msg <- merge.eval.string(env)

        # Check if message is existing
        if (nchar(msg) > 0) {
          cat(merge.eval.string(env), "\n")
        }

      }

    }

  }

  # Store attributes
  attr(callback, "call") <- match.call()
  attr(callback, "name") <- "cb.print.evaluation"

  # Return callback
  callback

}

cb.record.evaluation <- function() {

  # Create callback
  callback <- function(env) {

    # Return empty if empty evaluation list
    if (length(env$eval_list) <= 0) {
      return()
    }

    # Set if evaluation error
    is_eval_err <- length(env$eval_err_list) > 0

    # Check length of recorded evaluation
    if (length(env$model$record_evals) == 0) {

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
        env$model$record_evals[[data_name]][[name]]$eval <- list()
        env$model$record_evals[[data_name]][[name]]$eval_err <- list()

      }

    }

    # Loop through each evaluation list element
    for (j in seq_along(env$eval_list)) {

      # Get evaluation data
      eval_res <- env$eval_list[[j]]
      eval_err <- NULL
      if (is_eval_err) {
        eval_err <- env$eval_err_list[[j]]
      }

      # Store names
      data_name <- eval_res$data_name
      name <- eval_res$name

      # Store evaluation data
      env$model$record_evals[[data_name]][[name]]$eval <- c(env$model$record_evals[[data_name]][[name]]$eval, eval_res$value)
      env$model$record_evals[[data_name]][[name]]$eval_err <- c(env$model$record_evals[[data_name]][[name]]$eval_err, eval_err)

    }

  }

  # Store attributes
  attr(callback, "call") <- match.call()
  attr(callback, "name") <- "cb.record.evaluation"

  # Return callback
  callback

}

cb.early.stop <- function(stopping_rounds, verbose = TRUE) {

  # Initialize variables
  factor_to_bigger_better <- NULL
  best_iter <- NULL
  best_score <- NULL
  best_msg <- NULL
  eval_len <- NULL

  # Initialization function
  init <- function(env) {

    # Store evaluation length
    eval_len <<- length(env$eval_list)

    # Early stopping cannot work without metrics
    if (eval_len == 0) {
      stop("For early stopping, valids must have at least one element")
    }

    # Check if verbose or not
    if (isTRUE(verbose)) {
      cat("Will train until there is no improvement in ", stopping_rounds, " rounds.\n\n", sep = "")
    }

    # Maximization or minimization task
    factor_to_bigger_better <<- rep.int(1.0, eval_len)
    best_iter <<- rep.int(-1, eval_len)
    best_score <<- rep.int(-Inf, eval_len)
    best_msg <<- list()

    # Loop through evaluation elements
    for (i in seq_len(eval_len)) {

      # Prepend message
      best_msg <<- c(best_msg, "")

      # Check if maximization or minimization
      if (!env$eval_list[[i]]$higher_better) {
        factor_to_bigger_better[i] <<- -1.0
      }

    }

  }

  # Create callback
  callback <- function(env, finalize = FALSE) {

    # Check for empty evaluation
    if (is.null(eval_len)) {
      init(env)
    }

    # Store iteration
    cur_iter <- env$iteration

    # Loop through evaluation
    for (i in seq_len(eval_len)) {

      # Store score
      score <- env$eval_list[[i]]$value * factor_to_bigger_better[i]

        # Check if score is better
        if (score > best_score[i]) {

          # Store new scores
          best_score[i] <<- score
          best_iter[i] <<- cur_iter

          # Prepare to print if verbose
          if (verbose) {
            best_msg[[i]] <<- as.character(merge.eval.string(env))
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

              cat("Early stopping, best iteration is:", "\n")
              cat(best_msg[[i]], "\n")

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
          cat("Did not meet early stopping, best iteration is:", "\n")
          cat(best_msg[[i]], "\n")
        }

        # Store best iteration and stop
        env$best_iter <- best_iter[i]
        env$met_early_stop <- TRUE
      }
    }
  }

  # Set attributes
  attr(callback, "call") <- match.call()
  attr(callback, "name") <- "cb.early.stop"

  # Return callback
  callback

}

# Extract callback names from the list of callbacks
callback.names <- function(cb_list) { unlist(lapply(cb_list, attr, "name")) }

add.cb <- function(cb_list, cb) {

  # Combine two elements
  cb_list <- c(cb_list, cb)

  # Set names of elements
  names(cb_list) <- callback.names(cb_list)

  # Check for existence
  if ("cb.early.stop" %in% names(cb_list)) {

    # Concatenate existing elements
    cb_list <- c(cb_list, cb_list["cb.early.stop"])

    # Remove only the first one
    cb_list["cb.early.stop"] <- NULL

  }

  # Return element
  cb_list

}

categorize.callbacks <- function(cb_list) {

  # Check for pre-iteration or post-iteration
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

}
