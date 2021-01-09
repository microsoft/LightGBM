lgb.is.Booster <- function(x) {
  return(lgb.check.r6.class(object = x, name = "lgb.Booster"))
}

lgb.is.Dataset <- function(x) {
  return(lgb.check.r6.class(object = x, name = "lgb.Dataset"))
}

lgb.null.handle <- function() {
  if (.Machine$sizeof.pointer == 8L) {
    return(NA_real_)
  } else {
    return(NA_integer_)
  }
}

lgb.is.null.handle <- function(x) {
  return(is.null(x) || is.na(x))
}

lgb.encode.char <- function(arr, len) {
  if (!is.raw(arr)) {
    stop("lgb.encode.char: Can only encode from raw type")
  }
  return(rawToChar(arr[seq_len(len)]))
}

# [description] Raise an error. Before raising that error, check for any error message
#               stored in a buffer on the C++ side.
lgb.last_error <- function() {
  # Perform text error buffering
  buf_len <- 200L
  act_len <- 0L
  err_msg <- raw(buf_len)
  err_msg <- .Call(
    "LGBM_GetLastError_R"
    , buf_len
    , act_len
    , err_msg
    , PACKAGE = "lib_lightgbm"
  )

  # Check error buffer
  if (act_len > buf_len) {
    buf_len <- act_len
    err_msg <- raw(buf_len)
    err_msg <- .Call(
      "LGBM_GetLastError_R"
      , buf_len
      , act_len
      , err_msg
      , PACKAGE = "lib_lightgbm"
    )
  }

  stop("api error: ", lgb.encode.char(arr = err_msg, len = act_len))

  return(invisible(NULL))

}

lgb.call <- function(fun_name, ret, ...) {
  # Set call state to a zero value
  call_state <- 0L

  # Check for a ret call
  if (!is.null(ret)) {
    call_state <- .Call(
      fun_name
      , ...
      , ret
      , call_state
      , PACKAGE = "lib_lightgbm"
    )
  } else {
    call_state <- .Call(
      fun_name
      , ...
      , call_state
      , PACKAGE = "lib_lightgbm"
    )
  }
  call_state <- as.integer(call_state)
  # Check for call state value post call
  if (call_state != 0L) {
    lgb.last_error()
  }

  return(ret)

}

lgb.call.return.str <- function(fun_name, ...) {

  # Create buffer
  buf_len <- as.integer(1024L * 1024L)
  act_len <- 0L
  buf <- raw(buf_len)

  # Call buffer
  buf <- lgb.call(fun_name = fun_name, ret = buf, ..., buf_len, act_len)

  # Check for buffer content
  if (act_len > buf_len) {
    buf_len <- act_len
    buf <- raw(buf_len)
    buf <- lgb.call(fun_name = fun_name, ret = buf, ..., buf_len, act_len)
  }

  return(lgb.encode.char(arr = buf, len = act_len))

}

lgb.params2str <- function(params, ...) {

  # Check for a list as input
  if (!identical(class(params), "list")) {
    stop("params must be a list")
  }

  # Split parameter names
  names(params) <- gsub("\\.", "_", names(params))

  # Merge parameters from the params and the dots-expansion
  dot_params <- list(...)
  names(dot_params) <- gsub("\\.", "_", names(dot_params))

  # Check for identical parameters
  if (length(intersect(names(params), names(dot_params))) > 0L) {
    stop(
      "Same parameters in "
      , sQuote("params")
      , " and in the call are not allowed. Please check your "
      , sQuote("params")
      , " list"
    )
  }

  # Merge parameters
  params <- c(params, dot_params)

  # Setup temporary variable
  ret <- list()

  # Perform key value join
  for (key in names(params)) {

    # If a parameter has multiple values, join those values together with commas.
    # trimws() is necessary because format() will pad to make strings the same width
    val <- paste0(
      trimws(
        format(
          x = params[[key]]
          , scientific = FALSE
        )
      )
      , collapse = ","
    )
    if (nchar(val) <= 0L) next # Skip join

    # Join key value
    pair <- paste0(c(key, val), collapse = "=")
    ret <- c(ret, pair)

  }

  # Check ret length
  if (length(ret) == 0L) {
    return(lgb.c_str(x = ""))
  }

  return(lgb.c_str(x = paste0(ret, collapse = " ")))

}

lgb.check_interaction_constraints <- function(params, column_names) {

  # Convert interaction constraints to feature numbers
  string_constraints <- list()

  if (!is.null(params[["interaction_constraints"]])) {

    if (!methods::is(params[["interaction_constraints"]], "list")) {
        stop("interaction_constraints must be a list")
    }
    if (!all(sapply(params[["interaction_constraints"]], function(x) {is.character(x) || is.numeric(x)}))) {
        stop("every element in interaction_constraints must be a character vector or numeric vector")
    }

    for (constraint in params[["interaction_constraints"]]) {

      # Check for character name
      if (is.character(constraint)) {

          constraint_indices <- as.integer(match(constraint, column_names) - 1L)

          # Provided indices, but some indices are not existing?
          if (sum(is.na(constraint_indices)) > 0L) {
            stop(
              "supplied an unknown feature in interaction_constraints "
              , sQuote(constraint[is.na(constraint_indices)])
            )
          }

        } else {

          # Check that constraint indices are at most number of features
          if (max(constraint) > length(column_names)) {
            stop(
              "supplied a too large value in interaction_constraints: "
              , max(constraint)
              , " but only "
              , length(column_names)
              , " features"
            )
          }

          # Store indices as [0, n-1] indexed instead of [1, n] indexed
          constraint_indices <- as.integer(constraint - 1L)

        }

        # Convert constraint to string
        constraint_string <- paste0("[", paste0(constraint_indices, collapse = ","), "]")
        string_constraints <- append(string_constraints, constraint_string)
    }

  }

  return(string_constraints)

}

lgb.c_str <- function(x) {

  ret <- charToRaw(as.character(x))
  ret <- c(ret, as.raw(0L))
  return(ret)

}

lgb.check.r6.class <- function(object, name) {

  # Check for non-existence of R6 class or named class
  return(all(c("R6", name) %in% class(object)))

}

lgb.check.obj <- function(params, obj) {

  # List known objectives in a vector
  OBJECTIVES <- c(
    "regression"
    , "regression_l1"
    , "regression_l2"
    , "mean_squared_error"
    , "mse"
    , "l2_root"
    , "root_mean_squared_error"
    , "rmse"
    , "mean_absolute_error"
    , "mae"
    , "quantile"
    , "huber"
    , "fair"
    , "poisson"
    , "binary"
    , "lambdarank"
    , "multiclass"
    , "softmax"
    , "multiclassova"
    , "multiclass_ova"
    , "ova"
    , "ovr"
    , "xentropy"
    , "cross_entropy"
    , "xentlambda"
    , "cross_entropy_lambda"
    , "mean_absolute_percentage_error"
    , "mape"
    , "gamma"
    , "tweedie"
    , "rank_xendcg"
    , "xendcg"
    , "xe_ndcg"
    , "xe_ndcg_mart"
    , "xendcg_mart"
  )

  # Check whether the objective is empty or not, and take it from params if needed
  if (!is.null(obj)) {
    params$objective <- obj
  }

  # Check whether the objective is a character
  if (is.character(params$objective)) {

    # If the objective is a character, check if it is a known objective
    if (!(params$objective %in% OBJECTIVES)) {

      stop("lgb.check.obj: objective name error should be one of (", paste0(OBJECTIVES, collapse = ", "), ")")

    }

  } else if (!is.function(params$objective)) {

    stop("lgb.check.obj: objective should be a character or a function")

  }

  return(params)

}

# [description]
#     Take any character values from eval and store them in params$metric.
#     This has to account for the fact that `eval` could be a character vector,
#     a function, a list of functions, or a list with a mix of strings and
#     functions
lgb.check.eval <- function(params, eval) {

  if (is.null(params$metric)) {
    params$metric <- list()
  } else if (is.character(params$metric)) {
    params$metric <- as.list(params$metric)
  }

  # if 'eval' is a character vector or list, find the character
  # elements and add them to 'metric'
  if (!is.function(eval)) {
    for (i in seq_along(eval)) {
      element <- eval[[i]]
      if (is.character(element)) {
        params$metric <- append(params$metric, element)
      }
    }
  }

  # If more than one character metric was given, then "None" should
  # not be included
  if (length(params$metric) > 1L) {
    params$metric <- Filter(
        f = function(metric) {
          !(metric %in% .NO_METRIC_STRINGS())
        }
        , x = params$metric
    )
  }

  # duplicate metrics should be filtered out
  params$metric <- as.list(unique(unlist(params$metric)))

  return(params)
}


# [description]
#
#     Resolve differences between passed-in keyword arguments, parameters,
#     and parameter aliases. This function exists because some functions in the
#     package take in parameters through their own keyword arguments other than
#     the `params` list.
#
#     If the same underlying parameter is provided multiple
#     ways, the first item in this list is used:
#
#         1. the main (non-alias) parameter found in `params`
#         2. the first alias of that parameter found in `params`
#         3. the keyword argument passed in
#
#     For example, "num_iterations" can also be provided to lgb.train()
#     via keyword "nrounds". lgb.train() will choose one value for this parameter
#     based on the first match in this list:
#
#         1. params[["num_iterations]]
#         2. the first alias of "num_iterations" found in params
#         3. the nrounds keyword argument
#
#     If multiple aliases are found in `params` for the same parameter, they are
#     all removed before returning `params`.
#
# [return]
#     params with num_iterations set to the chosen value, and other aliases
#     of num_iterations removed
lgb.check.wrapper_param <- function(main_param_name, params, alternative_kwarg_value) {

  aliases <- .PARAMETER_ALIASES()[[main_param_name]]
  aliases_provided <- names(params)[names(params) %in% aliases]
  aliases_provided <- aliases_provided[aliases_provided != main_param_name]

  # prefer the main parameter
  if (!is.null(params[[main_param_name]])) {
    for (param in aliases_provided) {
      params[[param]] <- NULL
    }
    return(params)
  }

  # if the main parameter wasn't proovided, prefer the first alias
  if (length(aliases_provided) > 0L) {
    first_param <- aliases_provided[1L]
    params[[main_param_name]] <- params[[first_param]]
    for (param in aliases_provided) {
      params[[param]] <- NULL
    }
    return(params)
  }

  # if not provided in params at all, use the alternative value provided
  # through a keyword argument from lgb.train(), lgb.cv(), etc.
  params[[main_param_name]] <- alternative_kwarg_value
  return(params)
}
