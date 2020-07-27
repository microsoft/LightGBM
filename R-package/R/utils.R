lgb.is.Booster <- function(x) {
  lgb.check.r6.class(x, "lgb.Booster") # Checking if it is of class lgb.Booster or not
}

lgb.is.Dataset <- function(x) {
  lgb.check.r6.class(x, "lgb.Dataset") # Checking if it is of class lgb.Dataset or not
}

lgb.is.null.handle <- function(x) {
  is.null(x) || is.na(x)
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

  # Return error
  stop("api error: ", lgb.encode.char(err_msg, act_len))
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
  buf <- lgb.call(fun_name, ret = buf, ..., buf_len, act_len)

  # Check for buffer content
  if (act_len > buf_len) {
    buf_len <- act_len
    buf <- raw(buf_len)
    buf <- lgb.call(fun_name, ret = buf, ..., buf_len, act_len)
  }

  # Return encoded character
  return(lgb.encode.char(buf, act_len))

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
    return(lgb.c_str(""))
  }

  # Return string separated by a space per element
  return(lgb.c_str(paste0(ret, collapse = " ")))

}

lgb.check_interaction_constraints <- function(params, column_names) {

  # Convert interaction constraints to feature numbers
  string_constraints <- list()

  if (!is.null(params[["interaction_constraints"]])) {

    # validation
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

  # Perform character to raw conversion
  ret <- charToRaw(as.character(x))
  ret <- c(ret, as.raw(0L))
  ret

}

lgb.check.r6.class <- function(object, name) {

  # Check for non-existence of R6 class or named class
  all(c("R6", name) %in% class(object))

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

      # Interrupt on unknown objective name
      stop("lgb.check.obj: objective name error should be one of (", paste0(OBJECTIVES, collapse = ", "), ")")

    }

  } else if (!is.function(params$objective)) {

    # If objective is not a character nor a function, then stop
    stop("lgb.check.obj: objective should be a character or a function")

  }

  # Return parameters
  return(params)

}

# [description]
#     make sure that "metric" is populated on params,
#     and add any eval values to it
# [return]
#     params, where "metric" is a list
lgb.check.eval <- function(params, eval) {

  if (is.null(params$metric)) {
    params$metric <- list()
  } else if (is.character(params$metric)) {
    params$metric <- as.list(params$metric)
  }

  if (is.character(eval)) {
    params$metric <- append(params$metric, eval)
  }

  if (identical(class(eval), "list")) {
    params$metric <- append(params$metric, unlist(eval))
  }

  return(params)
}
