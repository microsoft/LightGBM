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
    stop("lgb.encode.char: Can only encode from raw type") # Not an object of type raw
  }
  rawToChar(arr[seq_len(len)]) # Return the conversion of raw type to character type

}

lgb.call <- function(fun_name, ret, ...) {
  # Set call state to a zero value
  call_state <- 0L

  # Check for a ret call
  if (!is.null(ret)) {
    call_state <- .Call(fun_name, ..., ret, call_state, PACKAGE = "lib_lightgbm") # Call with ret
  } else {
    call_state <- .Call(fun_name, ..., call_state, PACKAGE = "lib_lightgbm") # Call without ret
  }
  call_state <- as.integer(call_state)
  # Check for call state value post call
  if (call_state != 0L) {

    # Perform text error buffering
    buf_len <- 200L
    act_len <- 0L
    err_msg <- raw(buf_len)
    err_msg <- .Call("LGBM_GetLastError_R", buf_len, act_len, err_msg, PACKAGE = "lib_lightgbm")

    # Check error buffer
    if (act_len > buf_len) {
      buf_len <- act_len
      err_msg <- raw(buf_len)
      err_msg <- .Call("LGBM_GetLastError_R",
                        buf_len,
                        act_len,
                        err_msg,
                        PACKAGE = "lib_lightgbm")
    }

    # Return error
    stop("api error: ", lgb.encode.char(err_msg, act_len))

  }

  return(ret)

}

lgb.call.return.str <- function(fun_name, ...) {

  # Create buffer
  buf_len <- as.integer(1024 * 1024)
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
  if (!is.list(params)) {
    stop("params must be a list")
  }

  # Split parameter names
  names(params) <- gsub("\\.", "_", names(params))

  # Merge parameters from the params and the dots-expansion
  dot_params <- list(...)
  names(dot_params) <- gsub("\\.", "_", names(dot_params))

  # Check for identical parameters
  if (length(intersect(names(params), names(dot_params))) > 0) {
    stop("Same parameters in ", sQuote("params"), " and in the call are not allowed. Please check your ", sQuote("params"), " list")
  }

  # Merge parameters
  params <- c(params, dot_params)

  # Setup temporary variable
  ret <- list()

  # Perform key value join
  for (key in names(params)) {

    # Join multi value first
    val <- paste0(format(params[[key]], scientific = FALSE), collapse = ",")
    if (nchar(val) <= 0) next # Skip join

    # Join key value
    pair <- paste0(c(key, val), collapse = "=")
    ret <- c(ret, pair)

  }

  # Check ret length
  if (length(ret) == 0) {

    # Return empty string
    lgb.c_str("")

  } else {

    # Return string separated by a space per element
    lgb.c_str(paste0(ret, collapse = " "))

  }

}

lgb.c_str <- function(x) {

  # Perform character to raw conversion
  ret <- charToRaw(as.character(x))
  ret <- c(ret, as.raw(0))
  ret

}

lgb.check.r6.class <- function(object, name) {

  # Check for non-existence of R6 class or named class
  all(c("R6", name) %in% class(object))

}

lgb.check.params <- function(params) {

  # To-do
  params # Currently return params because this is not finalized

}

lgb.check.obj <- function(params, obj) {

  # List known objectives in a vector
  OBJECTIVES <- c("regression", "regression_l1", "regression_l2", "mean_squared_error", "mse", "l2_root", "root_mean_squared_error", "rmse",
                  "mean_absolute_error", "mae", "quantile",
                  "huber", "fair", "poisson", "binary", "lambdarank",
                  "multiclass", "softmax", "multiclassova", "multiclass_ova", "ova", "ovr",
                  "xentropy", "cross_entropy", "xentlambda", "cross_entropy_lambda", "mean_absolute_percentage_error", "mape",
                  "gamma", "tweedie")

  # Check whether the objective is empty or not, and take it from params if needed
  if (!is.null(obj)) { params$objective <- obj }

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

lgb.check.eval <- function(params, eval) {

  # Check if metric is null, if yes put a list instead
  if (is.null(params$metric)) {
    params$metric <- list()
  }

  # Check if evaluation metric is null, if not then append it
  if (!is.null(eval)) {

    # Append metric if character or list
    if (is.character(eval) || is.list(eval)) {

      # Append metrics
      params$metric <- append(params$metric, eval)

    }

  }
  # Return parameters
  return(params)
}
