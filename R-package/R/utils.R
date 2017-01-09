lgb.is.Booster <- function(x) { lgb.check.r6.class(x, "lgb.Booster") }

lgb.is.Dataset <- function(x) { lgb.check.r6.class(x, "lgb.Dataset") }

# use 64bit data to store address
lgb.new.handle <- function() { 0.0 }

lgb.is.null.handle <- function(x) { is.null(x) || x == 0 }

lgb.encode.char <- function(arr, len) {
  if (!is.raw(arr)) {
    stop("lgb.encode.char: Can only encode from raw type")
  }
  rawToChar(arr[seq_len(len)])
}

lgb.call <- function(fun_name, ret, ...) {
  call_state <- 0L
  if (!is.null(ret)) {
    call_state <- .Call(fun_name, ..., ret, call_state, PACKAGE = "lightgbm")
  } else {
    call_state <- .Call(fun_name, ..., call_state, PACKAGE = "lightgbm")
  }
  if (call_state != 0L) {
    buf_len <- 200L
    act_len <- 0L
    err_msg <- raw(buf_len)
    err_msg <- .Call("LGBM_GetLastError_R", buf_len, act_len, err_msg, PACKAGE = "lightgbm")
    if (act_len > buf_len) {
      buf_len <- act_len
      err_msg <- raw(buf_len)
      err_msg <- .Call("LGBM_GetLastError_R",
                        buf_len,
                        act_len,
                        err_msg,
                        PACKAGE = "lightgbm")
    }
    stop(paste0("api error: ", lgb.encode.char(err_msg, act_len)))
  }
  ret
}


lgb.call.return.str <- function(fun_name, ...) {
  buf_len <- as.integer(1024 * 1024)
  act_len <- 0L
  buf <- raw(buf_len)
  buf <- lgb.call(fun_name, ret = buf, ..., buf_len, act_len)
  if (act_len > buf_len) {
    buf_len <- act_len
    buf     <- raw(buf_len)
    buf     <- lgb.call(fun_name, ret = buf, ..., buf_len, act_len)
  }
  lgb.encode.char(buf, act_len)
}

lgb.params2str <- function(params, ...) {
  if (!is.list(params)) { stop("params must be a list") }
  names(params) <- gsub("\\.", "_", names(params))
  # merge parameters from the params and the dots-expansion
  dot_params <- list(...)
  names(dot_params) <- gsub("\\.", "_", names(dot_params))
  if (length(intersect(names(params),
                       names(dot_params))) > 0)
    stop(
      "Same parameters in ", sQuote("params"), " and in the call are not allowed. Please check your ", sQuote("params"), " list"
    )
  params <- c(params, dot_params)
  ret    <- list()
  for (key in names(params)) {
    # join multi value first
    val <- paste0(params[[key]], collapse = ",")
    if (nchar(val) <= 0) next
    # join key value
    pair <- paste0(c(key, val), collapse = "=")
    ret  <- c(ret, pair)
  }
  if (length(ret) == 0) {
    lgb.c_str("")
  } else {
    lgb.c_str(paste0(ret, collapse = " "))
  }
}

lgb.c_str <- function(x) {
  ret <- charToRaw(as.character(x))
  ret <- c(ret, as.raw(0))
  ret
}

lgb.check.r6.class <- function(object, name) {
  if (!("R6" %in% class(object))) {
    return(FALSE)
  }
  if (!(name %in% class(object))) {
    return(FALSE)
  }
  TRUE
}

lgb.check.params <- function(params) {
  # To-do
  params
}

lgb.check.obj <- function(params, obj) {
  OBJECTIVES <- c("regression", "binary", "multiclass", "lambdarank")
  if (!is.null(obj)) { params$objective <- obj }
  if (is.character(params$objective)) {
    if (!(params$objective %in% OBJECTIVES)) {
      stop("lgb.check.obj: objective name error should be one of (", paste0(OBJECTIVES, collapse = ", "), ")")
    }
  } else if (!is.function(objective)) {
    stop("lgb.check.obj: objective should be a character or a function")
  }
  params
}

lgb.check.eval <- function(params, eval) {
  if (is.null(params$metric)) { params$metric <- list() }
  if (!is.null(eval)) {
    # append metric
    if (is.character(eval) || is.list(eval)) {
      params$metric <- append(params$metric, eval)
    }
  }
  if (!is.function(eval)) {
    if (length(params$metric) == 0) {
      # add default metric
      params$metric <- switch(
        params$objective,
        regression = "l2",
        binary     = "binary_logloss",
        multiclass = "multi_logloss",
        lambdarank = "ndcg",
        stop("lgb.check.eval: No default metric available for objective ", sQuote(params$objective))
      )
    }
  }
  params
}
