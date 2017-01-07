lgb.new.handle <- function() {
  # use 64bit data to store address
  return(0.0)
}
lgb.is.null.handle <- function(x) {
  if (is.null(x)) {
    return(TRUE)
  }
  if (x == 0) {
    return(TRUE)
  }
  return(FALSE)
}

lgb.encode.char <- function(arr, len) {
  if (typeof(arr) != "raw") {
    stop("lgb.encode.char: only can encode from raw type")
  }
  return(rawToChar(arr[1:len]))
}

lgb.call <- function(fun_name, ret, ...) {
  call_state <- as.integer(0)
  if (!is.null(ret)) {
    call_state <-
      .Call(fun_name, ..., ret, call_state , PACKAGE = "lightgbm")
  } else {
    call_state <- .Call(fun_name, ..., call_state , PACKAGE = "lightgbm")
  }
  if (call_state != as.integer(0)) {
    buf_len <- as.integer(200)
    act_len <- as.integer(0)
    err_msg <- raw(buf_len)
    err_msg <-
      .Call("LGBM_GetLastError_R", buf_len, act_len, err_msg, PACKAGE = "lightgbm")
    if (act_len > buf_len) {
      buf_len <- act_len
      err_msg <- raw(buf_len)
      err_msg <-
        .Call("LGBM_GetLastError_R",
              buf_len,
              act_len,
              err_msg,
              PACKAGE = "lightgbm")
    }
    stop(paste0("api error: ", lgb.encode.char(err_msg, act_len)))
  }
  return(ret)
}


lgb.call.return.str <- function(fun_name, ...) {
  buf_len <- as.integer(1024 * 1024)
  act_len <- as.integer(0)
  buf <- raw(buf_len)
  buf <- lgb.call(fun_name, ret = buf, ..., buf_len, act_len)
  if (act_len > buf_len) {
    buf_len <- act_len
    buf <- raw(buf_len)
    buf <- lgb.call(fun_name, ret = buf, ..., buf_len, act_len)
  }
  return(lgb.encode.char(buf, act_len))
}

lgb.params2str <- function(params, ...) {
  if (typeof(params) != "list")
    stop("params must be a list")
  names(params) <- gsub("\\.", "_", names(params))
  # merge parameters from the params and the dots-expansion
  dot_params <- list(...)
  names(dot_params) <- gsub("\\.", "_", names(dot_params))
  if (length(intersect(names(params),
                       names(dot_params))) > 0)
    stop(
      "Same parameters in 'params' and in the call are not allowed. Please check your 'params' list."
    )
  params <- c(params, dot_params)
  ret <- list()
  for (key in names(params)) {
    # join multi value first
    val <- paste0(params[[key]], collapse = ",")
    if(nchar(val) <= 0) next
    # join key value
    pair <- paste0(c(key, val), collapse = "=")
    ret <- c(ret, pair)
  }
  if (length(ret) == 0) {
    return(lgb.c_str(""))
  } else{
    return(lgb.c_str(paste0(ret, collapse = " ")))
  }
}

lgb.c_str <- function(x) {
  ret <- charToRaw(as.character(x))
  ret <- c(ret, as.raw(0))
  return(ret)
}

lgb.check.r6.class <- function(object, name) {
  if (!("R6" %in% class(object))) {
    return(FALSE)
  }
  if (!(name %in% class(object))) {
    return(FALSE)
  }
  return(TRUE)
}

lgb.check.params <- function(params){
  # To-do
  return(params)
}

lgb.check.obj <- function(params, obj) {
  if(!is.null(obj)){
    params$objective <- obj
  }
  if(is.character(params$objective)){ 
    if(!(params$objective %in% c("regression", "binary", "multiclass", "lambdarank"))){
      stop("lgb.check.obj: objective name error should be (regression, binary, multiclass, lambdarank)")
    }
  } else if(typeof(params$objective) != "closure"){
    stop("lgb.check.obj: objective should be character or function")
  }
  return(params)
}

lgb.check.eval <- function(params, eval) {
  if(is.null(params$metric)){
    params$metric <- list()
  }
  if(!is.null(eval)){
    # append metric
    if(is.character(eval) || is.list(eval)){
      params$metric <- append(params$metric, eval)
    }
  }
  if (typeof(eval) != "closure"){
    if(is.null(params$metric) | length(params$metric) == 0) {
      # add default metric
      if(is.character(params$objective)){
        if(params$objective == "regression"){
          params$metric <- "l2"
        } else if(params$objective == "binary"){
          params$metric <- "binary_logloss"
        } else if(params$objective == "multiclass"){
          params$metric <- "multi_logloss"
        } else if(params$objective == "lambdarank"){
          params$metric <- "ndcg"
        }
      }
    }
  }
  return(params)
}

