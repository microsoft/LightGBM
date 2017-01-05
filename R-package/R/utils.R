lgb.new.handle <- function() {
  # use 64bit data to store address
  rep(0.0, 1)
}
lgb.is.null.handle <- function(x) {
  if(is.null(x) | x == 0.0){
    return(TRUE)
  }
  return(FALSE)
}

lgb.encode.char <- function(arr, len) {
  if(typeof(arr) != "raw"){
    stop("lgb.encode.char: only can encode from raw type")
  }
  return(rawToChar(arr[1:len]))
}

lgb.call <- function(fun_name, ret, ...){
  call_state <- 0
  if(!is.null(ret)){
    call_state <- .Call(fun_name, ..., ret, call_state , PACKAGE="lightgbm")
  } else {
    call_state <- .Call(fun_name, ..., call_state , PACKAGE="lightgbm")
  }
  if(call_state != 0){
    buf_len <- 200
    act_len <- 0
    err_msg <- raw(buf_len)
    err_msg <- .Call("LGBM_GetLastError_R", buf_len, act_len, err_msg, PACKAGE="lightgbm")
    if(act_len > buf_len) {
      buf_len <- act_len
      act_len <- 0
      err_msg <- raw(buf_len)
      err_msg <- .Call("LGBM_GetLastError_R", buf_len, act_len, err_msg, PACKAGE="lightgbm")
    }
    stop(lgb.encode.char(err_msg, act_len))
  }
  return(ret)
}

lgb.call.with.str <- function(fun_name, ...) {
  buf_len <- 1024*1024
  act_len <- 0
  buf <- raw(buf_len)
  buf <- lgb.call(fun_name, ret=buf, ..., buf_len, act_len)
  if(act_len > buf_len) {
    buf_len <- act_len
    act_len <- 0
    buf <- raw(buf_len)
    buf <- lgb.call(fun_name, ret=buf, ..., buf_len, act_len)
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
    stop("Same parameters in 'params' and in the call are not allowed. Please check your 'params' list.")
  params <- c(params, dot_params)
  ret <- list()
  ret <- c(ret, "")
  for( key in names(params) ) {
    # join multi value first
    val <- paste0(params[[key]], collapse=",")
    # join key value
    pair <- paste0(c(key, val), collapse="=")
    ret <- c(ret, pair)
  }
  
  return(paste0(ret, collapse=" "))
}

lgb.check.r6.class <- function(object, name) {
  if(!("R6" %in% class(object))){
    return(FALSE)
  }
  if(!(name %in% class(object))){
    return(FALSE)
  }
  return(TRUE)
}
