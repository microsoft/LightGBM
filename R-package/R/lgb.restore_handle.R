#' @name lgb.restore_handle
#' @title Restore the C++ component of a de-serialized LightGBM model
#' @description After a LightGBM model object is de-serialized through functions such as \code{save} or
#' \code{saveRDS}, its underlying C++ object will be blank and needs to be restored to able to use it. Such
#' object is restored automatically when calling functions such as \code{predict}, but this function can be
#' used to forcibly restore it beforehand. Note that the object will be modified in-place.
#' @param model \code{lgb.Booster} object which was de-serialized and whose underlying C++ object and R handle
#' need to be restored.
#'
#' @return \code{lgb.Booster} (the same `model` object that was passed as input, invisibly).
#' @seealso \link{lgb.make_serializable}, \link{lgb.drop_serialized}.
#' @examples
#' library(lightgbm)
#' data("agaricus.train")
#' model <- lightgbm(
#'   agaricus.train$data
#'   , agaricus.train$label
#'   , params = list(objective = "binary")
#'   , nrounds = 5L
#'   , verbose = 0)
#' fname <- tempfile(fileext="rds")
#' saveRDS(model, fname)
#'
#' model_new <- readRDS(fname)
#' model_new$check_null_handle()
#' lgb.restore_handle(model_new)
#' model_new$check_null_handle()
#' @export
lgb.restore_handle <- function(model) {
  if (!lgb.is.Booster(x = model)) {
    stop("lgb.restore_handle: model should be an ", sQuote("lgb.Booster"))
  }
  model$restore_handle()
  return(invisible(model))
}
