#' @name readRDS.lgb.Booster
#' @title readRDS for \code{lgb.Booster} models (DEPRECATED)
#' @description Calls \code{readRDS} in what is expected to be a serialized \code{lgb.Booster} object,
#'              and then restores its handle through \code{lgb.restore_handle}.
#'
#'              \bold{This function throws a warning and will be removed in future versions.}
#' @param file a connection or the name of the file where the R object is saved to or read from.
#' @param refhook a hook function for handling reference objects.
#'
#' @return \code{lgb.Booster}
#'
#' @examples
#' \donttest{
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(
#'   objective = "regression"
#'   , metric = "l2"
#'   , min_data = 1L
#'   , learning_rate = 1.0
#' )
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 10L
#'   , valids = valids
#'   , early_stopping_rounds = 5L
#' )
#' model_file <- tempfile(fileext = ".rds")
#' saveRDS.lgb.Booster(model, model_file)
#' new_model <- readRDS.lgb.Booster(model_file)
#' }
#' @export
readRDS.lgb.Booster <- function(file, refhook = NULL) {

  warning("'readRDS.lgb.Booster' is deprecated and will be removed in a future release. Use readRDS() instead.")

  object <- readRDS(file = file, refhook = refhook)
  lgb.restore_handle(object)
  return(object)
}
