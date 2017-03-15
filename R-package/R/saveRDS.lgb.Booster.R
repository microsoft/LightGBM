#' saveRDS for lgb.Booster models
#'
#' Attemps to save a model using RDS.
#' 
#' @param object R object to serialize.
#' @param file a connection or the name of the file where the R object is saved to or read from.
#' @param ascii a logical. If TRUE or NA, an ASCII representation is written; otherwise (default), a binary one is used. See the comments in the help for save.
#' @param version the workspace format version to use. NULL specifies the current default version (2). Versions prior to 2 are not supported, so this will only be relevant when there are later versions.
#' @param compress a logical specifying whether saving to a named file is to use "gzip" compression, or one of "gzip", "bzip2" or "xz" to indicate the type of compression to be used. Ignored if file is a connection.
#' @param refhook a hook function for handling reference objects.
#' 
#' @return NULL invisibly.
#' 
#' @examples
#' \dontrun{
#'   library(lightgbm)
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'   data(agaricus.test, package='lightgbm')
#'   test <- agaricus.test
#'   dtest <- lgb.Dataset.create.valid(dtrain, test$data, label=test$label)
#'   params <- list(objective="regression", metric="l2")
#'   valids <- list(test=dtest)
#'   model <- lgb.train(params, dtrain, 100, valids, min_data=1, learning_rate=1, early_stopping_rounds=10)
#' }
#' @export

saveRDS.lgb.Booster <- function(object, file = "", ascii = FALSE, version = NULL, compress = TRUE, refhook = NULL) {
  
  if (class(object$raw) == "function") {
    object$raw()
  }
  saveRDS(object, file = "", ascii = FALSE, version = NULL, compress = TRUE, refhook = NULL)
  
}