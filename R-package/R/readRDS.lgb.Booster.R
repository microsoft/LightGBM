#' readRDS for lgb.Booster models
#'
#' Attemps to load a model using RDS.
#' 
#' @param object R object to serialize.
#' @param file a connection or the name of the file where the R object is saved to or read from.
#' @param ascii a logical. If TRUE or NA, an ASCII representation is written; otherwise (default), a binary one is used. See the comments in the help for save.
#' @param version the workspace format version to use. \code{NULL} specifies the current default version (2). Versions prior to 2 are not supported, so this will only be relevant when there are later versions.
#' @param compress a logical specifying whether saving to a named file is to use "gzip" compression, or one of \code{"gzip"}, \code{"bzip2"} or \code{"xz"} to indicate the type of compression to be used. Ignored if file is a connection.
#' @param refhook a hook function for handling reference objects.
#' @param raw whether to save the model in a raw variable or not, recommended to leave it to \code{TRUE}.
#' 
#' @return an R object.
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
#'   saveRDS(model, "model.rds")
#' }
#' @export

readRDS.lgb.Booster <- function(file = "", refhook = NULL) {
  
  object <- readRDS(file = file, refhook = refhook)
  if (!is.na(object$raw)) {
    temp <- tempfile()
    write(object$raw, temp)
    object2 <- lgb.load(temp)
    file.remove(temp)
    object2$best_iter <- object$best_iter
    object2$record_evals <- object$record_evals
    return(object2)
  } else {
    return(object)
  }
  
}
