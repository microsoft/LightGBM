#' readRDS for lgb.Booster models
#'
#' Attemps to load a model using RDS.
#' 
#' @param file a connection or the name of the file where the R object is saved to or read from.
#' @param refhook a hook function for handling reference objects.
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
#'   saveRDS.lgb.Booster(model, "model.rds")
#'   new_model <- readRDS.lgb.Booster("model.rds")
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
