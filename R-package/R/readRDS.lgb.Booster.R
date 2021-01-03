#' @name readRDS.lgb.Booster
#' @title readRDS for \code{lgb.Booster} models
#' @description Attempts to load a model stored in a \code{.rds} file, using \code{\link[base]{readRDS}}
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
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 10L
#'   , valids = valids
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , early_stopping_rounds = 5L
#' )
#' model_file <- tempfile(fileext = ".rds")
#' saveRDS.lgb.Booster(model, model_file)
#' new_model <- readRDS.lgb.Booster(model_file)
#' }
#' @export
readRDS.lgb.Booster <- function(file, refhook = NULL) {

  object <- readRDS(file = file, refhook = refhook)

  # Check if object has the model stored
  if (!is.na(object$raw)) {

    # Create temporary model for the model loading
    object2 <- lgb.load(model_str = object$raw)

    # Restore best iteration and recorded evaluations
    object2$best_iter <- object$best_iter
    object2$record_evals <- object$record_evals
    object2$params <- object$params

    # Return newly loaded object
    return(object2)

  } else {

    # Return RDS loaded object
    return(object)

  }

}
