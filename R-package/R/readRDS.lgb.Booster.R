#' readRDS for \code{lgb.Booster} models
#'
#' Attempts to load a model using RDS.
#'
#' @param file a connection or the name of the file where the R object is saved to or read from.
#' @param refhook a hook function for handling reference objects.
#'
#' @return \code{lgb.Booster}.
#'
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- lgb.train(params,
#'                    dtrain,
#'                    10,
#'                    valids,
#'                    min_data = 1,
#'                    learning_rate = 1,
#'                    early_stopping_rounds = 5)
#' saveRDS.lgb.Booster(model, "model.rds")
#' new_model <- readRDS.lgb.Booster("model.rds")
#'
#' @export
readRDS.lgb.Booster <- function(file = "", refhook = NULL) {

  # Read RDS file
  object <- readRDS(file = file, refhook = refhook)

  # Check if object has the model stored
  if (!is.na(object$raw)) {

    # Create temporary model for the model loading
    object2 <- lgb.load(model_str = object$raw)

    # Restore best iteration and recorded evaluations
    object2$best_iter <- object$best_iter
    object2$record_evals <- object$record_evals

    # Return newly loaded object
    return(object2)

  } else {

    # Return RDS loaded object
    return(object)

  }

}
