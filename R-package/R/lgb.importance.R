#' Compute feature importance in a model
#'
#' Creates a \code{data.table} of feature importances in a model.
#'
#' @param model object of class \code{lgb.Booster}.
#' @param percentage whether to show importance in relative percentage.
#'
#' @return
#'
#' For a tree model, a \code{data.table} with the following columns:
#' \itemize{
#'   \item \code{Feature} Feature names in the model.
#'   \item \code{Gain} The total gain of this feature's splits.
#'   \item \code{Cover} The number of observation related to this feature.
#'   \item \code{Frequency} The number of times a feature splited in trees.
#' }
#'
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#'
#' params <- list(objective = "binary",
#'                learning_rate = 0.01, num_leaves = 63, max_depth = -1,
#'                min_data_in_leaf = 1, min_sum_hessian_in_leaf = 1)
#' model <- lgb.train(params, dtrain, 10)
#'
#' tree_imp1 <- lgb.importance(model, percentage = TRUE)
#' tree_imp2 <- lgb.importance(model, percentage = FALSE)
#'
#' @importFrom data.table := setnames setorderv
#' @export
lgb.importance <- function(model, percentage = TRUE) {

  # Check if model is a lightgbm model
  if (!inherits(model, "lgb.Booster")) {
    stop("'model' has to be an object of class lgb.Booster")
  }

  # Setup importance
  tree_dt <- lgb.model.dt.tree(model)

  # Extract elements
  tree_imp_dt <- tree_dt[
    !is.na(split_index)
    , .(Gain = sum(split_gain), Cover = sum(internal_count), Frequency = .N)
    , by = "split_feature"
  ]

  data.table::setnames(
    tree_imp_dt
    , old = "split_feature"
    , new = "Feature"
  )

  # Sort features by Gain
  data.table::setorderv(
    x = tree_imp_dt
    , cols = c("Gain")
    , order = -1
  )

  # Check if relative values are requested
  if (percentage) {
    tree_imp_dt[, ":="(Gain = Gain / sum(Gain),
                    Cover = Cover / sum(Cover),
                    Frequency = Frequency / sum(Frequency))]
  }

  # Return importance table
  return(tree_imp_dt)

}
