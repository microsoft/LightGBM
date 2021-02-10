#' @name lgb.unloader
#' @title Remove lightgbm and its objects from an environment
#' @description Attempts to unload LightGBM packages so you can remove objects cleanly without
#'              having to restart R. This is useful for instance if an object becomes stuck for no
#'              apparent reason and you do not want to restart R to fix the lost object.
#' @param restore Whether to reload \code{LightGBM} immediately after detaching from R.
#'                Defaults to \code{TRUE} which means automatically reload \code{LightGBM} once
#'                unloading is performed.
#' @param wipe Whether to wipe all \code{lgb.Dataset} and \code{lgb.Booster} from the global
#'             environment. Defaults to \code{FALSE} which means to not remove them.
#' @param envir The environment to perform wiping on if \code{wipe == TRUE}. Defaults to
#'              \code{.GlobalEnv} which is the global environment.
#'
#' @return NULL invisibly.
#'
#' @examples
#' \donttest{
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
#'   , nrounds = 5L
#'   , valids = valids
#'   , min_data = 1L
#'   , learning_rate = 1.0
#' )
#'
#' lgb.unloader(restore = FALSE, wipe = FALSE, envir = .GlobalEnv)
#' rm(model, dtrain, dtest) # Not needed if wipe = TRUE
#' gc() # Not needed if wipe = TRUE
#'
#' library(lightgbm)
#' # Do whatever you want again with LightGBM without object clashing
#' }
#' @export
lgb.unloader <- function(restore = TRUE, wipe = FALSE, envir = .GlobalEnv) {

  # Unload package
  try(detach("package:lightgbm", unload = TRUE), silent = TRUE)

  # Should we wipe variables? (lgb.Booster, lgb.Dataset)
  if (wipe) {
    boosters <- Filter(
      f = function(x) {
        inherits(get(x, envir = envir), "lgb.Booster")
      }
      , x = ls(envir = envir)
    )
    datasets <- Filter(
      f = function(x) {
        inherits(get(x, envir = envir), "lgb.Dataset")
      }
      , x = ls(envir = envir)
    )
    rm(list = c(boosters, datasets), envir = envir)
    gc(verbose = FALSE)
  }

  # Load package back?
  if (restore) {
    library(lightgbm)
  }

  return(invisible(NULL))

}
